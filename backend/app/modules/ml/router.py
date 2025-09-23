"""
Machine Learning API Router
FastAPI endpoints for AI model inference and species identification
"""
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form, BackgroundTasks
from pydantic import BaseModel
import numpy as np

from app.ml.model_manager import model_manager
from app.realtime.websocket_manager import websocket_manager
from app.realtime.analytics_engine import analytics_engine

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/ml", tags=["Machine Learning"])

# Pydantic models for requests/responses
class SpeciesIdentificationResponse(BaseModel):
    success: bool
    predictions: List[Dict[str, Any]]
    processing_time_seconds: float
    model_version: str
    timestamp: str
    confidence_threshold: float = 0.5
    image_features: Optional[Dict[str, Any]] = None

class BatchIdentificationRequest(BaseModel):
    species_type: str = "fish"
    return_top_n: int = 3
    confidence_threshold: float = 0.5

class ModelStatusResponse(BaseModel):
    loaded_models: List[str]
    model_metadata: Dict[str, Any]
    model_stats: Dict[str, Any]
    system_status: Dict[str, Any]

@router.post("/identify/species", response_model=SpeciesIdentificationResponse)
async def identify_species_from_image(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    species_type: str = Form("fish"),
    return_top_n: int = Form(3),
    model_id: str = Form("fish_primary"),
    confidence_threshold: float = Form(0.5)
):
    """
    Identify species from uploaded image using AI models
    
    Args:
        image: Image file to analyze
        species_type: Type of species (fish, otolith, etc.)
        return_top_n: Number of top predictions to return
        model_id: Model ID to use for identification
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        Species identification results
    """
    try:
        logger.info(f"Species identification request - Type: {species_type}, Model: {model_id}")
        
        # Validate file
        if not image.filename:
            raise HTTPException(status_code=400, detail="No image file provided")
        
        # Check file type
        allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/bmp", "image/tiff"]
        if image.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed types: {allowed_types}"
            )
        
        # Read image data
        image_data = await image.read()
        
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Perform identification using model manager
        result = model_manager.predict_species_sync(
            model_id=model_id,
            image=image_data,
            species_type=species_type,
            return_top_n=return_top_n
        )
        
        if not result.get('success', False):
            raise HTTPException(
                status_code=500,
                detail=f"Species identification failed: {result.get('error', 'Unknown error')}"
            )
        
        # Filter predictions by confidence threshold
        filtered_predictions = [
            pred for pred in result.get('predictions', [])
            if pred.get('confidence', 0) >= confidence_threshold
        ]
        
        # Prepare response
        response = SpeciesIdentificationResponse(
            success=True,
            predictions=filtered_predictions,
            processing_time_seconds=result.get('processing_time_seconds', 0),
            model_version=result.get('model_version', 'unknown'),
            timestamp=result.get('timestamp', datetime.now().isoformat()),
            confidence_threshold=confidence_threshold,
            image_features=result.get('image_features')
        )
        
        # Send real-time update in background
        background_tasks.add_task(
            _send_identification_update,
            result,
            image.filename,
            species_type
        )
        
        # Update analytics
        background_tasks.add_task(_update_identification_analytics, result)
        
        logger.info(f"Species identification completed - {len(filtered_predictions)} predictions")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in species identification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/identify/batch")
async def batch_species_identification(
    background_tasks: BackgroundTasks,
    images: List[UploadFile] = File(...),
    species_type: str = Form("fish"),
    return_top_n: int = Form(3),
    model_id: str = Form("fish_primary")
):
    """
    Batch species identification for multiple images
    
    Args:
        images: List of image files
        species_type: Type of species to identify
        return_top_n: Number of predictions per image
        model_id: Model to use for identification
        
    Returns:
        List of identification results
    """
    try:
        if not images:
            raise HTTPException(status_code=400, detail="No images provided")
        
        if len(images) > 20:  # Limit batch size
            raise HTTPException(status_code=400, detail="Maximum 20 images per batch")
        
        logger.info(f"Batch identification request - {len(images)} images")
        
        # Read all image data
        image_data_list = []
        for img in images:
            if img.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type in {img.filename}"
                )
            
            data = await img.read()
            image_data_list.append(data)
        
        # Perform batch identification
        results = await model_manager.batch_predict(
            model_id=model_id,
            images=image_data_list,
            species_type=species_type,
            return_top_n=return_top_n
        )
        
        # Add filename information
        for i, result in enumerate(results):
            if i < len(images):
                result['filename'] = images[i].filename
        
        # Send real-time updates in background
        background_tasks.add_task(
            _send_batch_identification_update,
            results,
            species_type
        )
        
        logger.info(f"Batch identification completed - {len(results)} results")
        return {
            'success': True,
            'batch_size': len(images),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch identification: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/status", response_model=ModelStatusResponse)
async def get_model_status():
    """Get status of all loaded ML models"""
    try:
        loaded_models = model_manager.get_loaded_models()
        model_metadata = model_manager.get_model_metadata()
        model_stats = model_manager.get_model_stats()
        system_status = model_manager.get_system_status()
        
        return ModelStatusResponse(
            loaded_models=loaded_models,
            model_metadata=model_metadata,
            model_stats=model_stats,
            system_status=system_status
        )
        
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/{model_id}/load")
async def load_model(model_id: str, model_type: str = "fish"):
    """Load a specific model"""
    try:
        success = model_manager.load_model_sync(model_id, model_type)
        
        if success:
            return {
                'success': True,
                'message': f'Model {model_id} loaded successfully',
                'model_id': model_id,
                'model_type': model_type
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f'Failed to load model {model_id}'
            )
            
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/models/{model_id}/unload")
async def unload_model(model_id: str):
    """Unload a specific model from memory"""
    try:
        success = model_manager.unload_model(model_id)
        
        if success:
            return {
                'success': True,
                'message': f'Model {model_id} unloaded successfully',
                'model_id': model_id
            }
        else:
            return {
                'success': False,
                'message': f'Model {model_id} was not loaded',
                'model_id': model_id
            }
            
    except Exception as e:
        logger.error(f"Error unloading model {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_id}/info")
async def get_model_info(model_id: str):
    """Get detailed information about a specific model"""
    try:
        model = model_manager.get_model(model_id)
        
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f'Model {model_id} not found or not loaded'
            )
        
        model_info = model.get_model_info()
        species_list = model.get_species_list()
        
        return {
            'success': True,
            'model_id': model_id,
            'model_info': model_info,
            'species_list': species_list,
            'metadata': model_manager.get_model_metadata(model_id),
            'stats': model_manager.get_model_stats(model_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info for {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/models/{model_id}/confidence-threshold")
async def update_model_confidence_threshold(model_id: str, threshold: float):
    """Update confidence threshold for a model"""
    try:
        if not 0 < threshold < 1:
            raise HTTPException(
                status_code=400,
                detail="Confidence threshold must be between 0 and 1"
            )
        
        model = model_manager.get_model(model_id)
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f'Model {model_id} not found or not loaded'
            )
        
        model.update_confidence_threshold(threshold)
        
        return {
            'success': True,
            'message': f'Confidence threshold updated for model {model_id}',
            'model_id': model_id,
            'new_threshold': threshold
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating confidence threshold: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/species/list")
async def get_identifiable_species(species_type: str = "fish", model_id: str = "fish_primary"):
    """Get list of species that can be identified by the model"""
    try:
        model = model_manager.get_model(model_id)
        if not model:
            # Try to load the model
            success = model_manager.load_model_sync(model_id, species_type)
            if not success:
                raise HTTPException(
                    status_code=404,
                    detail=f'Model {model_id} could not be loaded'
                )
            model = model_manager.get_model(model_id)
        
        species_list = model.get_species_list(species_type)
        
        return {
            'success': True,
            'model_id': model_id,
            'species_type': species_type,
            'total_species': len(species_list),
            'species': species_list
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting species list: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/performance")
async def get_ml_performance_analytics():
    """Get ML model performance analytics"""
    try:
        all_stats = model_manager.get_model_stats()
        system_status = model_manager.get_system_status()
        
        # Calculate aggregate metrics
        total_predictions = sum(stats.get('predictions_made', 0) for stats in all_stats.values())
        avg_processing_time = sum(stats.get('average_processing_time', 0) for stats in all_stats.values()) / len(all_stats) if all_stats else 0
        avg_success_rate = sum(stats.get('success_rate', 0) for stats in all_stats.values()) / len(all_stats) if all_stats else 0
        
        return {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'aggregate_metrics': {
                'total_predictions': total_predictions,
                'average_processing_time': round(avg_processing_time, 3),
                'average_success_rate': round(avg_success_rate, 3),
                'active_models': len(all_stats)
            },
            'model_stats': all_stats,
            'system_status': system_status
        }
        
    except Exception as e:
        logger.error(f"Error getting ML analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions
async def _send_identification_update(result: Dict[str, Any], filename: str, species_type: str):
    """Send real-time identification update via WebSocket"""
    try:
        update_data = {
            'filename': filename,
            'species_type': species_type,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        
        await websocket_manager.send_species_identification_update(update_data)
        
    except Exception as e:
        logger.error(f"Error sending identification update: {str(e)}")

async def _send_batch_identification_update(results: List[Dict[str, Any]], species_type: str):
    """Send batch identification updates via WebSocket"""
    try:
        update_data = {
            'batch_size': len(results),
            'species_type': species_type,
            'results_summary': {
                'successful': len([r for r in results if r.get('success', False)]),
                'failed': len([r for r in results if not r.get('success', False)])
            },
            'timestamp': datetime.now().isoformat()
        }
        
        await websocket_manager.send_species_identification_update(update_data)
        
    except Exception as e:
        logger.error(f"Error sending batch identification update: {str(e)}")

async def _update_identification_analytics(result: Dict[str, Any]):
    """Update analytics with identification results"""
    try:
        # This would update analytics engine metrics
        # For now, just log the activity
        logger.info(f"Analytics update: Species identification completed successfully")
        
    except Exception as e:
        logger.error(f"Error updating identification analytics: {str(e)}")

# Health check for ML module
@router.get("/health")
async def ml_health_check():
    """Health check for ML module"""
    try:
        system_status = model_manager.get_system_status()
        loaded_models = model_manager.get_loaded_models()
        
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'loaded_models': len(loaded_models),
            'system_status': system_status.get('average_success_rate', 0) > 0.5,
            'ml_service': 'operational'
        }
        
    except Exception as e:
        logger.error(f"ML health check failed: {str(e)}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }