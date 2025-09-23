"""
Model Management System for Ocean-Bio Platform
Handles loading, caching, and serving of AI/ML models
"""
import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

from .species_identifier import SpeciesIdentifier

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Centralized model management system
    Handles model loading, caching, and lifecycle management
    """
    
    def __init__(self, model_cache_size: int = 5):
        self.models: Dict[str, SpeciesIdentifier] = {}
        self.model_cache_size = model_cache_size
        self.model_metadata: Dict[str, Dict] = {}
        self.last_access_times: Dict[str, datetime] = {}
        self.model_stats: Dict[str, Dict] = {}
        
        # Initialize default models
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default models for all 5 data types"""
        try:
            logger.info("Initializing AI models for all 5 data types...")
            
            # Load models for all 5 data types
            self.load_model_sync('edna_primary', model_type='edna')
            self.load_model_sync('oceanographic_primary', model_type='oceanographic')
            self.load_model_sync('otolith_primary', model_type='otolith')
            self.load_model_sync('taxonomy_primary', model_type='taxonomy')
            self.load_model_sync('fisheries_primary', model_type='fisheries')
            
            logger.info(f"Initialized {len(self.models)} models for all data types")
            
        except Exception as e:
            logger.error(f"Error initializing default models: {str(e)}")
    
    async def load_model(
        self, 
        model_id: str, 
        model_type: str = 'fish',
        model_path: Optional[str] = None
    ) -> bool:
        """
        Load a model asynchronously
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model ('fish', 'otolith', etc.)
            model_path: Optional custom path to model files
            
        Returns:
            True if model loaded successfully
        """
        try:
            logger.info(f"Loading model: {model_id} (type: {model_type})")
            
            # Check cache size and evict if necessary
            if len(self.models) >= self.model_cache_size:
                await self._evict_least_used_model()
            
            # Create species identifier instance
            if model_path:
                identifier = SpeciesIdentifier(model_path=model_path)
            else:
                identifier = SpeciesIdentifier()
            
            # Store model and metadata
            self.models[model_id] = identifier
            self.last_access_times[model_id] = datetime.now()
            
            # Get model information
            model_info = identifier.get_model_info()
            self.model_metadata[model_id] = {
                'model_type': model_type,
                'loaded_at': datetime.now().isoformat(),
                'model_info': model_info,
                'usage_count': 0
            }
            
            # Initialize stats
            self.model_stats[model_id] = {
                'predictions_made': 0,
                'total_processing_time': 0.0,
                'average_processing_time': 0.0,
                'last_used': None,
                'success_rate': 1.0,
                'error_count': 0
            }
            
            logger.info(f"Successfully loaded model: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {str(e)}")
            return False
    
    def load_model_sync(
        self, 
        model_id: str, 
        model_type: str = 'fish',
        model_path: Optional[str] = None
    ) -> bool:
        """
        Synchronous version of load_model
        """
        try:
            logger.info(f"Loading model synchronously: {model_id}")
            
            # Check cache size and evict if necessary
            if len(self.models) >= self.model_cache_size:
                self._evict_least_used_model_sync()
            
            # Create species identifier instance
            if model_path:
                identifier = SpeciesIdentifier(model_path=model_path)
            else:
                identifier = SpeciesIdentifier()
            
            # Store model and metadata
            self.models[model_id] = identifier
            self.last_access_times[model_id] = datetime.now()
            
            # Get model information
            model_info = identifier.get_model_info()
            self.model_metadata[model_id] = {
                'model_type': model_type,
                'loaded_at': datetime.now().isoformat(),
                'model_info': model_info,
                'usage_count': 0
            }
            
            # Initialize stats
            self.model_stats[model_id] = {
                'predictions_made': 0,
                'total_processing_time': 0.0,
                'average_processing_time': 0.0,
                'last_used': None,
                'success_rate': 1.0,
                'error_count': 0
            }
            
            logger.info(f"Successfully loaded model synchronously: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {str(e)}")
            return False
    
    def get_model(self, model_id: str) -> Optional[SpeciesIdentifier]:
        """
        Get a loaded model by ID
        
        Args:
            model_id: Model identifier
            
        Returns:
            SpeciesIdentifier instance or None
        """
        if model_id in self.models:
            # Update access time
            self.last_access_times[model_id] = datetime.now()
            self.model_metadata[model_id]['usage_count'] += 1
            
            return self.models[model_id]
        
        logger.warning(f"Model {model_id} not found in cache")
        return None
    
    async def predict_species(
        self, 
        model_id: str, 
        image: Any, 
        species_type: str = 'fish',
        return_top_n: int = 3
    ) -> Dict[str, Any]:
        """
        Make species prediction using specified model
        
        Args:
            model_id: Model to use for prediction
            image: Input image
            species_type: Type of species to identify
            return_top_n: Number of top predictions
            
        Returns:
            Prediction results
        """
        try:
            start_time = datetime.now()
            
            # Get model
            model = self.get_model(model_id)
            if not model:
                # Try to load default model
                await self.load_model(model_id, species_type)
                model = self.get_model(model_id)
                
                if not model:
                    return {
                        'success': False,
                        'error': f'Model {model_id} not available',
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Make prediction
            result = model.identify_species(image, species_type, return_top_n)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_model_stats(model_id, processing_time, result.get('success', False))
            
            # Add model information to result
            result['model_id'] = model_id
            result['model_metadata'] = self.model_metadata.get(model_id, {})
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction with model {model_id}: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_model_stats(model_id, processing_time, False)
            
            return {
                'success': False,
                'error': str(e),
                'model_id': model_id,
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_species_sync(
        self, 
        model_id: str, 
        image: Any, 
        species_type: str = 'fish',
        return_top_n: int = 3
    ) -> Dict[str, Any]:
        """
        Synchronous species prediction
        """
        try:
            start_time = datetime.now()
            
            # Get model
            model = self.get_model(model_id)
            if not model:
                # Try to load default model
                self.load_model_sync(model_id, species_type)
                model = self.get_model(model_id)
                
                if not model:
                    return {
                        'success': False,
                        'error': f'Model {model_id} not available',
                        'timestamp': datetime.now().isoformat()
                    }
            
            # Make prediction
            result = model.identify_species(image, species_type, return_top_n)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_model_stats(model_id, processing_time, result.get('success', False))
            
            # Add model information to result
            result['model_id'] = model_id
            result['model_metadata'] = self.model_metadata.get(model_id, {})
            
            return result
            
        except Exception as e:
            logger.error(f"Error in synchronous prediction with model {model_id}: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_model_stats(model_id, processing_time, False)
            
            return {
                'success': False,
                'error': str(e),
                'model_id': model_id,
                'timestamp': datetime.now().isoformat()
            }
    
    async def batch_predict(
        self, 
        model_id: str, 
        images: List[Any],
        species_type: str = 'fish',
        return_top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Batch prediction for multiple images
        
        Args:
            model_id: Model to use
            images: List of images
            species_type: Type of species
            return_top_n: Top predictions per image
            
        Returns:
            List of prediction results
        """
        try:
            model = self.get_model(model_id)
            if not model:
                await self.load_model(model_id, species_type)
                model = self.get_model(model_id)
                
                if not model:
                    return [{
                        'success': False,
                        'error': f'Model {model_id} not available',
                        'timestamp': datetime.now().isoformat()
                    }]
            
            # Process batch
            results = model.batch_identify(images, species_type, return_top_n)
            
            # Add model metadata to each result
            for result in results:
                result['model_id'] = model_id
                result['model_metadata'] = self.model_metadata.get(model_id, {})
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")
            return [{
                'success': False,
                'error': str(e),
                'model_id': model_id,
                'timestamp': datetime.now().isoformat()
            }]
    
    def _update_model_stats(self, model_id: str, processing_time: float, success: bool):
        """Update model performance statistics"""
        if model_id not in self.model_stats:
            return
        
        stats = self.model_stats[model_id]
        
        # Update counters
        stats['predictions_made'] += 1
        stats['total_processing_time'] += processing_time
        stats['average_processing_time'] = stats['total_processing_time'] / stats['predictions_made']
        stats['last_used'] = datetime.now().isoformat()
        
        # Update success rate
        if success:
            current_success_count = stats['success_rate'] * (stats['predictions_made'] - 1)
            stats['success_rate'] = (current_success_count + 1) / stats['predictions_made']
        else:
            stats['error_count'] += 1
            current_success_count = stats['success_rate'] * (stats['predictions_made'] - 1)
            stats['success_rate'] = current_success_count / stats['predictions_made']
    
    async def _evict_least_used_model(self):
        """Evict least recently used model from cache"""
        if not self.models:
            return
        
        # Find least recently used model
        lru_model_id = min(self.last_access_times.keys(), 
                          key=lambda k: self.last_access_times[k])
        
        logger.info(f"Evicting least used model: {lru_model_id}")
        
        # Remove from caches
        del self.models[lru_model_id]
        del self.last_access_times[lru_model_id]
        
        if lru_model_id in self.model_metadata:
            del self.model_metadata[lru_model_id]
    
    def _evict_least_used_model_sync(self):
        """Synchronous version of model eviction"""
        if not self.models:
            return
        
        # Find least recently used model
        lru_model_id = min(self.last_access_times.keys(), 
                          key=lambda k: self.last_access_times[k])
        
        logger.info(f"Evicting least used model: {lru_model_id}")
        
        # Remove from caches
        del self.models[lru_model_id]
        del self.last_access_times[lru_model_id]
        
        if lru_model_id in self.model_metadata:
            del self.model_metadata[lru_model_id]
    
    def unload_model(self, model_id: str) -> bool:
        """
        Manually unload a model from cache
        
        Args:
            model_id: Model to unload
            
        Returns:
            True if model was unloaded
        """
        try:
            if model_id in self.models:
                del self.models[model_id]
                del self.last_access_times[model_id]
                
                if model_id in self.model_metadata:
                    del self.model_metadata[model_id]
                
                if model_id in self.model_stats:
                    del self.model_stats[model_id]
                
                logger.info(f"Unloaded model: {model_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error unloading model {model_id}: {str(e)}")
            return False
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded model IDs"""
        return list(self.models.keys())
    
    def get_model_metadata(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata for models
        
        Args:
            model_id: Specific model ID or None for all models
            
        Returns:
            Model metadata
        """
        if model_id:
            return self.model_metadata.get(model_id, {})
        else:
            return self.model_metadata
    
    def get_model_stats(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance statistics for models
        
        Args:
            model_id: Specific model ID or None for all models
            
        Returns:
            Model statistics
        """
        if model_id:
            return self.model_stats.get(model_id, {})
        else:
            return self.model_stats
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and health"""
        total_predictions = sum(stats['predictions_made'] for stats in self.model_stats.values())
        avg_success_rate = sum(stats['success_rate'] for stats in self.model_stats.values()) / len(self.model_stats) if self.model_stats else 0
        
        return {
            'loaded_models': len(self.models),
            'cache_size_limit': self.model_cache_size,
            'cache_usage': f"{len(self.models)}/{self.model_cache_size}",
            'total_predictions_made': total_predictions,
            'average_success_rate': round(avg_success_rate, 3),
            'system_uptime': datetime.now().isoformat(),
            'available_model_types': ['edna', 'oceanographic', 'otolith', 'taxonomy', 'fisheries'],
            'model_details': {
                model_id: {
                    'type': metadata.get('model_type', 'unknown'),
                    'loaded_at': metadata.get('loaded_at', 'unknown'),
                    'usage_count': metadata.get('usage_count', 0),
                    'last_access': self.last_access_times.get(model_id, datetime.now()).isoformat()
                }
                for model_id, metadata in self.model_metadata.items()
            }
        }
    
    def cleanup_expired_models(self, max_idle_hours: int = 24):
        """
        Remove models that haven't been used for specified hours
        
        Args:
            max_idle_hours: Maximum idle time before cleanup
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_idle_hours)
            expired_models = []
            
            for model_id, last_access in self.last_access_times.items():
                if last_access < cutoff_time:
                    expired_models.append(model_id)
            
            for model_id in expired_models:
                self.unload_model(model_id)
                logger.info(f"Cleaned up expired model: {model_id}")
            
            if expired_models:
                logger.info(f"Cleaned up {len(expired_models)} expired models")
            
        except Exception as e:
            logger.error(f"Error during model cleanup: {str(e)}")
    
    def reload_all_models(self):
        """Reload all currently loaded models"""
        try:
            model_ids = list(self.models.keys())
            model_types = [self.model_metadata[mid].get('model_type', 'fish') for mid in model_ids]
            
            # Unload all models
            for model_id in model_ids:
                self.unload_model(model_id)
            
            # Reload models
            for model_id, model_type in zip(model_ids, model_types):
                self.load_model_sync(model_id, model_type)
            
            logger.info(f"Reloaded {len(model_ids)} models")
            
        except Exception as e:
            logger.error(f"Error reloading models: {str(e)}")

# Global model manager instance
model_manager = ModelManager()