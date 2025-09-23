"""
📊 Predictive Modeling API Router

RESTful API endpoints for predictive analytics, stock assessment, and forecasting functionality.
Provides access to machine learning models, statistical analysis, and trend prediction.

Author: Ocean-Bio Development Team
Version: 2.0.0
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime, date
import logging

from ....predictive.stock_assessment import stock_assessor
from ....predictive.forecasting_engine import forecasting_engine
from ....predictive.trend_analysis import trend_analyzer
from ....core.database import get_db
from ....core.auth import get_current_user
from ....models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictive", tags=["predictive"])

# Pydantic models for API requests/responses

class StockAssessmentRequest(BaseModel):
    """Stock assessment request parameters"""
    species_name: str = Field(..., description="Species name for assessment")
    data_source: str = Field(..., description="Data source identifier")
    model_type: str = Field("surplus_production", description="Assessment model type")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")
    time_series_data: List[Dict[str, Any]] = Field(..., description="Historical data")

class ForecastingRequest(BaseModel):
    """Forecasting request parameters"""
    forecast_type: str = Field(..., description="Type of forecast")
    time_series_data: List[Dict[str, Any]] = Field(..., description="Historical time series")
    forecast_horizon: int = Field(12, ge=1, le=120, description="Number of periods to forecast")
    confidence_level: float = Field(0.95, ge=0.5, le=0.99, description="Confidence level")
    model_parameters: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")

class TrendAnalysisRequest(BaseModel):
    """Trend analysis request parameters"""
    analysis_type: str = Field(..., description="Type of trend analysis")
    data: List[Dict[str, Any]] = Field(..., description="Data for analysis")
    variables: List[str] = Field(..., description="Variables to analyze")
    time_column: str = Field("date", description="Time column name")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Analysis parameters")

class ModelTrainingRequest(BaseModel):
    """Model training request parameters"""
    model_type: str = Field(..., description="Type of model to train")
    training_data: List[Dict[str, Any]] = Field(..., description="Training dataset")
    features: List[str] = Field(..., description="Feature columns")
    target: str = Field(..., description="Target variable")
    validation_split: float = Field(0.2, ge=0.1, le=0.5, description="Validation split ratio")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")

class PredictionRequest(BaseModel):
    """Prediction request parameters"""
    model_id: str = Field(..., description="Model ID for predictions")
    input_data: List[Dict[str, Any]] = Field(..., description="Input data for predictions")
    return_probabilities: bool = Field(False, description="Return prediction probabilities")

# API Endpoints

@router.get("/health")
async def predictive_health_check():
    """Health check for predictive services"""
    try:
        # Test stock assessor
        stock_status = stock_assessor.get_status()
        
        # Test forecasting engine
        forecasting_status = forecasting_engine.get_status()
        
        # Test trend analyzer
        trend_status = trend_analyzer.get_status()
        
        return {
            "status": "healthy",
            "services": {
                "stock_assessment": stock_status,
                "forecasting_engine": forecasting_status,
                "trend_analysis": trend_status
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Predictive health check failed: {e}")
        raise HTTPException(status_code=503, detail="Predictive services unavailable")

@router.post("/stock/assess")
async def perform_stock_assessment(
    request: StockAssessmentRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Perform stock assessment analysis"""
    try:
        logger.info(f"📊 Stock assessment: {request.species_name} using {request.model_type}")
        
        # Validate model type
        available_models = stock_assessor.get_available_models()
        if request.model_type not in available_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model type. Available: {available_models}"
            )
        
        # Perform assessment
        assessment_results = await stock_assessor.assess_stock(
            species_name=request.species_name,
            data_source=request.data_source,
            model_type=request.model_type,
            time_series_data=request.time_series_data,
            parameters=request.parameters
        )
        
        return {
            "species_name": request.species_name,
            "model_type": request.model_type,
            "assessment_results": assessment_results,
            "data_points": len(request.time_series_data),
            "assessment_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Stock assessment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/forecast/generate")
async def generate_forecast(
    request: ForecastingRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Generate forecasts using time series analysis"""
    try:
        logger.info(f"🔮 Forecast generation: {request.forecast_type}, horizon={request.forecast_horizon}")
        
        # Validate forecast type
        available_types = forecasting_engine.get_available_forecast_types()
        if request.forecast_type not in available_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid forecast type. Available: {available_types}"
            )
        
        # Generate forecast
        forecast_results = await forecasting_engine.generate_forecast(
            forecast_type=request.forecast_type,
            time_series_data=request.time_series_data,
            forecast_horizon=request.forecast_horizon,
            confidence_level=request.confidence_level,
            parameters=request.model_parameters
        )
        
        return {
            "forecast_type": request.forecast_type,
            "forecast_horizon": request.forecast_horizon,
            "confidence_level": request.confidence_level,
            "forecast_results": forecast_results,
            "input_data_points": len(request.time_series_data),
            "forecast_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Forecast generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trends/analyze")
async def analyze_trends(
    request: TrendAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Perform trend analysis on time series data"""
    try:
        logger.info(f"📈 Trend analysis: {request.analysis_type}")
        
        # Validate analysis type
        available_analyses = trend_analyzer.get_available_analyses()
        if request.analysis_type not in available_analyses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid analysis type. Available: {available_analyses}"
            )
        
        # Perform trend analysis
        trend_results = await trend_analyzer.analyze_trends(
            analysis_type=request.analysis_type,
            data=request.data,
            variables=request.variables,
            time_column=request.time_column,
            parameters=request.parameters
        )
        
        return {
            "analysis_type": request.analysis_type,
            "variables_analyzed": request.variables,
            "trend_results": trend_results,
            "data_points": len(request.data),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Trend analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/train")
async def train_model(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Train a predictive model"""
    try:
        logger.info(f"🎯 Model training: {request.model_type}")
        
        # Start model training (async task)
        training_task = await forecasting_engine.train_model(
            model_type=request.model_type,
            training_data=request.training_data,
            features=request.features,
            target=request.target,
            validation_split=request.validation_split,
            hyperparameters=request.hyperparameters
        )
        
        return {
            "training_task_id": training_task.get("task_id"),
            "model_type": request.model_type,
            "training_data_size": len(request.training_data),
            "features": request.features,
            "target": request.target,
            "status": "training_started",
            "training_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Model training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/status/{task_id}")
async def get_training_status(
    task_id: str = Path(..., description="Training task ID"),
    current_user: User = Depends(get_current_user)
):
    """Get model training status"""
    try:
        status = await forecasting_engine.get_training_status(task_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Training task not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/predict")
async def make_predictions(
    request: PredictionRequest,
    current_user: User = Depends(get_current_user)
):
    """Make predictions using a trained model"""
    try:
        logger.info(f"🔍 Making predictions with model: {request.model_id}")
        
        # Make predictions
        predictions = await forecasting_engine.predict(
            model_id=request.model_id,
            input_data=request.input_data,
            return_probabilities=request.return_probabilities
        )
        
        return {
            "model_id": request.model_id,
            "predictions": predictions,
            "input_samples": len(request.input_data),
            "prediction_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/available")
async def get_available_models(current_user: User = Depends(get_current_user)):
    """Get list of available predictive models"""
    try:
        models = {
            "stock_assessment": stock_assessor.get_available_models(),
            "forecasting": forecasting_engine.get_available_models(),
            "trend_analysis": trend_analyzer.get_available_analyses()
        }
        
        return {
            "available_models": models,
            "total_models": sum(len(v) for v in models.values()),
            "retrieved_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get available models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_id}")
async def get_model_details(
    model_id: str = Path(..., description="Model ID"),
    current_user: User = Depends(get_current_user)
):
    """Get details of a specific model"""
    try:
        model_details = await forecasting_engine.get_model_details(model_id)
        
        if not model_details:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return model_details
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get model details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str = Path(..., description="Model ID to delete"),
    current_user: User = Depends(get_current_user)
):
    """Delete a trained model"""
    try:
        success = await forecasting_engine.delete_model(model_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return {
            "message": "Model deleted successfully",
            "model_id": model_id,
            "deletion_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stock/species/{species_name}/history")
async def get_species_assessment_history(
    species_name: str = Path(..., description="Species name"),
    limit: int = Query(50, ge=1, le=500, description="Number of records to return"),
    current_user: User = Depends(get_current_user)
):
    """Get assessment history for a species"""
    try:
        history = await stock_assessor.get_assessment_history(
            species_name=species_name,
            limit=limit
        )
        
        return {
            "species_name": species_name,
            "assessment_history": history,
            "total_assessments": len(history),
            "retrieved_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get assessment history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/forecast/accuracy/{model_id}")
async def get_forecast_accuracy(
    model_id: str = Path(..., description="Model ID"),
    current_user: User = Depends(get_current_user)
):
    """Get forecast accuracy metrics for a model"""
    try:
        accuracy_metrics = await forecasting_engine.get_accuracy_metrics(model_id)
        
        if not accuracy_metrics:
            raise HTTPException(status_code=404, detail="Model not found or no accuracy data available")
        
        return accuracy_metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get accuracy metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate/data")
async def validate_input_data(
    data: List[Dict[str, Any]],
    data_type: str = Query(..., description="Type of data to validate"),
    current_user: User = Depends(get_current_user)
):
    """Validate input data for predictive models"""
    try:
        logger.info(f"✅ Data validation: {data_type}")
        
        validation_result = await forecasting_engine.validate_data(
            data=data,
            data_type=data_type
        )
        
        return {
            "data_type": data_type,
            "validation_result": validation_result,
            "data_points": len(data),
            "validation_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Data validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reports/summary")
async def get_predictive_summary(
    date_from: Optional[date] = Query(None, description="Start date for summary"),
    date_to: Optional[date] = Query(None, description="End date for summary"),
    current_user: User = Depends(get_current_user)
):
    """Get summary of predictive analytics activities"""
    try:
        summary = await forecasting_engine.get_activity_summary(
            date_from=date_from,
            date_to=date_to
        )
        
        return {
            "summary": summary,
            "date_range": {
                "from": date_from.isoformat() if date_from else None,
                "to": date_to.isoformat() if date_to else None
            },
            "summary_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints

@router.get("/parameters/{model_type}")
async def get_model_parameters(
    model_type: str = Path(..., description="Model type"),
    current_user: User = Depends(get_current_user)
):
    """Get default parameters for a model type"""
    try:
        parameters = forecasting_engine.get_model_parameters(model_type)
        
        if not parameters:
            raise HTTPException(status_code=404, detail="Model type not found")
        
        return {
            "model_type": model_type,
            "parameters": parameters,
            "retrieved_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get model parameters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate/model")
async def evaluate_model_performance(
    model_id: str = Query(..., description="Model ID to evaluate"),
    test_data: List[Dict[str, Any]] = ...,
    evaluation_metrics: List[str] = Query(default=["mse", "mae", "r2"], description="Metrics to calculate"),
    current_user: User = Depends(get_current_user)
):
    """Evaluate model performance on test data"""
    try:
        logger.info(f"📊 Model evaluation: {model_id}")
        
        evaluation_results = await forecasting_engine.evaluate_model(
            model_id=model_id,
            test_data=test_data,
            metrics=evaluation_metrics
        )
        
        return {
            "model_id": model_id,
            "evaluation_results": evaluation_results,
            "test_data_size": len(test_data),
            "metrics_calculated": evaluation_metrics,
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Model evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))