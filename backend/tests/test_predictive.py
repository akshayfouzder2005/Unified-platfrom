"""
📈 Comprehensive Test Suite for Predictive Modeling Components

Unit and integration tests for stock assessment, forecasting, and ML model management.
Tests model training, prediction accuracy, and data pipeline functionality.

Author: Ocean-Bio Development Team
Version: 2.0.0
"""

import pytest
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
import asyncio
from sklearn.metrics import mean_absolute_error, r2_score

# Import the modules under test
from backend.app.predictive.stock_assessment import StockAssessmentEngine
from backend.app.predictive.forecasting import ForecastingService
from backend.app.predictive.model_management import ModelManager
from backend.app.predictive.trend_analysis import TrendAnalyzer

class TestStockAssessmentEngine:
    """Test suite for stock assessment engine"""
    
    @pytest.fixture
    def stock_engine(self):
        """Create stock assessment engine instance for testing"""
        return StockAssessmentEngine()
    
    @pytest.fixture
    def sample_fish_data(self):
        """Sample fish stock data for testing"""
        dates = pd.date_range('2020-01-01', '2024-09-01', freq='M')
        return pd.DataFrame({
            'date': dates,
            'species': 'Pomfret' * len(dates),
            'biomass': np.random.normal(1000, 200, len(dates)),
            'catch_rate': np.random.normal(50, 10, len(dates)),
            'fishing_mortality': np.random.normal(0.3, 0.05, len(dates)),
            'recruitment': np.random.normal(100, 20, len(dates)),
            'spawning_biomass': np.random.normal(800, 150, len(dates))
        })
    
    @pytest.fixture
    def sample_assessment_params(self):
        """Sample parameters for stock assessment"""
        return {
            'species': 'Pomfret',
            'assessment_type': 'surplus_production',
            'time_period': {
                'start_date': '2020-01-01',
                'end_date': '2024-09-01'
            },
            'biological_parameters': {
                'natural_mortality': 0.2,
                'max_age': 15,
                'length_weight_a': 0.01,
                'length_weight_b': 3.0,
                'maturity_length': 25.0
            },
            'fishery_parameters': {
                'selectivity_type': 'logistic',
                'selectivity_params': {'L50': 20.0, 'L95': 30.0}
            }
        }
    
    def test_get_available_species(self, stock_engine):
        """Test getting available species for assessment"""
        species_list = stock_engine.get_available_species()
        
        assert isinstance(species_list, list)
        assert len(species_list) > 0
        
        # Check species structure
        for species in species_list:
            assert 'name' in species
            assert 'scientific_name' in species
            assert 'stock_status' in species
    
    def test_get_assessment_types(self, stock_engine):
        """Test getting available assessment types"""
        assessment_types = stock_engine.get_assessment_types()
        
        assert isinstance(assessment_types, list)
        assert 'surplus_production' in assessment_types
        assert 'virtual_population_analysis' in assessment_types
        assert 'statistical_catch_at_age' in assessment_types
    
    @pytest.mark.asyncio
    async def test_perform_assessment(self, stock_engine, sample_fish_data, sample_assessment_params):
        """Test stock assessment performance"""
        with patch.object(stock_engine, '_load_species_data') as mock_load:
            mock_load.return_value = sample_fish_data
            
            result = await stock_engine.perform_assessment(**sample_assessment_params)
            
            assert 'species' in result
            assert result['species'] == 'Pomfret'
            assert 'assessment_type' in result
            assert 'current_biomass' in result
            assert 'fishing_mortality' in result
            assert 'stock_status' in result
            assert 'recommendations' in result
            assert 'confidence_intervals' in result
    
    @pytest.mark.asyncio
    async def test_calculate_reference_points(self, stock_engine, sample_fish_data):
        """Test biological reference points calculation"""
        reference_points = await stock_engine._calculate_reference_points(
            data=sample_fish_data,
            biological_params={'natural_mortality': 0.2}
        )
        
        assert 'MSY' in reference_points  # Maximum Sustainable Yield
        assert 'BMSY' in reference_points  # Biomass at MSY
        assert 'FMSY' in reference_points  # Fishing mortality at MSY
        assert 'B0' in reference_points   # Virgin biomass
        
        # Check that values are reasonable
        assert reference_points['MSY'] > 0
        assert reference_points['BMSY'] > 0
        assert reference_points['FMSY'] > 0
    
    def test_stock_status_classification(self, stock_engine):
        """Test stock status classification"""
        # Test overfished stock
        overfished_result = stock_engine._classify_stock_status(
            current_biomass=300, 
            bmsy=1000,
            current_f=0.8,
            fmsy=0.3
        )
        assert overfished_result['status'] == 'overfished'
        assert overfished_result['overfishing'] is True
        
        # Test healthy stock
        healthy_result = stock_engine._classify_stock_status(
            current_biomass=1200,
            bmsy=1000,
            current_f=0.2,
            fmsy=0.3
        )
        assert healthy_result['status'] == 'healthy'
        assert healthy_result['overfishing'] is False
    
    @pytest.mark.asyncio
    async def test_generate_recommendations(self, stock_engine):
        """Test management recommendations generation"""
        stock_status = {
            'status': 'overfished',
            'overfishing': True,
            'biomass_ratio': 0.3,  # B/BMSY
            'f_ratio': 2.5         # F/FMSY
        }
        
        recommendations = await stock_engine._generate_recommendations(stock_status)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check recommendation structure
        for rec in recommendations:
            assert 'category' in rec
            assert 'description' in rec
            assert 'priority' in rec
            assert 'timeframe' in rec


class TestForecastingService:
    """Test suite for forecasting service"""
    
    @pytest.fixture
    def forecasting_service(self):
        """Create forecasting service instance for testing"""
        return ForecastingService()
    
    @pytest.fixture
    def sample_time_series(self):
        """Sample time series data for forecasting"""
        dates = pd.date_range('2020-01-01', '2024-08-01', freq='M')
        # Create synthetic data with trend and seasonality
        trend = np.linspace(100, 150, len(dates))
        seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
        noise = np.random.normal(0, 5, len(dates))
        values = trend + seasonal + noise
        
        return pd.DataFrame({
            'date': dates,
            'value': values,
            'species': 'Test Species',
            'location': 'Arabian Sea'
        })
    
    @pytest.fixture
    def forecast_params(self):
        """Sample forecasting parameters"""
        return {
            'forecast_horizon': 12,  # months
            'model_type': 'arima',
            'confidence_interval': 0.95,
            'include_external_factors': True,
            'external_variables': ['sea_temperature', 'precipitation']
        }
    
    def test_get_available_models(self, forecasting_service):
        """Test getting available forecasting models"""
        models = forecasting_service.get_available_models()
        
        assert isinstance(models, list)
        assert 'arima' in models
        assert 'exponential_smoothing' in models
        assert 'prophet' in models
        assert 'lstm' in models
    
    def test_get_model_info(self, forecasting_service):
        """Test getting model information"""
        info = forecasting_service.get_model_info('arima')
        
        assert 'name' in info
        assert 'description' in info
        assert 'parameters' in info
        assert 'suitable_for' in info
        assert 'requirements' in info
    
    @pytest.mark.asyncio
    async def test_create_forecast(self, forecasting_service, sample_time_series, forecast_params):
        """Test forecast creation"""
        result = await forecasting_service.create_forecast(
            data=sample_time_series,
            target_column='value',
            **forecast_params
        )
        
        assert 'forecast_id' in result
        assert 'model_type' in result
        assert 'forecast_values' in result
        assert 'confidence_bounds' in result
        assert 'model_metrics' in result
        
        # Check forecast structure
        forecast_values = result['forecast_values']
        assert len(forecast_values) == forecast_params['forecast_horizon']
        
        # Check confidence bounds
        confidence_bounds = result['confidence_bounds']
        assert 'lower' in confidence_bounds
        assert 'upper' in confidence_bounds
        assert len(confidence_bounds['lower']) == forecast_params['forecast_horizon']
    
    @pytest.mark.asyncio
    async def test_batch_forecast(self, forecasting_service, sample_time_series):
        """Test batch forecasting for multiple series"""
        # Create multiple series
        series_data = []
        for species in ['Species A', 'Species B', 'Species C']:
            series = sample_time_series.copy()
            series['species'] = species
            series['value'] = series['value'] * np.random.uniform(0.8, 1.2)
            series_data.append(series)
        
        batch_params = {
            'forecast_horizon': 6,
            'model_type': 'exponential_smoothing'
        }
        
        result = await forecasting_service.batch_forecast(
            series_list=series_data,
            target_column='value',
            group_by='species',
            **batch_params
        )
        
        assert 'batch_id' in result
        assert 'forecasts' in result
        assert len(result['forecasts']) == 3
        
        # Check individual forecasts
        for forecast in result['forecasts']:
            assert 'series_id' in forecast
            assert 'forecast_values' in forecast
    
    def test_model_validation(self, forecasting_service, sample_time_series):
        """Test model validation functionality"""
        validation_result = forecasting_service._validate_model(
            data=sample_time_series,
            target_column='value',
            model_type='arima',
            test_size=0.2
        )
        
        assert 'mae' in validation_result
        assert 'rmse' in validation_result
        assert 'mape' in validation_result
        assert 'r2_score' in validation_result
        
        # Check that metrics are reasonable
        assert validation_result['mae'] >= 0
        assert validation_result['rmse'] >= 0
        assert validation_result['mape'] >= 0
    
    @pytest.mark.asyncio
    async def test_forecast_update(self, forecasting_service, sample_time_series):
        """Test forecast updating with new data"""
        forecast_id = 'test_forecast_001'
        
        # Add new data points
        new_dates = pd.date_range('2024-09-01', '2024-10-01', freq='M')
        new_data = pd.DataFrame({
            'date': new_dates,
            'value': [155, 158],
            'species': 'Test Species',
            'location': 'Arabian Sea'
        })
        
        with patch.object(forecasting_service, '_load_forecast') as mock_load:
            mock_forecast = {
                'model_type': 'arima',
                'parameters': {},
                'data': sample_time_series
            }
            mock_load.return_value = mock_forecast
            
            result = await forecasting_service.update_forecast(
                forecast_id=forecast_id,
                new_data=new_data,
                target_column='value'
            )
            
            assert 'forecast_id' in result
            assert result['forecast_id'] == forecast_id
            assert 'updated_forecast' in result
            assert 'model_performance' in result


class TestModelManager:
    """Test suite for model management system"""
    
    @pytest.fixture
    def model_manager(self):
        """Create model manager instance for testing"""
        return ModelManager()
    
    @pytest.fixture
    def sample_model_config(self):
        """Sample model configuration"""
        return {
            'name': 'test_fish_prediction_model',
            'type': 'regression',
            'algorithm': 'random_forest',
            'features': ['temperature', 'salinity', 'depth', 'season'],
            'target': 'fish_abundance',
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5
            },
            'validation_strategy': 'time_series_split',
            'performance_threshold': 0.8
        }
    
    @pytest.fixture
    def sample_training_data(self):
        """Sample training data"""
        n_samples = 1000
        return pd.DataFrame({
            'temperature': np.random.normal(25, 3, n_samples),
            'salinity': np.random.normal(35, 2, n_samples),
            'depth': np.random.uniform(10, 200, n_samples),
            'season': np.random.choice(['spring', 'summer', 'autumn', 'winter'], n_samples),
            'fish_abundance': np.random.lognormal(3, 1, n_samples)
        })
    
    def test_get_available_algorithms(self, model_manager):
        """Test getting available ML algorithms"""
        algorithms = model_manager.get_available_algorithms()
        
        assert isinstance(algorithms, dict)
        assert 'regression' in algorithms
        assert 'classification' in algorithms
        assert 'clustering' in algorithms
        assert 'time_series' in algorithms
        
        # Check algorithm structure
        regression_algos = algorithms['regression']
        assert 'random_forest' in regression_algos
        assert 'gradient_boosting' in regression_algos
        assert 'neural_network' in regression_algos
    
    def test_get_model_status(self, model_manager):
        """Test getting model status"""
        model_id = 'test_model_001'
        
        with patch.object(model_manager, '_load_model_metadata') as mock_load:
            mock_load.return_value = {
                'model_id': model_id,
                'status': 'trained',
                'created_at': datetime.now().isoformat(),
                'last_trained': datetime.now().isoformat(),
                'performance_metrics': {'r2_score': 0.85, 'mae': 12.3}
            }
            
            status = model_manager.get_model_status(model_id)
            
            assert status['model_id'] == model_id
            assert 'status' in status
            assert 'performance_metrics' in status
    
    @pytest.mark.asyncio
    async def test_train_model(self, model_manager, sample_model_config, sample_training_data):
        """Test model training functionality"""
        result = await model_manager.train_model(
            config=sample_model_config,
            training_data=sample_training_data
        )
        
        assert 'model_id' in result
        assert 'training_metrics' in result
        assert 'model_path' in result
        assert 'feature_importance' in result
        
        # Check training metrics
        metrics = result['training_metrics']
        assert 'train_score' in metrics
        assert 'validation_score' in metrics
        assert 'training_time' in metrics
    
    @pytest.mark.asyncio
    async def test_make_prediction(self, model_manager, sample_training_data):
        """Test model prediction functionality"""
        model_id = 'test_model_001'
        prediction_data = sample_training_data.head(10).drop('fish_abundance', axis=1)
        
        with patch.object(model_manager, '_load_trained_model') as mock_load:
            # Mock trained model
            mock_model = Mock()
            mock_model.predict.return_value = np.random.random(10) * 100
            mock_model.predict_proba = None
            mock_load.return_value = mock_model
            
            result = await model_manager.make_prediction(
                model_id=model_id,
                input_data=prediction_data
            )
            
            assert 'predictions' in result
            assert 'model_id' in result
            assert 'prediction_metadata' in result
            
            predictions = result['predictions']
            assert len(predictions) == 10
    
    @pytest.mark.asyncio
    async def test_batch_prediction(self, model_manager):
        """Test batch prediction functionality"""
        model_id = 'test_model_001'
        batch_data = [
            {'temperature': 24.5, 'salinity': 35.2, 'depth': 50, 'season': 'summer'},
            {'temperature': 22.1, 'salinity': 34.8, 'depth': 75, 'season': 'winter'},
            {'temperature': 26.3, 'salinity': 35.5, 'depth': 30, 'season': 'spring'}
        ]
        
        with patch.object(model_manager, '_load_trained_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([45.2, 32.1, 58.7])
            mock_load.return_value = mock_model
            
            result = await model_manager.batch_prediction(
                model_id=model_id,
                batch_data=batch_data
            )
            
            assert 'batch_id' in result
            assert 'predictions' in result
            assert len(result['predictions']) == 3
    
    def test_model_evaluation(self, model_manager, sample_training_data):
        """Test model evaluation functionality"""
        model_id = 'test_model_001'
        test_data = sample_training_data.tail(100)
        
        with patch.object(model_manager, '_load_trained_model') as mock_load:
            mock_model = Mock()
            # Generate predictions with some correlation to actual values
            actual_values = test_data['fish_abundance'].values
            predictions = actual_values + np.random.normal(0, actual_values * 0.1)
            mock_model.predict.return_value = predictions
            mock_load.return_value = mock_model
            
            evaluation = model_manager.evaluate_model(
                model_id=model_id,
                test_data=test_data,
                target_column='fish_abundance'
            )
            
            assert 'model_id' in evaluation
            assert 'metrics' in evaluation
            assert 'feature_importance' in evaluation
            
            metrics = evaluation['metrics']
            assert 'r2_score' in metrics
            assert 'mae' in metrics
            assert 'rmse' in metrics
    
    @pytest.mark.asyncio
    async def test_retrain_model(self, model_manager, sample_model_config, sample_training_data):
        """Test model retraining functionality"""
        model_id = 'test_model_001'
        
        # Add some new data
        new_data = sample_training_data.tail(100).copy()
        new_data['fish_abundance'] = new_data['fish_abundance'] * 1.1  # Slight shift
        
        with patch.object(model_manager, '_load_model_config') as mock_config:
            mock_config.return_value = sample_model_config
            
            result = await model_manager.retrain_model(
                model_id=model_id,
                new_data=new_data,
                retrain_strategy='incremental'
            )
            
            assert 'model_id' in result
            assert result['model_id'] == model_id
            assert 'retraining_metrics' in result
            assert 'performance_comparison' in result


class TestTrendAnalyzer:
    """Test suite for trend analysis functionality"""
    
    @pytest.fixture
    def trend_analyzer(self):
        """Create trend analyzer instance for testing"""
        return TrendAnalyzer()
    
    @pytest.fixture
    def sample_trend_data(self):
        """Sample data for trend analysis"""
        dates = pd.date_range('2020-01-01', '2024-09-01', freq='M')
        # Create data with multiple trend components
        linear_trend = np.linspace(50, 80, len(dates))
        seasonal = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
        cyclical = 3 * np.sin(2 * np.pi * np.arange(len(dates)) / 36)  # 3-year cycle
        noise = np.random.normal(0, 2, len(dates))
        
        return pd.DataFrame({
            'date': dates,
            'biomass': linear_trend + seasonal + cyclical + noise,
            'temperature': np.random.normal(25, 3, len(dates)),
            'fishing_pressure': np.random.uniform(0.1, 0.8, len(dates)),
            'species': 'Test Species'
        })
    
    def test_get_available_analyses(self, trend_analyzer):
        """Test getting available trend analyses"""
        analyses = trend_analyzer.get_available_analyses()
        
        assert isinstance(analyses, list)
        assert 'linear_trend' in analyses
        assert 'seasonal_decomposition' in analyses
        assert 'changepoint_detection' in analyses
        assert 'correlation_analysis' in analyses
    
    def test_get_analysis_info(self, trend_analyzer):
        """Test getting analysis information"""
        info = trend_analyzer.get_analysis_info('seasonal_decomposition')
        
        assert 'name' in info
        assert 'description' in info
        assert 'parameters' in info
        assert 'output_description' in info
    
    @pytest.mark.asyncio
    async def test_linear_trend_analysis(self, trend_analyzer, sample_trend_data):
        """Test linear trend analysis"""
        result = await trend_analyzer.analyze_trend(
            data=sample_trend_data,
            value_column='biomass',
            analysis_type='linear_trend'
        )
        
        assert 'analysis_type' in result
        assert result['analysis_type'] == 'linear_trend'
        assert 'trend_parameters' in result
        assert 'statistical_tests' in result
        assert 'trend_strength' in result
        
        # Check trend parameters
        trend_params = result['trend_parameters']
        assert 'slope' in trend_params
        assert 'intercept' in trend_params
        assert 'r_squared' in trend_params
        assert 'p_value' in trend_params
    
    @pytest.mark.asyncio
    async def test_seasonal_decomposition(self, trend_analyzer, sample_trend_data):
        """Test seasonal decomposition analysis"""
        result = await trend_analyzer.analyze_trend(
            data=sample_trend_data,
            value_column='biomass',
            analysis_type='seasonal_decomposition',
            parameters={'period': 12}
        )
        
        assert 'analysis_type' in result
        assert result['analysis_type'] == 'seasonal_decomposition'
        assert 'components' in result
        assert 'seasonal_strength' in result
        assert 'trend_strength' in result
        
        components = result['components']
        assert 'trend' in components
        assert 'seasonal' in components
        assert 'residual' in components
        assert len(components['trend']) == len(sample_trend_data)
    
    @pytest.mark.asyncio
    async def test_changepoint_detection(self, trend_analyzer, sample_trend_data):
        """Test changepoint detection analysis"""
        result = await trend_analyzer.analyze_trend(
            data=sample_trend_data,
            value_column='biomass',
            analysis_type='changepoint_detection'
        )
        
        assert 'analysis_type' in result
        assert result['analysis_type'] == 'changepoint_detection'
        assert 'changepoints' in result
        assert 'confidence_scores' in result
        assert 'trend_segments' in result
    
    @pytest.mark.asyncio
    async def test_correlation_analysis(self, trend_analyzer, sample_trend_data):
        """Test correlation analysis"""
        result = await trend_analyzer.analyze_trend(
            data=sample_trend_data,
            value_column='biomass',
            analysis_type='correlation_analysis',
            parameters={
                'correlation_variables': ['temperature', 'fishing_pressure']
            }
        )
        
        assert 'analysis_type' in result
        assert result['analysis_type'] == 'correlation_analysis'
        assert 'correlations' in result
        assert 'significance_tests' in result
        
        correlations = result['correlations']
        assert 'temperature' in correlations
        assert 'fishing_pressure' in correlations
    
    def test_trend_significance_testing(self, trend_analyzer):
        """Test statistical significance testing for trends"""
        # Create data with known significant trend
        x = np.arange(100)
        y_significant = 2 * x + np.random.normal(0, 5, 100)  # Strong linear trend
        y_no_trend = np.random.normal(50, 10, 100)  # No trend
        
        # Test significant trend
        result_sig = trend_analyzer._test_trend_significance(x, y_significant)
        assert result_sig['is_significant'] is True
        assert result_sig['p_value'] < 0.05
        
        # Test non-significant trend
        result_no_trend = trend_analyzer._test_trend_significance(x, y_no_trend)
        assert result_no_trend['p_value'] > 0.05


class TestIntegration:
    """Integration tests for predictive modeling components"""
    
    @pytest.fixture
    def services(self):
        """Create all service instances for integration testing"""
        return {
            'stock': StockAssessmentEngine(),
            'forecast': ForecastingService(),
            'models': ModelManager(),
            'trends': TrendAnalyzer()
        }
    
    @pytest.fixture
    def integrated_dataset(self):
        """Comprehensive dataset for integration testing"""
        dates = pd.date_range('2020-01-01', '2024-08-01', freq='M')
        n_points = len(dates)
        
        return pd.DataFrame({
            'date': dates,
            'species': 'Pomfret',
            'biomass': np.random.lognormal(6, 0.3, n_points),
            'catch_rate': np.random.normal(45, 8, n_points),
            'fishing_mortality': np.random.normal(0.25, 0.05, n_points),
            'sea_temperature': np.random.normal(26, 2, n_points),
            'salinity': np.random.normal(35, 1, n_points),
            'depth': np.random.uniform(20, 150, n_points)
        })
    
    @pytest.mark.asyncio
    async def test_complete_predictive_workflow(self, services, integrated_dataset):
        """Test complete predictive modeling workflow"""
        # Step 1: Stock Assessment
        assessment_params = {
            'species': 'Pomfret',
            'assessment_type': 'surplus_production',
            'time_period': {
                'start_date': '2020-01-01',
                'end_date': '2024-08-01'
            }
        }
        
        with patch.object(services['stock'], '_load_species_data') as mock_stock_data:
            mock_stock_data.return_value = integrated_dataset
            
            assessment_result = await services['stock'].perform_assessment(**assessment_params)
            assert 'stock_status' in assessment_result
        
        # Step 2: Trend Analysis
        trend_result = await services['trends'].analyze_trend(
            data=integrated_dataset,
            value_column='biomass',
            analysis_type='seasonal_decomposition',
            parameters={'period': 12}
        )
        assert 'components' in trend_result
        
        # Step 3: Model Training
        model_config = {
            'name': 'integrated_biomass_model',
            'type': 'regression',
            'algorithm': 'random_forest',
            'features': ['sea_temperature', 'salinity', 'depth', 'fishing_mortality'],
            'target': 'biomass'
        }
        
        training_result = await services['models'].train_model(
            config=model_config,
            training_data=integrated_dataset
        )
        model_id = training_result['model_id']
        
        # Step 4: Forecasting
        forecast_result = await services['forecast'].create_forecast(
            data=integrated_dataset,
            target_column='biomass',
            forecast_horizon=6,
            model_type='arima'
        )
        assert 'forecast_values' in forecast_result
        
        # Step 5: Model Prediction
        prediction_data = integrated_dataset.tail(5)[model_config['features']]
        
        with patch.object(services['models'], '_load_trained_model') as mock_model:
            mock_ml_model = Mock()
            mock_ml_model.predict.return_value = np.random.random(5) * 1000
            mock_model.return_value = mock_ml_model
            
            prediction_result = await services['models'].make_prediction(
                model_id=model_id,
                input_data=prediction_data
            )
            assert 'predictions' in prediction_result
    
    @pytest.mark.asyncio
    async def test_cross_component_data_flow(self, services, integrated_dataset):
        """Test data flow between different predictive components"""
        # Trend analysis results inform forecasting
        trend_result = await services['trends'].analyze_trend(
            data=integrated_dataset,
            value_column='biomass',
            analysis_type='linear_trend'
        )
        
        # Use trend information to inform forecasting model selection
        if trend_result['trend_strength'] > 0.7:
            model_type = 'prophet'  # Better for strong trends
        else:
            model_type = 'arima'
        
        forecast_result = await services['forecast'].create_forecast(
            data=integrated_dataset,
            target_column='biomass',
            model_type=model_type,
            forecast_horizon=3
        )
        
        assert forecast_result['model_type'] == model_type
    
    def test_performance_benchmarks(self, services, integrated_dataset):
        """Test performance benchmarks for predictive operations"""
        import time
        
        # Benchmark trend analysis
        start_time = time.time()
        asyncio.run(services['trends'].analyze_trend(
            data=integrated_dataset,
            value_column='biomass',
            analysis_type='linear_trend'
        ))
        trend_time = time.time() - start_time
        
        # Should complete trend analysis in less than 5 seconds
        assert trend_time < 5.0
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, services):
        """Test error handling and recovery across services"""
        # Test with empty dataset
        empty_data = pd.DataFrame()
        
        with pytest.raises((ValueError, IndexError)):
            await services['trends'].analyze_trend(
                data=empty_data,
                value_column='biomass',
                analysis_type='linear_trend'
            )
        
        # Test with invalid model configuration
        invalid_config = {
            'name': 'invalid_model',
            'type': 'invalid_type',
            'algorithm': 'nonexistent_algo'
        }
        
        with pytest.raises((ValueError, KeyError)):
            await services['models'].train_model(
                config=invalid_config,
                training_data=pd.DataFrame({'a': [1, 2, 3]})
            )


# Test fixtures and utilities
@pytest.fixture
def mock_ml_model():
    """Mock machine learning model for testing"""
    mock_model = Mock()
    mock_model.fit.return_value = None
    mock_model.predict.return_value = np.array([1, 2, 3, 4, 5])
    mock_model.score.return_value = 0.85
    mock_model.feature_importances_ = np.array([0.3, 0.25, 0.25, 0.2])
    return mock_model

@pytest.fixture
def mock_time_series_model():
    """Mock time series model for testing"""
    mock_model = Mock()
    mock_model.fit.return_value = None
    mock_model.forecast.return_value = (
        np.array([100, 102, 104, 106]),  # forecast values
        np.array([[95, 105], [96, 108], [97, 111], [98, 114]])  # confidence intervals
    )
    mock_model.aic = 150.5
    mock_model.bic = 165.2
    return mock_model

# Run tests with coverage
if __name__ == '__main__':
    pytest.main([
        __file__,
        '-v',
        '--cov=backend.app.predictive',
        '--cov-report=html',
        '--cov-report=term-missing'
    ])