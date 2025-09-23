"""
📈 Predictive Analytics Package

Phase 2 - Spatial & Predictive Analytics
Ocean-Bio Marine Data Platform

This package provides comprehensive predictive modeling capabilities:
- Stock assessment and fisheries forecasting
- Time-series analysis and trend prediction
- Population dynamics modeling
- Environmental correlation analysis
- Statistical forecasting algorithms

Author: Ocean-Bio Development Team
Version: 2.0.0
"""

from .stock_assessment import StockAssessmentEngine
from .forecasting_engine import ForecastingEngine
from .trend_analysis import TrendAnalyzer
from .population_models import PopulationModeler
from .environmental_models import EnvironmentalModeler

__all__ = [
    'StockAssessmentEngine',
    'ForecastingEngine', 
    'TrendAnalyzer',
    'PopulationModeler',
    'EnvironmentalModeler'
]

__version__ = "2.0.0"