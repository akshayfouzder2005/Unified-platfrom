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

# Resilient imports for development environments
__all__ = []

try:
    from .stock_assessment import StockAssessmentEngine
    __all__.append('StockAssessmentEngine')
except ImportError:
    pass

try:
    from .forecasting_engine import ForecastingEngine
    __all__.append('ForecastingEngine')
except ImportError:
    pass

try:
    from .trend_analysis import TrendAnalyzer
    __all__.append('TrendAnalyzer')
except ImportError:
    pass

try:
    from .population_models import PopulationModeler
    __all__.append('PopulationModeler')
except ImportError:
    pass

try:
    from .environmental_models import EnvironmentalModeler
    __all__.append('EnvironmentalModeler')
except ImportError:
    pass

__version__ = "2.0.0"