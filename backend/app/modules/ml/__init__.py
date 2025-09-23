"""
Machine Learning API Module
Provides AI model endpoints for species identification and analysis
"""

from .router import router as ml_router

__all__ = ['ml_router']
