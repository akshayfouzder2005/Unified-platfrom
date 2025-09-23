"""
🗺️ Geospatial Analysis Package

Phase 2 - Spatial & Predictive Analytics
Ocean-Bio Marine Data Platform

This package provides comprehensive geospatial analysis capabilities:
- PostGIS integration and spatial queries
- Interactive mapping and visualization
- Spatial analysis algorithms and tools
- Coordinate system transformations
- Geographic data clustering and hotspot analysis

Author: Ocean-Bio Development Team
Version: 2.0.0
"""

from .gis_manager import GISManager
from .mapping_service import MappingService
from .spatial_analysis import SpatialAnalysis
from .coordinate_system import CoordinateSystem

__all__ = [
    'GISManager',
    'MappingService', 
    'SpatialAnalysis',
    'CoordinateSystem'
]

__version__ = "2.0.0"