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

# Resilient imports for development environments
__all__ = []

try:
    from .gis_manager import GISManager
    __all__.append('GISManager')
except ImportError:
    pass

try:
    from .mapping_service import MappingService
    __all__.append('MappingService')
except ImportError:
    pass

try:
    from .spatial_analysis import SpatialAnalysis
    __all__.append('SpatialAnalysis')
except ImportError:
    pass

try:
    from .coordinate_system import CoordinateSystem
    __all__.append('CoordinateSystem')
except ImportError:
    pass

# Always try to import gis_integration for the service
try:
    from .gis_integration import gis_service
    __all__.append('gis_service')
except ImportError:
    pass

__version__ = "2.0.0"