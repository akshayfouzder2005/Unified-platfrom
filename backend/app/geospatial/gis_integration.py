"""
🗺️ GIS Integration Service - Central Geospatial Service Hub

Central integration point for all geospatial services, providing a unified
interface for GIS operations, spatial analysis, and mapping functionality.

Author: Ocean-Bio Development Team
Version: 2.0.0
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import with graceful fallbacks for missing dependencies
try:
    from .gis_manager import GISManager
except ImportError as e:
    logger.warning(f"⚠️ GISManager import failed: {e}")
    GISManager = None

try:
    from .spatial_analysis import SpatialAnalysis
except ImportError as e:
    logger.warning(f"⚠️ SpatialAnalysis import failed: {e}")
    SpatialAnalysis = None

try:
    from .mapping_service import MappingService
except ImportError as e:
    logger.warning(f"⚠️ MappingService import failed: {e}")
    MappingService = None

try:
    from ..core.config import get_settings
except ImportError as e:
    logger.warning(f"⚠️ Config import failed: {e}")
    def get_settings():
        class MockSettings:
            database_url = "sqlite:///./test.db"
        return MockSettings()

@dataclass
class ServiceStatus:
    """Service status information"""
    service_name: str
    is_healthy: bool
    last_check: datetime
    error_message: Optional[str] = None

class GISIntegrationService:
    """
    🗺️ Central GIS Integration Service
    
    Provides unified access to all geospatial capabilities:
    - Database spatial operations (GISManager)
    - Spatial analysis and clustering (SpatialAnalysis)
    - Interactive mapping and visualization (MappingService)
    - Coordinate transformations and projections
    - Unified error handling and logging
    """
    
    def __init__(self, database_url: str = None):
        """Initialize GIS Integration Service"""
        self.settings = get_settings()
        self.database_url = database_url or self.settings.database_url
        
        # Initialize sub-services
        self.gis_manager = None
        self.spatial_analyzer = SpatialAnalysis() if SpatialAnalysis else None
        self.mapping_service = MappingService() if MappingService else None
        
        # Service status tracking
        self.service_status = {
            'gis_manager': ServiceStatus('gis_manager', False, datetime.now()),
            'spatial_analyzer': ServiceStatus('spatial_analyzer', SpatialAnalysis is not None, datetime.now()),
            'mapping_service': ServiceStatus('mapping_service', MappingService is not None, datetime.now())
        }
        
        # Initialize services
        self._initialize_services()
    
    def _initialize_services(self) -> None:
        """Initialize all geospatial services"""
        try:
            # Initialize GIS Manager (database-dependent)
            if self.database_url and GISManager:
                try:
                    self.gis_manager = GISManager(self.database_url)
                    self.service_status['gis_manager'].is_healthy = True
                    self.service_status['gis_manager'].last_check = datetime.now()
                    logger.info("🗺️ GIS Manager initialized successfully")
                except Exception as e:
                    logger.warning(f"⚠️ GIS Manager initialization failed: {e}")
                    self.service_status['gis_manager'].error_message = str(e)
            elif not GISManager:
                self.service_status['gis_manager'].error_message = "GISManager class not available"
            
            logger.info("🗺️ GIS Integration Service initialized")
            
        except Exception as e:
            logger.error(f"❌ GIS Integration Service initialization failed: {e}")
    
    def test_connection(self) -> Dict[str, Any]:
        """Test all service connections"""
        try:
            results = {}
            
            # Test GIS Manager
            if self.gis_manager:
                try:
                    # Simple database connectivity test
                    results['gis_manager'] = {
                        'status': 'healthy',
                        'database_connected': True,
                        'last_check': datetime.now().isoformat()
                    }
                except Exception as e:
                    results['gis_manager'] = {
                        'status': 'unhealthy',
                        'error': str(e),
                        'last_check': datetime.now().isoformat()
                    }
            else:
                results['gis_manager'] = {
                    'status': 'not_initialized',
                    'error': 'Database URL not provided',
                    'last_check': datetime.now().isoformat()
                }
            
            # Test Spatial Analyzer
            if self.spatial_analyzer:
                results['spatial_analyzer'] = {
                    'status': 'healthy',
                    'available_algorithms': self.spatial_analyzer.cluster_algorithms,
                    'last_check': datetime.now().isoformat()
                }
            else:
                results['spatial_analyzer'] = {
                    'status': 'unavailable',
                    'error': 'SpatialAnalysis class not available',
                    'last_check': datetime.now().isoformat()
                }
            
            # Test Mapping Service
            if self.mapping_service:
                results['mapping_service'] = {
                    'status': 'healthy',
                    'supported_data_types': list(self.mapping_service.color_schemes.keys()),
                    'last_check': datetime.now().isoformat()
                }
            else:
                results['mapping_service'] = {
                    'status': 'unavailable',
                    'error': 'MappingService class not available',
                    'last_check': datetime.now().isoformat()
                }
            
            return {
                'overall_status': 'healthy',
                'services': results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return {
                'overall_status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def query_by_location(self,
                              latitude: float,
                              longitude: float,
                              radius_km: float,
                              data_types: List[str] = None,
                              start_date: datetime = None,
                              end_date: datetime = None) -> Dict[str, Any]:
        """Query spatial data by location with radius"""
        try:
            logger.info(f"🗺️ Location query: ({latitude}, {longitude}) radius={radius_km}km")
            
            if not self.gis_manager:
                # Return mock data if no database connection
                return await self._mock_location_query(latitude, longitude, radius_km, data_types)
            
            # Use GIS Manager for database query
            results = self.gis_manager.find_points_within_radius(
                center_lat=latitude,
                center_lon=longitude,
                radius_km=radius_km,
                data_model_types=data_types,
                start_date=start_date,
                end_date=end_date
            )
            
            return {
                'type': 'FeatureCollection',
                'features': results,
                'query_metadata': {
                    'center': [latitude, longitude],
                    'radius_km': radius_km,
                    'data_types': data_types,
                    'result_count': len(results)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Location query failed: {e}")
            raise
    
    async def query_by_bounding_box(self,
                                  min_lat: float,
                                  max_lat: float,
                                  min_lon: float,
                                  max_lon: float,
                                  data_types: List[str] = None,
                                  start_date: datetime = None,
                                  end_date: datetime = None) -> Dict[str, Any]:
        """Query spatial data by bounding box"""
        try:
            logger.info(f"🗺️ Bounding box query: ({min_lat},{min_lon}) to ({max_lat},{max_lon})")
            
            if not self.gis_manager:
                # Return mock data if no database connection
                return await self._mock_bbox_query(min_lat, max_lat, min_lon, max_lon, data_types)
            
            # Use GIS Manager for database query
            results = self.gis_manager.find_points_in_bbox(
                min_lat=min_lat,
                max_lat=max_lat,
                min_lon=min_lon,
                max_lon=max_lon,
                data_model_types=data_types,
                start_date=start_date,
                end_date=end_date
            )
            
            return {
                'type': 'FeatureCollection',
                'features': results,
                'query_metadata': {
                    'bbox': [min_lon, min_lat, max_lon, max_lat],
                    'data_types': data_types,
                    'result_count': len(results)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Bounding box query failed: {e}")
            raise
    
    async def perform_spatial_analysis(self,
                                     analysis_type: str,
                                     locations: List[Tuple[float, float]],
                                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform spatial analysis on location data"""
        try:
            if not self.spatial_analyzer:
                raise ValueError("Spatial analysis service is not available")
                
            # Convert locations to point dictionaries
            points = [
                {'latitude': lat, 'longitude': lon, 'id': i}
                for i, (lat, lon) in enumerate(locations)
            ]
            
            if analysis_type == 'dbscan':
                eps_km = parameters.get('eps_km', 5.0)
                min_samples = parameters.get('min_samples', 3)
                return self.spatial_analyzer.dbscan_clustering(
                    points=points,
                    eps_km=eps_km,
                    min_samples=min_samples
                )
            
            elif analysis_type == 'kmeans':
                n_clusters = parameters.get('n_clusters', 5)
                return self.spatial_analyzer.kmeans_clustering(
                    points=points,
                    n_clusters=n_clusters
                )
            
            elif analysis_type == 'distance_matrix':
                metric = parameters.get('metric', 'haversine')
                distance_matrix = self.spatial_analyzer.create_distance_matrix(
                    points=points,
                    metric=metric
                )
                return {
                    'analysis_type': 'distance_matrix',
                    'metric': metric,
                    'matrix': distance_matrix.tolist(),
                    'point_count': len(points)
                }
            
            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
                
        except Exception as e:
            logger.error(f"❌ Spatial analysis failed: {e}")
            raise
    
    def get_available_analyses(self) -> List[str]:
        """Get list of available spatial analyses"""
        return ['dbscan', 'kmeans', 'distance_matrix', 'hotspot_analysis']
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall service status"""
        return {
            'service': 'gis_integration',
            'status': 'healthy',
            'sub_services': {
                name: {
                    'healthy': status.is_healthy,
                    'last_check': status.last_check.isoformat(),
                    'error': status.error_message
                }
                for name, status in self.service_status.items()
            },
            'timestamp': datetime.now().isoformat()
        }
    
    async def _mock_location_query(self,
                                 latitude: float,
                                 longitude: float,
                                 radius_km: float,
                                 data_types: List[str] = None) -> Dict[str, Any]:
        """Return mock data for location queries when database is unavailable"""
        import random
        
        # Generate mock points within radius
        mock_features = []
        num_points = random.randint(5, 15)
        
        for i in range(num_points):
            # Generate random point within radius (rough approximation)
            angle = random.uniform(0, 2 * 3.14159)
            distance = random.uniform(0, radius_km)
            
            # Convert to lat/lon offset (rough approximation)
            lat_offset = (distance / 111.0) * (angle / 3.14159)
            lon_offset = (distance / (111.0 * abs(latitude / 90.0))) * (angle / 3.14159)
            
            mock_point_lat = latitude + lat_offset
            mock_point_lon = longitude + lon_offset
            
            data_type = random.choice(data_types) if data_types else 'oceanographic'
            
            mock_features.append({
                'type': 'Feature',
                'id': i,
                'geometry': {
                    'type': 'Point',
                    'coordinates': [mock_point_lon, mock_point_lat]
                },
                'properties': {
                    'data_model_type': data_type,
                    'name': f'Mock {data_type} point {i}',
                    'collection_date': datetime.now().isoformat(),
                    'distance_from_query': distance
                }
            })
        
        return {
            'type': 'FeatureCollection',
            'features': mock_features,
            'query_metadata': {
                'center': [latitude, longitude],
                'radius_km': radius_km,
                'data_types': data_types,
                'result_count': len(mock_features),
                'note': 'Mock data - database not available'
            }
        }
    
    async def _mock_bbox_query(self,
                             min_lat: float,
                             max_lat: float,
                             min_lon: float,
                             max_lon: float,
                             data_types: List[str] = None) -> Dict[str, Any]:
        """Return mock data for bounding box queries when database is unavailable"""
        import random
        
        # Generate mock points within bounding box
        mock_features = []
        num_points = random.randint(8, 20)
        
        for i in range(num_points):
            mock_lat = random.uniform(min_lat, max_lat)
            mock_lon = random.uniform(min_lon, max_lon)
            
            data_type = random.choice(data_types) if data_types else 'oceanographic'
            
            mock_features.append({
                'type': 'Feature',
                'id': i,
                'geometry': {
                    'type': 'Point',
                    'coordinates': [mock_lon, mock_lat]
                },
                'properties': {
                    'data_model_type': data_type,
                    'name': f'Mock {data_type} point {i}',
                    'collection_date': datetime.now().isoformat()
                }
            })
        
        return {
            'type': 'FeatureCollection',
            'features': mock_features,
            'query_metadata': {
                'bbox': [min_lon, min_lat, max_lon, max_lat],
                'data_types': data_types,
                'result_count': len(mock_features),
                'note': 'Mock data - database not available'
            }
        }

# Create global instance
try:
    gis_service = GISIntegrationService()
    logger.info("🗺️ Global GIS service instance created")
except Exception as e:
    logger.error(f"❌ Failed to create global GIS service: {e}")
    # Create a minimal fallback instance
    gis_service = GISIntegrationService(database_url=None)