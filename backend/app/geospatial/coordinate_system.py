"""
🗺️ Coordinate System Management - CRS Transformations & Projections

Advanced coordinate reference system management for marine geospatial data.
Handles transformations, projections, and datum conversions.

Author: Ocean-Bio Development Team
Version: 2.0.0
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import pyproj
from pyproj import CRS, Transformer
import math
import numpy as np

logger = logging.getLogger(__name__)

class CoordinateSystem:
    """
    🗺️ Advanced Coordinate Reference System Manager
    
    Provides comprehensive CRS management capabilities:
    - Coordinate transformations between different systems
    - Projection and datum conversions
    - Marine-specific coordinate systems
    - Accuracy assessment and validation
    - Batch processing for large datasets
    """
    
    def __init__(self):
        """Initialize the coordinate system manager"""
        self.supported_crs = {
            # Global Systems
            'WGS84': 'EPSG:4326',           # World Geodetic System 1984
            'WGS84_PSEUDO_MERCATOR': 'EPSG:3857',  # Web Mercator
            
            # Indian Systems
            'INDIAN_1975': 'EPSG:4240',     # Indian 1975 datum
            'EVEREST_INDIA_NEPAL': 'EPSG:4242',  # Everest India and Nepal
            'KALIANPUR_1937': 'EPSG:4144',  # Kalianpur 1937
            'KALIANPUR_1962': 'EPSG:4145',  # Kalianpur 1962
            'KALIANPUR_1975': 'EPSG:4146',  # Kalianpur 1975
            
            # UTM Zones for India
            'UTM_42N_WGS84': 'EPSG:32642',  # UTM Zone 42N (Western India)
            'UTM_43N_WGS84': 'EPSG:32643',  # UTM Zone 43N (Central India)
            'UTM_44N_WGS84': 'EPSG:32644',  # UTM Zone 44N (Eastern India)
            'UTM_45N_WGS84': 'EPSG:32645',  # UTM Zone 45N (Far Eastern India)
            'UTM_46N_WGS84': 'EPSG:32646',  # UTM Zone 46N (Northeastern India)
            
            # Marine-specific systems
            'MERCATOR_WORLD': 'EPSG:3395',  # World Mercator
            'ROBINSON': 'EPSG:54030',       # Robinson projection (good for global marine maps)
            'MOLLWEIDE': 'EPSG:54009',      # Mollweide projection
            
            # Regional systems
            'INDIA_TM': 'EPSG:7755',        # Indian Transverse Mercator
            'INDIA_LAMBERT': 'EPSG:7756',   # Indian Lambert Conformal Conic
        }
        
        self.utm_zones_india = {
            'western': {'zone': 42, 'epsg': 'EPSG:32642', 'bounds': [68, 74]},
            'central': {'zone': 43, 'epsg': 'EPSG:32643', 'bounds': [74, 80]},
            'eastern': {'zone': 44, 'epsg': 'EPSG:32644', 'bounds': [80, 86]},
            'far_eastern': {'zone': 45, 'epsg': 'EPSG:32645', 'bounds': [86, 92]},
            'northeastern': {'zone': 46, 'epsg': 'EPSG:32646', 'bounds': [92, 98]}
        }
        
        # Initialize transformers cache
        self._transformers_cache = {}
    
    def get_transformer(self, from_crs: str, to_crs: str) -> Transformer:
        """Get or create a coordinate transformer"""
        try:
            cache_key = f"{from_crs}_{to_crs}"
            
            if cache_key not in self._transformers_cache:
                # Resolve CRS names to EPSG codes
                from_epsg = self.supported_crs.get(from_crs, from_crs)
                to_epsg = self.supported_crs.get(to_crs, to_crs)
                
                transformer = Transformer.from_crs(
                    from_epsg, to_epsg, 
                    always_xy=True  # Ensure consistent lon,lat order
                )
                self._transformers_cache[cache_key] = transformer
                
                logger.info(f"🗺️ Created transformer: {from_epsg} → {to_epsg}")
            
            return self._transformers_cache[cache_key]
            
        except Exception as e:
            logger.error(f"❌ Transformer creation failed: {e}")
            raise
    
    def transform_point(self, 
                       longitude: float, 
                       latitude: float, 
                       from_crs: str, 
                       to_crs: str) -> Tuple[float, float]:
        """Transform a single point between coordinate systems"""
        try:
            transformer = self.get_transformer(from_crs, to_crs)
            x, y = transformer.transform(longitude, latitude)
            
            logger.debug(f"🗺️ Transformed point: ({longitude}, {latitude}) → ({x}, {y})")
            return float(x), float(y)
            
        except Exception as e:
            logger.error(f"❌ Point transformation failed: {e}")
            return longitude, latitude  # Return original coordinates on failure
    
    def transform_points_batch(self, 
                              points: List[Tuple[float, float]], 
                              from_crs: str, 
                              to_crs: str) -> List[Tuple[float, float]]:
        """Transform multiple points efficiently"""
        try:
            if not points:
                return []
            
            transformer = self.get_transformer(from_crs, to_crs)
            
            # Separate coordinates for batch processing
            lons, lats = zip(*points)
            
            # Transform all points at once
            x_coords, y_coords = transformer.transform(lons, lats)
            
            # Combine results
            if isinstance(x_coords, np.ndarray):
                transformed_points = list(zip(x_coords.tolist(), y_coords.tolist()))
            else:
                transformed_points = [(float(x_coords), float(y_coords))]
            
            logger.info(f"🗺️ Batch transformed {len(points)} points")
            return transformed_points
            
        except Exception as e:
            logger.error(f"❌ Batch transformation failed: {e}")
            return points  # Return original points on failure
    
    def get_optimal_utm_zone(self, longitude: float, latitude: float) -> Dict[str, Any]:
        """Determine the optimal UTM zone for a location in India"""
        try:
            # Check which UTM zone this longitude falls into
            for region, zone_info in self.utm_zones_india.items():
                lon_min, lon_max = zone_info['bounds']
                if lon_min <= longitude < lon_max:
                    return {
                        'region': region,
                        'zone_number': zone_info['zone'],
                        'epsg_code': zone_info['epsg'],
                        'hemisphere': 'N' if latitude >= 0 else 'S',
                        'recommended': True
                    }
            
            # If outside Indian zones, calculate standard UTM zone
            zone_number = int((longitude + 180) / 6) + 1
            hemisphere = 'N' if latitude >= 0 else 'S'
            epsg_code = f"EPSG:326{zone_number:02d}" if hemisphere == 'N' else f"EPSG:327{zone_number:02d}"
            
            return {
                'region': 'global',
                'zone_number': zone_number,
                'epsg_code': epsg_code,
                'hemisphere': hemisphere,
                'recommended': False
            }
            
        except Exception as e:
            logger.error(f"❌ UTM zone determination failed: {e}")
            return {'error': str(e)}
    
    def calculate_scale_factor(self, 
                              longitude: float, 
                              latitude: float, 
                              crs: str) -> float:
        """Calculate scale factor at a given location for a projection"""
        try:
            crs_obj = CRS.from_string(self.supported_crs.get(crs, crs))
            
            if crs_obj.is_geographic:
                # Geographic coordinates - scale factor is 1
                return 1.0
            
            # For projected systems, estimate scale factor
            if 'UTM' in crs.upper() or 'utm' in str(crs_obj).lower():
                # UTM scale factor calculation
                # Get central meridian of the zone
                utm_info = self.get_optimal_utm_zone(longitude, latitude)
                central_meridian = (utm_info['zone_number'] - 1) * 6 - 177
                
                # Calculate distance from central meridian
                delta_lon = abs(longitude - central_meridian)
                
                # UTM scale factor approximation
                k0 = 0.9996  # UTM scale factor at central meridian
                scale_factor = k0 * (1 + (delta_lon * math.pi / 180) ** 2 * math.cos(latitude * math.pi / 180) ** 2 / 2)
                
                return scale_factor
            
            # For other projections, return approximate value
            return 1.0
            
        except Exception as e:
            logger.error(f"❌ Scale factor calculation failed: {e}")
            return 1.0
    
    def validate_coordinates(self, 
                           longitude: float, 
                           latitude: float, 
                           crs: str = 'WGS84') -> Dict[str, Any]:
        """Validate coordinate values for a given CRS"""
        try:
            crs_obj = CRS.from_string(self.supported_crs.get(crs, crs))
            
            validation = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'crs_info': {
                    'name': crs_obj.name,
                    'type': 'geographic' if crs_obj.is_geographic else 'projected',
                    'units': crs_obj.axis_info[0].unit_name if crs_obj.axis_info else 'unknown'
                }
            }
            
            if crs_obj.is_geographic:
                # Validate geographic coordinates
                if not (-180 <= longitude <= 180):
                    validation['valid'] = False
                    validation['errors'].append(f"Longitude {longitude} outside valid range [-180, 180]")
                
                if not (-90 <= latitude <= 90):
                    validation['valid'] = False
                    validation['errors'].append(f"Latitude {latitude} outside valid range [-90, 90]")
                
                # Check if coordinates are in Indian region
                if 68 <= longitude <= 98 and 8 <= latitude <= 37:
                    validation['warnings'].append("Coordinates are within Indian region")
                else:
                    validation['warnings'].append("Coordinates are outside typical Indian region")
            
            else:
                # For projected coordinates, basic range checks
                if abs(longitude) > 10000000 or abs(latitude) > 10000000:
                    validation['warnings'].append("Coordinates seem unusually large for projected system")
            
            return validation
            
        except Exception as e:
            logger.error(f"❌ Coordinate validation failed: {e}")
            return {'valid': False, 'errors': [str(e)]}
    
    def get_crs_info(self, crs: str) -> Dict[str, Any]:
        """Get detailed information about a coordinate reference system"""
        try:
            crs_code = self.supported_crs.get(crs, crs)
            crs_obj = CRS.from_string(crs_code)
            
            info = {
                'name': crs_obj.name,
                'epsg_code': crs_code,
                'type': 'geographic' if crs_obj.is_geographic else 'projected',
                'datum': crs_obj.datum.name if crs_obj.datum else 'Unknown',
                'ellipsoid': crs_obj.ellipsoid.name if crs_obj.ellipsoid else 'Unknown',
                'units': crs_obj.axis_info[0].unit_name if crs_obj.axis_info else 'degrees',
                'area_of_use': None,
                'accuracy': None
            }
            
            # Get area of use if available
            if crs_obj.area_of_use:
                info['area_of_use'] = {
                    'bounds': [
                        crs_obj.area_of_use.west,
                        crs_obj.area_of_use.south,
                        crs_obj.area_of_use.east,
                        crs_obj.area_of_use.north
                    ],
                    'description': crs_obj.area_of_use.name
                }
            
            # Add specific information for UTM zones
            if 'UTM' in crs.upper():
                utm_info = self._extract_utm_info(crs_code)
                if utm_info:
                    info.update(utm_info)
            
            return info
            
        except Exception as e:
            logger.error(f"❌ CRS info retrieval failed: {e}")
            return {'error': str(e)}
    
    def suggest_best_crs(self, 
                        points: List[Tuple[float, float]], 
                        purpose: str = 'analysis') -> Dict[str, Any]:
        """Suggest the best CRS for a set of points and intended use"""
        try:
            if not points:
                return {'error': 'No points provided'}
            
            # Calculate bounds
            lons, lats = zip(*points)
            min_lon, max_lon = min(lons), max(lons)
            min_lat, max_lat = min(lats), max(lats)
            center_lon = (min_lon + max_lon) / 2
            center_lat = (min_lat + max_lat) / 2
            
            suggestions = []
            
            # Check if data is within India
            if 68 <= center_lon <= 98 and 8 <= center_lat <= 37:
                # Indian region - suggest appropriate UTM zone
                utm_info = self.get_optimal_utm_zone(center_lon, center_lat)
                
                if purpose == 'analysis' or purpose == 'measurement':
                    suggestions.append({
                        'crs': utm_info['epsg_code'],
                        'name': f"UTM Zone {utm_info['zone_number']}N",
                        'reason': 'Optimal for distance and area calculations in this region',
                        'priority': 1,
                        'suitable_for': ['analysis', 'measurement', 'mapping']
                    })
                
                # Suggest Indian national systems
                suggestions.append({
                    'crs': 'EPSG:7755',
                    'name': 'Indian Transverse Mercator',
                    'reason': 'National coordinate system for India',
                    'priority': 2,
                    'suitable_for': ['national_mapping', 'administration']
                })
            
            # Always suggest WGS84 for display and web mapping
            if purpose == 'display' or purpose == 'web':
                suggestions.append({
                    'crs': 'EPSG:4326',
                    'name': 'WGS 84',
                    'reason': 'Universal compatibility and web mapping standard',
                    'priority': 1,
                    'suitable_for': ['display', 'web', 'gps']
                })
                
                suggestions.append({
                    'crs': 'EPSG:3857',
                    'name': 'Web Mercator',
                    'reason': 'Standard for web mapping applications',
                    'priority': 2,
                    'suitable_for': ['web', 'display', 'tiles']
                })
            
            # For large area analysis, suggest equal-area projections
            if purpose == 'area_analysis':
                suggestions.append({
                    'crs': 'EPSG:54009',
                    'name': 'Mollweide',
                    'reason': 'Equal-area projection good for area calculations',
                    'priority': 1,
                    'suitable_for': ['area_analysis', 'global_mapping']
                })
            
            # Sort suggestions by priority
            suggestions.sort(key=lambda x: x['priority'])
            
            return {
                'center_point': [center_lon, center_lat],
                'bounds': [min_lon, min_lat, max_lon, max_lat],
                'purpose': purpose,
                'suggestions': suggestions,
                'recommended': suggestions[0] if suggestions else None
            }
            
        except Exception as e:
            logger.error(f"❌ CRS suggestion failed: {e}")
            return {'error': str(e)}
    
    def _extract_utm_info(self, crs_code: str) -> Optional[Dict[str, Any]]:
        """Extract UTM zone information from CRS code"""
        try:
            if 'EPSG:326' in crs_code or 'EPSG:327' in crs_code:
                zone_code = crs_code.split(':')[1]
                hemisphere = 'N' if zone_code.startswith('326') else 'S'
                zone_number = int(zone_code[-2:])
                
                return {
                    'utm_zone': zone_number,
                    'utm_hemisphere': hemisphere,
                    'central_meridian': (zone_number - 1) * 6 - 177,
                    'false_easting': 500000,
                    'false_northing': 0 if hemisphere == 'N' else 10000000
                }
            
            return None
            
        except Exception:
            return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get coordinate system manager status"""
        try:
            return {
                'supported_systems': len(self.supported_crs),
                'cached_transformers': len(self._transformers_cache),
                'utm_zones_india': list(self.utm_zones_india.keys()),
                'available_crs': list(self.supported_crs.keys()),
                'pyproj_version': pyproj.__version__,
                'status': 'operational'
            }
            
        except Exception as e:
            logger.error(f"❌ Status check failed: {e}")
            return {'status': 'error', 'error': str(e)}

# Global coordinate system instance
coordinate_system = CoordinateSystem()