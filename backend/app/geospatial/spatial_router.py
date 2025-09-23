"""
🗺️ Geospatial API Router - Spatial Analysis Endpoints

RESTful API endpoints for geospatial analysis and mapping functionality.
Provides access to GIS operations, spatial analysis, and coordinate transformations.

Author: Ocean-Bio Development Team
Version: 2.0.0
"""

from fastapi import APIRouter, HTTPException, Query, Body, Depends
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from .gis_manager import get_gis_manager, GISManager
from .mapping_service import mapping_service
from .spatial_analysis import spatial_analyzer
from .coordinate_system import coordinate_system

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/geospatial",
    tags=["geospatial"],
    responses={404: {"description": "Not found"}}
)

# Pydantic models for request/response validation
class SpatialPoint(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")
    data_model_type: str = Field(..., description="Data model type (edna, oceanographic, etc.)")
    record_id: int = Field(..., description="Reference to data record")
    point_name: Optional[str] = Field(None, description="Optional point name")
    elevation_m: Optional[float] = Field(None, description="Elevation in meters")
    depth_m: Optional[float] = Field(None, description="Depth in meters")
    collection_date: Optional[datetime] = Field(None, description="Collection date")
    properties: Optional[Dict[str, Any]] = Field(None, description="Additional properties")

class SpatialRegion(BaseModel):
    region_name: str = Field(..., description="Name of the region")
    region_type: str = Field(..., description="Type of region (marine_protected_area, fishing_zone, etc.)")
    geometry_wkt: str = Field(..., description="Geometry in WKT format")
    data_model_types: List[str] = Field(..., description="Applicable data model types")
    properties: Optional[Dict[str, Any]] = Field(None, description="Additional properties")

class ClusteringRequest(BaseModel):
    points: List[Dict[str, Any]] = Field(..., description="List of spatial points")
    algorithm: str = Field("dbscan", description="Clustering algorithm (dbscan, kmeans)")
    parameters: Dict[str, Any] = Field({}, description="Algorithm-specific parameters")

class HotspotRequest(BaseModel):
    points: List[Dict[str, Any]] = Field(..., description="List of spatial points")
    radius_km: float = Field(10.0, description="Search radius in kilometers")
    min_points: int = Field(5, description="Minimum points for hotspot")

class CoordinateTransformRequest(BaseModel):
    points: List[List[float]] = Field(..., description="List of [longitude, latitude] coordinates")
    from_crs: str = Field(..., description="Source coordinate system")
    to_crs: str = Field(..., description="Target coordinate system")

# ===== GIS MANAGER ENDPOINTS =====

@router.post("/points/add")
async def add_spatial_point(
    point: SpatialPoint,
    gis: GISManager = Depends(get_gis_manager)
):
    """Add a new spatial point to the database"""
    try:
        spatial_id = gis.add_spatial_point(
            data_model_type=point.data_model_type,
            record_id=point.record_id,
            latitude=point.latitude,
            longitude=point.longitude,
            point_name=point.point_name,
            elevation_m=point.elevation_m,
            depth_m=point.depth_m,
            collection_date=point.collection_date,
            properties=point.properties
        )
        
        if spatial_id:
            return {"success": True, "spatial_id": spatial_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to add spatial point")
            
    except Exception as e:
        logger.error(f"❌ Add spatial point failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/points/radius")
async def find_points_within_radius(
    center_lat: float = Query(..., description="Center latitude"),
    center_lon: float = Query(..., description="Center longitude"),
    radius_km: float = Query(..., description="Search radius in kilometers"),
    data_model_type: Optional[str] = Query(None, description="Filter by data model type"),
    gis: GISManager = Depends(get_gis_manager)
):
    """Find spatial points within specified radius"""
    try:
        points = gis.find_points_within_radius(
            center_lat=center_lat,
            center_lon=center_lon,
            radius_km=radius_km,
            data_model_type=data_model_type
        )
        
        return {
            "points": points,
            "count": len(points),
            "search_parameters": {
                "center": [center_lat, center_lon],
                "radius_km": radius_km,
                "data_model_type": data_model_type
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Radius search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/points/polygon")
async def find_points_in_polygon(
    polygon_wkt: str = Body(..., description="Polygon geometry in WKT format"),
    data_model_type: Optional[str] = Body(None, description="Filter by data model type"),
    gis: GISManager = Depends(get_gis_manager)
):
    """Find spatial points within a polygon"""
    try:
        points = gis.find_points_in_polygon(
            polygon_wkt=polygon_wkt,
            data_model_type=data_model_type
        )
        
        return {
            "points": points,
            "count": len(points),
            "search_parameters": {
                "polygon_wkt": polygon_wkt,
                "data_model_type": data_model_type
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Polygon search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/density/grid")
async def calculate_point_density(
    bounds: Dict[str, float] = Body(..., description="Bounds dictionary with min_lat, max_lat, min_lon, max_lon"),
    grid_size_km: float = Body(10.0, description="Grid cell size in kilometers"),
    data_model_type: Optional[str] = Body(None, description="Filter by data model type"),
    gis: GISManager = Depends(get_gis_manager)
):
    """Calculate point density grid within specified bounds"""
    try:
        density_result = gis.calculate_point_density(
            bounds=bounds,
            grid_size_km=grid_size_km,
            data_model_type=data_model_type
        )
        
        return density_result
        
    except Exception as e:
        logger.error(f"❌ Density calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/regions/create")
async def create_geographic_region(
    region: SpatialRegion,
    gis: GISManager = Depends(get_gis_manager)
):
    """Create a new geographic region"""
    try:
        region_id = gis.create_geographic_region(
            region_name=region.region_name,
            region_type=region.region_type,
            geometry_wkt=region.geometry_wkt,
            data_model_types=region.data_model_types,
            properties=region.properties
        )
        
        if region_id:
            return {"success": True, "region_id": region_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to create geographic region")
            
    except Exception as e:
        logger.error(f"❌ Region creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics/{data_model_type}")
async def get_spatial_statistics(
    data_model_type: str,
    gis: GISManager = Depends(get_gis_manager)
):
    """Get spatial statistics for a specific data model type"""
    try:
        stats = gis.get_data_model_statistics(data_model_type)
        return stats
        
    except Exception as e:
        logger.error(f"❌ Statistics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== SPATIAL ANALYSIS ENDPOINTS =====

@router.post("/analysis/clustering")
async def perform_clustering(request: ClusteringRequest):
    """Perform spatial clustering analysis"""
    try:
        if request.algorithm == "dbscan":
            eps_km = request.parameters.get("eps_km", 5.0)
            min_samples = request.parameters.get("min_samples", 3)
            result = spatial_analyzer.dbscan_clustering(
                request.points, eps_km=eps_km, min_samples=min_samples
            )
        elif request.algorithm == "kmeans":
            n_clusters = request.parameters.get("n_clusters", 5)
            result = spatial_analyzer.kmeans_clustering(
                request.points, n_clusters=n_clusters
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported algorithm: {request.algorithm}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Clustering analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analysis/hotspots")
async def identify_hotspots(request: HotspotRequest):
    """Identify spatial hotspots"""
    try:
        result = spatial_analyzer.identify_hotspots(
            points=request.points,
            radius_km=request.radius_km,
            min_points=request.min_points
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Hotspot analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analysis/spatial-statistics")
async def calculate_spatial_statistics(
    points: List[Dict[str, Any]] = Body(..., description="List of spatial points")
):
    """Calculate comprehensive spatial statistics"""
    try:
        stats = spatial_analyzer.calculate_spatial_statistics(points)
        return stats
        
    except Exception as e:
        logger.error(f"❌ Spatial statistics calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analysis/nearest-neighbors")
async def find_nearest_neighbors(
    target_point: Dict[str, Any] = Body(..., description="Target point"),
    candidate_points: List[Dict[str, Any]] = Body(..., description="Candidate points"),
    k: int = Body(5, description="Number of neighbors to find")
):
    """Find k nearest neighbors to a target point"""
    try:
        neighbors = spatial_analyzer.find_nearest_neighbors(
            target_point=target_point,
            candidate_points=candidate_points,
            k=k
        )
        
        return {
            "target_point": target_point,
            "neighbors": neighbors,
            "k": k
        }
        
    except Exception as e:
        logger.error(f"❌ Nearest neighbor search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analysis/patterns")
async def analyze_spatial_patterns(
    points: List[Dict[str, Any]] = Body(..., description="List of spatial points"),
    data_model_type: Optional[str] = Body(None, description="Filter by data model type")
):
    """Perform comprehensive spatial pattern analysis"""
    try:
        analysis = spatial_analyzer.analyze_spatial_patterns(
            points=points,
            data_model_type=data_model_type
        )
        
        return analysis
        
    except Exception as e:
        logger.error(f"❌ Spatial pattern analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== COORDINATE SYSTEM ENDPOINTS =====

@router.post("/coordinates/transform")
async def transform_coordinates(request: CoordinateTransformRequest):
    """Transform coordinates between different coordinate systems"""
    try:
        transformed_points = coordinate_system.transform_points_batch(
            points=[(lon, lat) for lon, lat in request.points],
            from_crs=request.from_crs,
            to_crs=request.to_crs
        )
        
        return {
            "original_crs": request.from_crs,
            "target_crs": request.to_crs,
            "original_points": request.points,
            "transformed_points": [[lon, lat] for lon, lat in transformed_points],
            "count": len(transformed_points)
        }
        
    except Exception as e:
        logger.error(f"❌ Coordinate transformation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/coordinates/utm-zone")
async def get_optimal_utm_zone(
    longitude: float = Query(..., description="Longitude"),
    latitude: float = Query(..., description="Latitude")
):
    """Get optimal UTM zone for a location"""
    try:
        utm_info = coordinate_system.get_optimal_utm_zone(longitude, latitude)
        return utm_info
        
    except Exception as e:
        logger.error(f"❌ UTM zone determination failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/coordinates/validate")
async def validate_coordinates(
    longitude: float = Query(..., description="Longitude"),
    latitude: float = Query(..., description="Latitude"),
    crs: str = Query("WGS84", description="Coordinate reference system")
):
    """Validate coordinate values for a given CRS"""
    try:
        validation = coordinate_system.validate_coordinates(longitude, latitude, crs)
        return validation
        
    except Exception as e:
        logger.error(f"❌ Coordinate validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/coordinates/crs-info/{crs}")
async def get_crs_info(crs: str):
    """Get detailed information about a coordinate reference system"""
    try:
        info = coordinate_system.get_crs_info(crs)
        return info
        
    except Exception as e:
        logger.error(f"❌ CRS info retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/coordinates/suggest-crs")
async def suggest_best_crs(
    points: List[List[float]] = Body(..., description="List of [longitude, latitude] coordinates"),
    purpose: str = Body("analysis", description="Intended use (analysis, display, web, area_analysis)")
):
    """Suggest the best CRS for a set of points and intended use"""
    try:
        suggestions = coordinate_system.suggest_best_crs(
            points=[(lon, lat) for lon, lat in points],
            purpose=purpose
        )
        
        return suggestions
        
    except Exception as e:
        logger.error(f"❌ CRS suggestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== MAPPING SERVICE ENDPOINTS =====

@router.post("/mapping/create-map")
async def create_interactive_map(
    spatial_data: Dict[str, Any] = Body(..., description="Spatial data for mapping"),
    center: Optional[List[float]] = Body(None, description="Map center [latitude, longitude]"),
    title: str = Body("Marine Data Visualization", description="Map title")
):
    """Create an interactive multi-layer map"""
    try:
        map_obj = mapping_service.create_multi_layer_map(
            spatial_data=spatial_data,
            center=center,
            title=title
        )
        
        if map_obj:
            # Get map statistics
            stats = mapping_service.get_map_statistics(spatial_data)
            
            return {
                "success": True,
                "message": "Interactive map created successfully",
                "statistics": stats
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create map")
            
    except Exception as e:
        logger.error(f"❌ Map creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/mapping/statistics")
async def get_mapping_statistics(
    data_type: Optional[str] = Query(None, description="Data model type filter")
):
    """Get mapping and visualization statistics"""
    try:
        # This would typically fetch actual spatial data from database
        # For now, return example statistics
        stats = {
            "total_layers": 5,
            "total_points": 1250,
            "data_type_counts": {
                "edna": 320,
                "oceanographic": 280,
                "otolith": 210,
                "taxonomy": 240,
                "fisheries": 200
            },
            "supported_projections": len(coordinate_system.supported_crs),
            "available_analysis_tools": [
                "clustering", "hotspot_identification", "density_analysis",
                "nearest_neighbor", "spatial_statistics"
            ]
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Statistics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== SYSTEM STATUS ENDPOINTS =====

@router.get("/status")
async def get_geospatial_system_status(gis: GISManager = Depends(get_gis_manager)):
    """Get comprehensive geospatial system status"""
    try:
        gis_status = gis.get_system_status()
        coordinate_status = coordinate_system.get_system_status()
        
        return {
            "gis_manager": gis_status,
            "coordinate_system": coordinate_status,
            "spatial_analyzer": {
                "algorithms": spatial_analyzer.cluster_algorithms,
                "distance_metrics": spatial_analyzer.distance_metrics,
                "status": "operational"
            },
            "mapping_service": {
                "supported_data_types": list(mapping_service.color_schemes.keys()),
                "marker_icons": mapping_service.marker_icons,
                "status": "operational"
            },
            "overall_status": "operational"
        }
        
    except Exception as e:
        logger.error(f"❌ System status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "service": "geospatial",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }