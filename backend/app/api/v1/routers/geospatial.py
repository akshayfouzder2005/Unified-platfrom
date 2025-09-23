"""
🗺️ Geospatial Analysis API Router

RESTful API endpoints for geospatial analysis, GIS operations, and mapping functionality.
Provides access to spatial data analysis, location-based queries, and geographic visualizations.

Author: Ocean-Bio Development Team
Version: 2.0.0
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, validator
from datetime import datetime
import logging

from ....geospatial.gis_integration import gis_service
from ....geospatial.spatial_analysis import spatial_analyzer
from ....geospatial.mapping_service import mapping_service
from ....core.database import get_db
from ....core.auth import get_current_user
from ....models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/geospatial", tags=["geospatial"])

# Pydantic models for API requests/responses

class LocationQuery(BaseModel):
    """Location-based query parameters"""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    radius_km: float = Field(10.0, gt=0, le=1000, description="Search radius in kilometers")
    data_types: Optional[List[str]] = Field(None, description="Data types to include")
    start_date: Optional[datetime] = Field(None, description="Start date filter")
    end_date: Optional[datetime] = Field(None, description="End date filter")

class BoundingBoxQuery(BaseModel):
    """Bounding box query parameters"""
    min_lat: float = Field(..., ge=-90, le=90)
    max_lat: float = Field(..., ge=-90, le=90)
    min_lon: float = Field(..., ge=-180, le=180)
    max_lon: float = Field(..., ge=-180, le=180)
    data_types: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    @validator('max_lat')
    def validate_lat_range(cls, v, values):
        if 'min_lat' in values and v <= values['min_lat']:
            raise ValueError('max_lat must be greater than min_lat')
        return v
    
    @validator('max_lon')
    def validate_lon_range(cls, v, values):
        if 'min_lon' in values and v <= values['min_lon']:
            raise ValueError('max_lon must be greater than min_lon')
        return v

class SpatialAnalysisRequest(BaseModel):
    """Spatial analysis request parameters"""
    analysis_type: str = Field(..., description="Type of spatial analysis")
    locations: List[Tuple[float, float]] = Field(..., description="List of (lat, lon) coordinates")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Analysis parameters")

class MapGenerationRequest(BaseModel):
    """Map generation request parameters"""
    map_type: str = Field(..., description="Type of map to generate")
    region: BoundingBoxQuery = Field(..., description="Geographic region")
    layers: List[str] = Field(default_factory=list, description="Map layers to include")
    style_options: Dict[str, Any] = Field(default_factory=dict, description="Map styling options")

# API Endpoints

@router.get("/health")
async def geospatial_health_check():
    """Health check for geospatial services"""
    try:
        # Test GIS connection
        gis_status = gis_service.test_connection()
        
        # Test spatial analyzer
        analyzer_status = spatial_analyzer.get_status()
        
        # Test mapping service
        mapping_status = mapping_service.get_status()
        
        return {
            "status": "healthy",
            "services": {
                "gis_integration": gis_status,
                "spatial_analysis": analyzer_status,
                "mapping_service": mapping_status
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Geospatial health check failed: {e}")
        raise HTTPException(status_code=503, detail="Geospatial services unavailable")

@router.post("/query/location")
async def query_by_location(
    query: LocationQuery,
    current_user: User = Depends(get_current_user)
):
    """Query data by geographic location with radius"""
    try:
        logger.info(f"🗺️ Location query: ({query.latitude}, {query.longitude}) radius={query.radius_km}km")
        
        # Perform spatial query
        results = await gis_service.query_by_location(
            latitude=query.latitude,
            longitude=query.longitude,
            radius_km=query.radius_km,
            data_types=query.data_types,
            start_date=query.start_date,
            end_date=query.end_date
        )
        
        return {
            "query_parameters": query.dict(),
            "results": results,
            "total_records": len(results.get("features", [])),
            "query_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Location query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query/bbox")
async def query_by_bounding_box(
    query: BoundingBoxQuery,
    current_user: User = Depends(get_current_user)
):
    """Query data by bounding box"""
    try:
        logger.info(f"🗺️ Bounding box query: ({query.min_lat},{query.min_lon}) to ({query.max_lat},{query.max_lon})")
        
        # Perform spatial query
        results = await gis_service.query_by_bounding_box(
            min_lat=query.min_lat,
            max_lat=query.max_lat,
            min_lon=query.min_lon,
            max_lon=query.max_lon,
            data_types=query.data_types,
            start_date=query.start_date,
            end_date=query.end_date
        )
        
        return {
            "query_parameters": query.dict(),
            "results": results,
            "total_records": len(results.get("features", [])),
            "query_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Bounding box query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/spatial")
async def perform_spatial_analysis(
    request: SpatialAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Perform spatial analysis on geographic data"""
    try:
        logger.info(f"🔬 Spatial analysis: {request.analysis_type}")
        
        # Validate analysis type
        available_analyses = spatial_analyzer.get_available_analyses()
        if request.analysis_type not in available_analyses:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid analysis type. Available: {available_analyses}"
            )
        
        # Perform analysis
        analysis_results = await spatial_analyzer.perform_analysis(
            analysis_type=request.analysis_type,
            locations=request.locations,
            parameters=request.parameters
        )
        
        return {
            "analysis_type": request.analysis_type,
            "input_locations": len(request.locations),
            "results": analysis_results,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Spatial analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/maps/generate")
async def generate_map(
    request: MapGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Generate interactive map with specified layers and styling"""
    try:
        logger.info(f"🗺️ Map generation: {request.map_type}")
        
        # Generate map
        map_result = await mapping_service.generate_map(
            map_type=request.map_type,
            region=request.region.dict(),
            layers=request.layers,
            style_options=request.style_options
        )
        
        return {
            "map_id": map_result.get("map_id"),
            "map_url": map_result.get("map_url"),
            "map_type": request.map_type,
            "region": request.region.dict(),
            "layers": request.layers,
            "generation_timestamp": datetime.now().isoformat(),
            "metadata": map_result.get("metadata", {})
        }
        
    except Exception as e:
        logger.error(f"❌ Map generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/maps/{map_id}")
async def get_map_details(
    map_id: str = Path(..., description="Map ID"),
    current_user: User = Depends(get_current_user)
):
    """Get details of a generated map"""
    try:
        map_details = await mapping_service.get_map_details(map_id)
        
        if not map_details:
            raise HTTPException(status_code=404, detail="Map not found")
        
        return map_details
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get map details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/layers/available")
async def get_available_layers(current_user: User = Depends(get_current_user)):
    """Get list of available map layers"""
    try:
        layers = mapping_service.get_available_layers()
        
        return {
            "available_layers": layers,
            "total_layers": len(layers),
            "retrieved_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get available layers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analysis/types")
async def get_analysis_types(current_user: User = Depends(get_current_user)):
    """Get available spatial analysis types"""
    try:
        analyses = spatial_analyzer.get_available_analyses()
        
        analysis_details = {}
        for analysis_type in analyses:
            details = spatial_analyzer.get_analysis_info(analysis_type)
            analysis_details[analysis_type] = details
        
        return {
            "available_analyses": analyses,
            "analysis_details": analysis_details,
            "total_analyses": len(analyses),
            "retrieved_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get analysis types: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/data/summary")
async def get_geospatial_data_summary(
    data_type: Optional[str] = Query(None, description="Filter by data type"),
    region: Optional[str] = Query(None, description="Region filter"),
    current_user: User = Depends(get_current_user)
):
    """Get summary of available geospatial data"""
    try:
        summary = await gis_service.get_data_summary(
            data_type=data_type,
            region=region
        )
        
        return {
            "data_summary": summary,
            "filters_applied": {
                "data_type": data_type,
                "region": region
            },
            "summary_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get data summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/maps/{map_id}")
async def delete_map(
    map_id: str = Path(..., description="Map ID to delete"),
    current_user: User = Depends(get_current_user)
):
    """Delete a generated map"""
    try:
        success = await mapping_service.delete_map(map_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Map not found")
        
        return {
            "message": "Map deleted successfully",
            "map_id": map_id,
            "deletion_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete map: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints

@router.get("/coordinates/validate")
async def validate_coordinates(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180)
):
    """Validate geographic coordinates"""
    try:
        validation_result = gis_service.validate_coordinates(latitude, longitude)
        
        return {
            "coordinates": {
                "latitude": latitude,
                "longitude": longitude
            },
            "is_valid": validation_result.get("is_valid", False),
            "location_info": validation_result.get("location_info", {}),
            "validation_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Coordinate validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/distance/calculate")
async def calculate_distance(
    lat1: float = Query(..., ge=-90, le=90),
    lon1: float = Query(..., ge=-180, le=180),
    lat2: float = Query(..., ge=-90, le=90),
    lon2: float = Query(..., ge=-180, le=180)
):
    """Calculate distance between two points"""
    try:
        distance_result = gis_service.calculate_distance(lat1, lon1, lat2, lon2)
        
        return {
            "point1": {"latitude": lat1, "longitude": lon1},
            "point2": {"latitude": lat2, "longitude": lon2},
            "distance_km": distance_result.get("distance_km"),
            "distance_nm": distance_result.get("distance_nm"),
            "bearing": distance_result.get("bearing"),
            "calculation_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Distance calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))