"""
Search API Router
Provides intelligent search, suggestions, and analytics endpoints
"""
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Body, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime

from .elasticsearch_service import elasticsearch_service

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/search", tags=["Search"])

# Pydantic models for requests/responses
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    data_types: Optional[List[str]] = Field(None, description="Data types to search in")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    size: int = Field(20, ge=1, le=100, description="Number of results to return")
    from_: int = Field(0, ge=0, description="Offset for pagination", alias="from")
    sort: Optional[List[Dict[str, Any]]] = Field(None, description="Sort configuration")

class SearchResponse(BaseModel):
    success: bool
    total_hits: int
    took_ms: int
    results: List[Dict[str, Any]]
    facets: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    query_info: Dict[str, Any]

class SuggestionRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Prefix for suggestions")
    data_type: str = Field("species", description="Data type for suggestions")
    limit: int = Field(10, ge=1, le=20, description="Maximum suggestions to return")

class IndexRequest(BaseModel):
    data_type: str = Field(..., description="Type of data to index")
    documents: List[Dict[str, Any]] = Field(..., description="Documents to index")

class BulkIndexResponse(BaseModel):
    success: bool
    indexed_count: int
    failed_count: int
    total_requested: int
    processing_time_seconds: float

# Search endpoints

@router.post("/", response_model=SearchResponse)
async def search_data(search_request: SearchRequest):
    """
    Perform intelligent search across Ocean-Bio data
    
    Supports:
    - Multi-field text search with fuzzy matching
    - Faceted search with filters
    - Geospatial search
    - Taxonomic search
    - Pagination and sorting
    """
    try:
        start_time = datetime.now()
        
        logger.info(f"Search request: '{search_request.query}' in {search_request.data_types}")
        
        # Execute search
        search_result = await elasticsearch_service.search(
            query=search_request.query,
            data_types=search_request.data_types,
            filters=search_request.filters,
            size=search_request.size,
            from_=search_request.from_,
            sort=search_request.sort
        )
        
        # Process results
        hits = search_result.get('hits', {})
        total_hits = hits.get('total', {}).get('value', 0)
        results = []
        
        for hit in hits.get('hits', []):
            result_item = {
                'id': hit.get('_id'),
                'type': hit.get('_index', '').replace('ocean_bio_', ''),
                'score': hit.get('_score'),
                'source': hit.get('_source', {}),
                'highlight': hit.get('highlight', {})
            }
            results.append(result_item)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Build facets from aggregations if available
        facets = None
        if 'aggregations' in search_result:
            facets = search_result['aggregations']
        
        response = SearchResponse(
            success=True,
            total_hits=total_hits,
            took_ms=int(processing_time * 1000),
            results=results,
            facets=facets,
            query_info={
                'query': search_request.query,
                'data_types': search_request.data_types,
                'filters': search_request.filters,
                'size': search_request.size,
                'from': search_request.from_
            }
        )
        
        logger.info(f"Search completed: {total_hits} results in {processing_time:.3f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/quick")
async def quick_search(
    q: str = Query(..., min_length=1, description="Search query"),
    type: Optional[str] = Query(None, description="Data type to search"),
    limit: int = Query(10, ge=1, le=50, description="Result limit")
):
    """Quick search with minimal parameters"""
    try:
        data_types = [type] if type else None
        
        search_result = await elasticsearch_service.search(
            query=q,
            data_types=data_types,
            size=limit
        )
        
        hits = search_result.get('hits', {})
        results = []
        
        for hit in hits.get('hits', []):
            results.append({
                'id': hit.get('_id'),
                'type': hit.get('_index', '').replace('ocean_bio_', ''),
                'title': hit.get('_source', {}).get('common_name') or 
                        hit.get('_source', {}).get('scientific_name') or 
                        hit.get('_source', {}).get('project_title', 'Unknown'),
                'description': hit.get('_source', {}).get('description', ''),
                'score': hit.get('_score')
            })
        
        return {
            'success': True,
            'query': q,
            'total': hits.get('total', {}).get('value', 0),
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error in quick search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/species")
async def search_species(
    q: str = Query(..., description="Species search query"),
    family: Optional[str] = Query(None, description="Filter by family"),
    conservation_status: Optional[str] = Query(None, description="Filter by conservation status"),
    habitat: Optional[str] = Query(None, description="Filter by habitat"),
    size: int = Query(20, ge=1, le=100)
):
    """Specialized species search with taxonomic filters"""
    try:
        filters = {}
        
        if family:
            filters['family'] = family
        if conservation_status:
            filters['conservation_status'] = conservation_status
        if habitat:
            filters['habitat'] = habitat
        
        search_result = await elasticsearch_service.search(
            query=q,
            data_types=['species'],
            filters=filters,
            size=size
        )
        
        hits = search_result.get('hits', {})
        species_results = []
        
        for hit in hits.get('hits', []):
            source = hit.get('_source', {})
            species_results.append({
                'id': source.get('id'),
                'common_name': source.get('common_name'),
                'scientific_name': source.get('scientific_name'),
                'family': source.get('family'),
                'order': source.get('order'),
                'conservation_status': source.get('conservation_status'),
                'habitat': source.get('habitat'),
                'description': source.get('description', '')[:200] + '...' if len(source.get('description', '')) > 200 else source.get('description', ''),
                'score': hit.get('_score'),
                'highlight': hit.get('highlight', {})
            })
        
        return {
            'success': True,
            'total_species': hits.get('total', {}).get('value', 0),
            'species': species_results,
            'filters_applied': filters
        }
        
    except Exception as e:
        logger.error(f"Error in species search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/geospatial")
async def geospatial_search(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    distance: str = Query("10km", description="Search radius"),
    data_types: Optional[str] = Query(None, description="Comma-separated data types"),
    size: int = Query(20, ge=1, le=100)
):
    """Search for data within geographic proximity"""
    try:
        filters = {
            'location': {
                'geo_distance': {
                    'distance': distance,
                    'location': {'lat': lat, 'lon': lon}
                }
            }
        }
        
        search_data_types = data_types.split(',') if data_types else None
        
        search_result = await elasticsearch_service.search(
            query="*",  # Match all for geospatial search
            data_types=search_data_types,
            filters=filters,
            size=size
        )
        
        hits = search_result.get('hits', {})
        geo_results = []
        
        for hit in hits.get('hits', []):
            source = hit.get('_source', {})
            location = source.get('location', {})
            
            geo_results.append({
                'id': source.get('id'),
                'type': hit.get('_index', '').replace('ocean_bio_', ''),
                'title': source.get('common_name') or source.get('scientific_name') or source.get('project_title', 'Unknown'),
                'location': location,
                'distance_sort': hit.get('sort', [None])[0] if hit.get('sort') else None,
                'source': source
            })
        
        return {
            'success': True,
            'search_center': {'lat': lat, 'lon': lon},
            'search_radius': distance,
            'total_found': hits.get('total', {}).get('value', 0),
            'results': geo_results
        }
        
    except Exception as e:
        logger.error(f"Error in geospatial search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Suggestion endpoints

@router.post("/suggest", response_model=List[str])
async def get_suggestions(suggestion_request: SuggestionRequest):
    """Get search suggestions for autocomplete"""
    try:
        suggestions = await elasticsearch_service.suggest(
            query=suggestion_request.query,
            data_type=suggestion_request.data_type
        )
        
        return suggestions[:suggestion_request.limit]
        
    except Exception as e:
        logger.error(f"Error getting suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/suggest/{query}")
async def get_quick_suggestions(
    query: str,
    type: str = Query("species", description="Data type for suggestions"),
    limit: int = Query(10, ge=1, le=20)
):
    """Quick suggestions endpoint"""
    try:
        suggestions = await elasticsearch_service.suggest(query, type)
        return {
            'success': True,
            'query': query,
            'suggestions': suggestions[:limit]
        }
        
    except Exception as e:
        logger.error(f"Error getting quick suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Indexing endpoints

@router.post("/index", response_model=BulkIndexResponse)
async def bulk_index_documents(
    background_tasks: BackgroundTasks,
    index_request: IndexRequest
):
    """Bulk index documents into Elasticsearch"""
    try:
        start_time = datetime.now()
        
        logger.info(f"Indexing {len(index_request.documents)} documents of type {index_request.data_type}")
        
        # Validate data type
        if index_request.data_type not in elasticsearch_service.index_configs:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid data type. Available types: {list(elasticsearch_service.index_configs.keys())}"
            )
        
        # Perform bulk indexing
        indexed_count = await elasticsearch_service.bulk_index_documents(
            data_type=index_request.data_type,
            documents=index_request.documents
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        failed_count = len(index_request.documents) - indexed_count
        
        response = BulkIndexResponse(
            success=indexed_count > 0,
            indexed_count=indexed_count,
            failed_count=failed_count,
            total_requested=len(index_request.documents),
            processing_time_seconds=processing_time
        )
        
        logger.info(f"Bulk indexing completed: {indexed_count}/{len(index_request.documents)} successful")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk indexing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/index/{data_type}/{doc_id}")
async def index_single_document(
    data_type: str,
    doc_id: str,
    document: Dict[str, Any] = Body(...)
):
    """Index a single document"""
    try:
        if data_type not in elasticsearch_service.index_configs:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid data type. Available types: {list(elasticsearch_service.index_configs.keys())}"
            )
        
        success = await elasticsearch_service.index_document(
            data_type=data_type,
            doc_id=doc_id,
            document=document
        )
        
        if success:
            return {
                'success': True,
                'message': f'Document {doc_id} indexed successfully',
                'data_type': data_type,
                'doc_id': doc_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to index document")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error indexing single document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoints

@router.get("/analytics")
async def get_search_analytics(
    data_type: Optional[str] = Query(None, description="Data type for analytics"),
    time_range: str = Query("7d", description="Time range for analytics")
):
    """Get search and indexing analytics"""
    try:
        analytics = await elasticsearch_service.get_search_analytics(
            data_type=data_type,
            time_range=time_range
        )
        
        return {
            'success': True,
            'time_range': time_range,
            'data_type': data_type,
            'analytics': analytics,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting search analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/cluster-health")
async def get_cluster_health():
    """Get Elasticsearch cluster health status"""
    try:
        health = await elasticsearch_service.get_cluster_health()
        
        return {
            'success': True,
            'cluster_health': health,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting cluster health: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Data discovery endpoints

@router.get("/discover/species-families")
async def discover_species_families():
    """Discover available species families"""
    try:
        analytics = await elasticsearch_service.get_search_analytics(
            data_type='species',
            time_range='30d'
        )
        
        families = []
        if 'specific_aggs' in analytics and 'top_families' in analytics['specific_aggs']:
            families = [
                {
                    'family': bucket['key'],
                    'species_count': bucket['doc_count']
                }
                for bucket in analytics['specific_aggs']['top_families'].get('buckets', [])
            ]
        
        return {
            'success': True,
            'families': families,
            'total_families': len(families)
        }
        
    except Exception as e:
        logger.error(f"Error discovering species families: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/discover/data-types")
async def discover_data_types():
    """Discover available data types and their document counts"""
    try:
        analytics = await elasticsearch_service.get_search_analytics(time_range='30d')
        
        data_types = []
        if 'by_type' in analytics:
            for bucket in analytics['by_type']:
                index_name = bucket['key']
                data_type = index_name.replace('ocean_bio_', '')
                data_types.append({
                    'data_type': data_type,
                    'document_count': bucket['doc_count'],
                    'index_name': index_name
                })
        
        return {
            'success': True,
            'data_types': data_types,
            'total_documents': analytics.get('total_documents', 0)
        }
        
    except Exception as e:
        logger.error(f"Error discovering data types: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint

@router.get("/health")
async def search_health_check():
    """Health check for search service"""
    try:
        if not elasticsearch_service.is_connected:
            return {
                'status': 'unhealthy',
                'elasticsearch': 'disconnected',
                'timestamp': datetime.now().isoformat()
            }
        
        # Try a simple search to verify functionality
        test_result = await elasticsearch_service.search("*", size=1)
        
        return {
            'status': 'healthy',
            'elasticsearch': 'connected',
            'indices_available': list(elasticsearch_service.index_configs.keys()),
            'test_search_successful': 'hits' in test_result,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Search health check failed: {str(e)}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }