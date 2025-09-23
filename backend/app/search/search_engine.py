"""
Search Engine for Ocean-Bio Platform
Provides unified search interface across all data types
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

from .elasticsearch_service import elasticsearch_service

logger = logging.getLogger(__name__)

class SearchEngine:
    """
    Unified search engine for the Ocean-Bio platform
    Handles intelligent search across all data types
    """
    
    def __init__(self):
        self.is_initialized = False
        self.elasticsearch_available = False
        
    async def initialize(self):
        """Initialize the search engine"""
        try:
            logger.info("🔍 Initializing search engine...")
            
            # Try to initialize Elasticsearch
            self.elasticsearch_available = await elasticsearch_service.initialize()
            
            if self.elasticsearch_available:
                logger.info("✅ Search engine initialized with Elasticsearch")
            else:
                logger.info("⚠️ Search engine initialized in fallback mode (no Elasticsearch)")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize search engine: {str(e)}")
            self.is_initialized = True  # Continue in fallback mode
            return False
    
    async def search(
        self,
        query: str,
        data_type: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        size: int = 10,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Perform intelligent search
        
        Args:
            query: Search query string
            data_type: Optional filter by data type ('species', 'specimens', etc.)
            filters: Additional filters
            size: Number of results to return
            offset: Result offset for pagination
            
        Returns:
            Search results dictionary
        """
        try:
            if self.elasticsearch_available:
                return await elasticsearch_service.search(
                    query=query,
                    index=data_type,
                    filters=filters,
                    size=size,
                    offset=offset
                )
            else:
                # Fallback search implementation
                return await self._fallback_search(query, data_type, filters, size, offset)
                
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return {
                'hits': {'total': {'value': 0}, 'hits': []},
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def suggest(self, query: str, size: int = 5) -> List[str]:
        """
        Get search suggestions
        
        Args:
            query: Partial query string
            size: Number of suggestions
            
        Returns:
            List of suggested queries
        """
        try:
            if self.elasticsearch_available:
                return await elasticsearch_service.suggest(query, size)
            else:
                # Fallback suggestions
                return await self._fallback_suggestions(query, size)
                
        except Exception as e:
            logger.error(f"Error getting suggestions: {str(e)}")
            return []
    
    async def search_species(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        size: int = 10
    ) -> Dict[str, Any]:
        """
        Search specifically for species data
        
        Args:
            query: Search query
            filters: Species-specific filters
            size: Number of results
            
        Returns:
            Species search results
        """
        return await self.search(
            query=query,
            data_type='species',
            filters=filters,
            size=size
        )
    
    async def search_specimens(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        size: int = 10
    ) -> Dict[str, Any]:
        """
        Search specifically for specimen data
        """
        return await self.search(
            query=query,
            data_type='specimens',
            filters=filters,
            size=size
        )
    
    async def geospatial_search(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 10.0,
        data_type: Optional[str] = None,
        size: int = 10
    ) -> Dict[str, Any]:
        """
        Perform geospatial search
        
        Args:
            latitude: Search center latitude
            longitude: Search center longitude
            radius_km: Search radius in kilometers
            data_type: Optional data type filter
            size: Number of results
            
        Returns:
            Geospatial search results
        """
        try:
            if self.elasticsearch_available:
                return await elasticsearch_service.geo_search(
                    lat=latitude,
                    lon=longitude,
                    distance=f"{radius_km}km",
                    index=data_type,
                    size=size
                )
            else:
                # Fallback geo search
                return await self._fallback_geo_search(latitude, longitude, radius_km, size)
                
        except Exception as e:
            logger.error(f"Error in geospatial search: {str(e)}")
            return {
                'hits': {'total': {'value': 0}, 'hits': []},
                'error': str(e)
            }
    
    async def _fallback_search(
        self,
        query: str,
        data_type: Optional[str],
        filters: Optional[Dict[str, Any]],
        size: int,
        offset: int
    ) -> Dict[str, Any]:
        """Fallback search implementation when Elasticsearch is not available"""
        logger.info(f"Performing fallback search for: {query}")
        
        # Mock search results for demonstration
        mock_results = [
            {
                '_id': f'mock_result_{i}',
                '_source': {
                    'title': f'Mock Species {i}',
                    'description': f'This is a mock search result for query: {query}',
                    'type': data_type or 'species',
                    'score': 1.0 - (i * 0.1)
                }
            }
            for i in range(min(size, 3))  # Return max 3 mock results
        ]
        
        return {
            'hits': {
                'total': {'value': len(mock_results)},
                'hits': mock_results[offset:offset + size]
            },
            'took': 1,
            'timed_out': False,
            'fallback': True,
            'message': 'Elasticsearch not available - using fallback search'
        }
    
    async def _fallback_suggestions(self, query: str, size: int) -> List[str]:
        """Fallback suggestions when Elasticsearch is not available"""
        suggestions = [
            f"{query} species",
            f"{query} fish",
            f"{query} marine",
            f"{query} ocean",
            f"{query} biology"
        ]
        return suggestions[:size]
    
    async def _fallback_geo_search(
        self,
        latitude: float,
        longitude: float,
        radius_km: float,
        size: int
    ) -> Dict[str, Any]:
        """Fallback geospatial search"""
        logger.info(f"Fallback geo search near {latitude}, {longitude}")
        
        return {
            'hits': {
                'total': {'value': 0},
                'hits': []
            },
            'fallback': True,
            'message': 'Geospatial search requires Elasticsearch'
        }
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        return {
            'is_initialized': self.is_initialized,
            'elasticsearch_available': self.elasticsearch_available,
            'fallback_mode': not self.elasticsearch_available,
            'capabilities': {
                'text_search': True,
                'suggestions': True,
                'geospatial_search': self.elasticsearch_available,
                'advanced_filtering': self.elasticsearch_available,
                'faceted_search': self.elasticsearch_available
            }
        }

# Global search engine instance
search_engine = SearchEngine()