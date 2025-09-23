"""
Smart Search System for Ocean-Bio Platform
Provides Elasticsearch integration and intelligent search capabilities
"""
from .search_engine import SearchEngine, search_engine
from .indexing_service import IndexingService, indexing_service
from .elasticsearch_service import elasticsearch_service
from .router import router as search_router

__all__ = ["SearchEngine", "search_engine", "IndexingService", "indexing_service", "elasticsearch_service", "search_router"]
