"""
Indexing Service for Ocean-Bio Platform
Handles data indexing for search functionality
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

from .elasticsearch_service import elasticsearch_service

logger = logging.getLogger(__name__)

class IndexingService:
    """
    Service for indexing data into search indices
    """
    
    def __init__(self):
        self.is_initialized = False
        self.elasticsearch_available = False
        self.indexed_document_count = 0
        
    async def initialize(self):
        """Initialize the indexing service"""
        try:
            logger.info("📚 Initializing indexing service...")
            
            # Check Elasticsearch availability
            self.elasticsearch_available = elasticsearch_service.is_connected
            
            if self.elasticsearch_available:
                logger.info("✅ Indexing service initialized with Elasticsearch")
            else:
                logger.info("⚠️ Indexing service initialized in fallback mode")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize indexing service: {str(e)}")
            return False
    
    async def index_document(
        self,
        index_name: str,
        document_id: str,
        document: Dict[str, Any]
    ) -> bool:
        """
        Index a single document
        
        Args:
            index_name: Name of the index
            document_id: Unique document identifier
            document: Document data to index
            
        Returns:
            True if indexed successfully
        """
        try:
            if self.elasticsearch_available:
                result = await elasticsearch_service.index_document(
                    index=index_name,
                    doc_id=document_id,
                    document=document
                )
                
                if result:
                    self.indexed_document_count += 1
                    logger.debug(f"Indexed document {document_id} in {index_name}")
                    return True
                else:
                    logger.error(f"Failed to index document {document_id}")
                    return False
            else:
                # In fallback mode, we just log
                logger.info(f"Fallback mode: Would index document {document_id} in {index_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error indexing document {document_id}: {str(e)}")
            return False
    
    async def index_species(self, species_data: Dict[str, Any]) -> bool:
        """Index species data"""
        return await self.index_document(
            index_name='species',
            document_id=str(species_data.get('id', 'unknown')),
            document=species_data
        )
    
    async def index_specimen(self, specimen_data: Dict[str, Any]) -> bool:
        """Index specimen data"""
        return await self.index_document(
            index_name='specimens',
            document_id=str(specimen_data.get('id', 'unknown')),
            document=specimen_data
        )
    
    async def index_fishery_data(self, fishery_data: Dict[str, Any]) -> bool:
        """Index fisheries data"""
        return await self.index_document(
            index_name='fisheries',
            document_id=str(fishery_data.get('id', 'unknown')),
            document=fishery_data
        )
    
    async def index_research_data(self, research_data: Dict[str, Any]) -> bool:
        """Index research data"""
        return await self.index_document(
            index_name='research',
            document_id=str(research_data.get('id', 'unknown')),
            document=research_data
        )
    
    async def bulk_index(
        self,
        index_name: str,
        documents: List[Dict[str, Any]],
        id_field: str = 'id'
    ) -> Dict[str, Any]:
        """
        Bulk index multiple documents
        
        Args:
            index_name: Name of the index
            documents: List of documents to index
            id_field: Field to use as document ID
            
        Returns:
            Bulk indexing results
        """
        try:
            if self.elasticsearch_available:
                results = await elasticsearch_service.bulk_index(
                    index=index_name,
                    documents=documents,
                    id_field=id_field
                )
                
                indexed_count = results.get('indexed', 0)
                self.indexed_document_count += indexed_count
                
                logger.info(f"Bulk indexed {indexed_count} documents in {index_name}")
                return results
            else:
                # Fallback mode
                logger.info(f"Fallback mode: Would bulk index {len(documents)} documents in {index_name}")
                return {
                    'indexed': len(documents),
                    'errors': 0,
                    'fallback': True
                }
                
        except Exception as e:
            logger.error(f"Error in bulk indexing: {str(e)}")
            return {
                'indexed': 0,
                'errors': len(documents),
                'error': str(e)
            }
    
    async def delete_document(self, index_name: str, document_id: str) -> bool:
        """Delete a document from the index"""
        try:
            if self.elasticsearch_available:
                return await elasticsearch_service.delete_document(index_name, document_id)
            else:
                logger.info(f"Fallback mode: Would delete document {document_id} from {index_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False
    
    async def update_document(
        self,
        index_name: str,
        document_id: str,
        update_data: Dict[str, Any]
    ) -> bool:
        """Update a document in the index"""
        try:
            if self.elasticsearch_available:
                return await elasticsearch_service.update_document(
                    index=index_name,
                    doc_id=document_id,
                    update_data=update_data
                )
            else:
                logger.info(f"Fallback mode: Would update document {document_id} in {index_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating document {document_id}: {str(e)}")
            return False
    
    async def reindex_all(self, source_data: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """
        Reindex all data
        
        Args:
            source_data: Optional source data to reindex
            
        Returns:
            Reindexing results
        """
        try:
            logger.info("🔄 Starting full reindex...")
            results = {
                'species': {'indexed': 0, 'errors': 0},
                'specimens': {'indexed': 0, 'errors': 0},
                'fisheries': {'indexed': 0, 'errors': 0},
                'research': {'indexed': 0, 'errors': 0},
                'total_time': 0
            }
            
            start_time = datetime.now()
            
            if source_data:
                # Index provided data
                for data_type, documents in source_data.items():
                    if documents:
                        result = await self.bulk_index(data_type, documents)
                        results[data_type] = result
            else:
                # Mock reindexing in demo mode
                logger.info("Mock reindexing - no source data provided")
                results['message'] = 'Mock reindexing completed'
            
            end_time = datetime.now()
            results['total_time'] = (end_time - start_time).total_seconds()
            
            logger.info(f"✅ Reindexing completed in {results['total_time']:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error during reindexing: {str(e)}")
            return {'error': str(e)}
    
    def get_indexing_stats(self) -> Dict[str, Any]:
        """Get indexing service statistics"""
        return {
            'is_initialized': self.is_initialized,
            'elasticsearch_available': self.elasticsearch_available,
            'total_indexed_documents': self.indexed_document_count,
            'indices_available': [
                'species', 'specimens', 'fisheries', 'research', 'observations'
            ] if self.elasticsearch_available else [],
            'fallback_mode': not self.elasticsearch_available
        }

# Global indexing service instance
indexing_service = IndexingService()