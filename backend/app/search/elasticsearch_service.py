"""
Elasticsearch Service for Ocean-Bio Platform
Provides intelligent search, semantic search, and data discovery capabilities
"""
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from elasticsearch import AsyncElasticsearch, Elasticsearch
try:
    from elasticsearch.exceptions import ConnectionError as ESConnectionError, NotFoundError
except ImportError:
    # Fallback for different elasticsearch versions
    ESConnectionError = Exception
    NotFoundError = Exception
import json

logger = logging.getLogger(__name__)

class ElasticsearchService:
    """
    Advanced Elasticsearch service for Ocean-Bio platform
    Handles indexing, searching, and analytics
    """
    
    def __init__(self, elasticsearch_url: str = "http://localhost:9200"):
        self.elasticsearch_url = elasticsearch_url
        self.client: Optional[AsyncElasticsearch] = None
        self.sync_client: Optional[Elasticsearch] = None
        self.is_connected = False
        
        # Index configurations for all 5 data model types plus additional indices
        self.index_configs = {
            'edna': {
                'index': 'ocean_bio_edna',
                'doc_type': '_doc',
                'mapping': self._get_edna_mapping()
            },
            'oceanographic': {
                'index': 'ocean_bio_oceanographic',
                'doc_type': '_doc',
                'mapping': self._get_oceanographic_mapping()
            },
            'otolith': {
                'index': 'ocean_bio_otolith',
                'doc_type': '_doc',
                'mapping': self._get_otolith_mapping()
            },
            'taxonomy': {
                'index': 'ocean_bio_taxonomy',
                'doc_type': '_doc',
                'mapping': self._get_taxonomy_mapping()
            },
            'fisheries': {
                'index': 'ocean_bio_fisheries',
                'doc_type': '_doc',
                'mapping': self._get_fisheries_mapping()
            },
            # Additional indices for related data
            'species': {
                'index': 'ocean_bio_species',
                'doc_type': '_doc',
                'mapping': self._get_species_mapping()
            },
            'specimens': {
                'index': 'ocean_bio_specimens',
                'doc_type': '_doc',
                'mapping': self._get_specimens_mapping()
            },
            'research': {
                'index': 'ocean_bio_research',
                'doc_type': '_doc',
                'mapping': self._get_research_mapping()
            },
            'observations': {
                'index': 'ocean_bio_observations',
                'doc_type': '_doc',
                'mapping': self._get_observations_mapping()
            }
        }
        
        # Search configuration
        self.default_search_config = {
            'size': 20,
            'from': 0,
            'highlight': {
                'fields': {
                    '*': {}
                },
                'pre_tags': ['<mark>'],
                'post_tags': ['</mark>']
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize Elasticsearch connection and create indices"""
        try:
            # Create async client
            self.client = AsyncElasticsearch(
                hosts=[self.elasticsearch_url],
                timeout=30,
                retry_on_timeout=True
            )
            
            # Create sync client for some operations
            self.sync_client = Elasticsearch(
                hosts=[self.elasticsearch_url],
                timeout=30,
                retry_on_timeout=True
            )
            
            # Test connection
            if await self.client.ping():
                self.is_connected = True
                logger.info("Elasticsearch connection established")
                
                # Create indices if they don't exist
                await self._create_indices()
                
                return True
            else:
                logger.error("Failed to ping Elasticsearch")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch: {str(e)}")
            return False
    
    async def close(self):
        """Close Elasticsearch connections"""
        if self.client:
            await self.client.close()
        if self.sync_client:
            self.sync_client.close()
        
        self.is_connected = False
        logger.info("Elasticsearch connections closed")
    
    async def _create_indices(self):
        """Create all necessary indices with proper mappings"""
        for data_type, config in self.index_configs.items():
            try:
                index_name = config['index']
                
                # Check if index exists
                if not await self.client.indices.exists(index=index_name):
                    # Create index with mapping
                    await self.client.indices.create(
                        index=index_name,
                        body={
                            'settings': {
                                'number_of_shards': 1,
                                'number_of_replicas': 0,
                                'analysis': self._get_analysis_settings()
                            },
                            'mappings': config['mapping']
                        }
                    )
                    logger.info(f"Created Elasticsearch index: {index_name}")
                else:
                    logger.info(f"Elasticsearch index already exists: {index_name}")
                    
            except Exception as e:
                logger.error(f"Error creating index {data_type}: {str(e)}")
    
    def _get_analysis_settings(self) -> Dict[str, Any]:
        """Get custom analysis settings for better search"""
        return {
            'analyzer': {
                'scientific_name_analyzer': {
                    'type': 'custom',
                    'tokenizer': 'standard',
                    'filter': ['lowercase', 'scientific_name_filter']
                },
                'location_analyzer': {
                    'type': 'custom',
                    'tokenizer': 'standard',
                    'filter': ['lowercase', 'location_filter']
                }
            },
            'filter': {
                'scientific_name_filter': {
                    'type': 'pattern_replace',
                    'pattern': r'[^\w\s]',
                    'replacement': ''
                },
                'location_filter': {
                    'type': 'pattern_replace',
                    'pattern': r'[^\w\s\-]',
                    'replacement': ''
                }
            }
        }
    
    def _get_species_mapping(self) -> Dict[str, Any]:
        """Get mapping for species index"""
        return {
            'properties': {
                'id': {'type': 'integer'},
                'common_name': {
                    'type': 'text',
                    'analyzer': 'standard',
                    'fields': {
                        'keyword': {'type': 'keyword'},
                        'suggest': {
                            'type': 'completion',
                            'analyzer': 'standard'
                        }
                    }
                },
                'scientific_name': {
                    'type': 'text',
                    'analyzer': 'scientific_name_analyzer',
                    'fields': {
                        'keyword': {'type': 'keyword'},
                        'suggest': {
                            'type': 'completion',
                            'analyzer': 'scientific_name_analyzer'
                        }
                    }
                },
                'family': {
                    'type': 'text',
                    'fields': {'keyword': {'type': 'keyword'}}
                },
                'order': {
                    'type': 'text',
                    'fields': {'keyword': {'type': 'keyword'}}
                },
                'class': {
                    'type': 'text',
                    'fields': {'keyword': {'type': 'keyword'}}
                },
                'description': {
                    'type': 'text',
                    'analyzer': 'standard'
                },
                'habitat': {
                    'type': 'text',
                    'analyzer': 'standard',
                    'fields': {'keyword': {'type': 'keyword'}}
                },
                'distribution': {
                    'type': 'text',
                    'analyzer': 'location_analyzer'
                },
                'conservation_status': {
                    'type': 'keyword'
                },
                'commercial_importance': {
                    'type': 'keyword'
                },
                'size_range': {
                    'properties': {
                        'min_length': {'type': 'float'},
                        'max_length': {'type': 'float'},
                        'min_weight': {'type': 'float'},
                        'max_weight': {'type': 'float'}
                    }
                },
                'created_at': {'type': 'date'},
                'updated_at': {'type': 'date'},
                'tags': {'type': 'keyword'},
                'location': {'type': 'geo_point'}
            }
        }
    
    def _get_specimens_mapping(self) -> Dict[str, Any]:
        """Get mapping for specimens index"""
        return {
            'properties': {
                'id': {'type': 'integer'},
                'species_id': {'type': 'integer'},
                'species_name': {
                    'type': 'text',
                    'analyzer': 'standard',
                    'fields': {'keyword': {'type': 'keyword'}}
                },
                'scientific_name': {
                    'type': 'text',
                    'analyzer': 'scientific_name_analyzer',
                    'fields': {'keyword': {'type': 'keyword'}}
                },
                'location': {'type': 'geo_point'},
                'location_name': {
                    'type': 'text',
                    'analyzer': 'location_analyzer',
                    'fields': {'keyword': {'type': 'keyword'}}
                },
                'collection_date': {'type': 'date'},
                'collector': {
                    'type': 'text',
                    'fields': {'keyword': {'type': 'keyword'}}
                },
                'collection_method': {'type': 'keyword'},
                'depth_range': {
                    'properties': {
                        'min_depth': {'type': 'float'},
                        'max_depth': {'type': 'float'}
                    }
                },
                'measurements': {
                    'properties': {
                        'length': {'type': 'float'},
                        'weight': {'type': 'float'},
                        'width': {'type': 'float'},
                        'height': {'type': 'float'}
                    }
                },
                'condition': {'type': 'keyword'},
                'life_stage': {'type': 'keyword'},
                'sex': {'type': 'keyword'},
                'notes': {'type': 'text'},
                'images': {'type': 'keyword'},
                'created_at': {'type': 'date'}
            }
        }
    
    def _get_fisheries_mapping(self) -> Dict[str, Any]:
        """Get mapping for fisheries index"""
        return {
            'properties': {
                'id': {'type': 'integer'},
                'vessel_id': {'type': 'keyword'},
                'vessel_name': {
                    'type': 'text',
                    'fields': {'keyword': {'type': 'keyword'}}
                },
                'fishing_area': {
                    'type': 'text',
                    'analyzer': 'location_analyzer',
                    'fields': {'keyword': {'type': 'keyword'}}
                },
                'location': {'type': 'geo_point'},
                'fishing_date': {'type': 'date'},
                'species_caught': {
                    'type': 'nested',
                    'properties': {
                        'species_id': {'type': 'integer'},
                        'species_name': {
                            'type': 'text',
                            'fields': {'keyword': {'type': 'keyword'}}
                        },
                        'scientific_name': {
                            'type': 'text',
                            'analyzer': 'scientific_name_analyzer'
                        },
                        'catch_weight': {'type': 'float'},
                        'count': {'type': 'integer'},
                        'size_distribution': {
                            'type': 'nested',
                            'properties': {
                                'size_class': {'type': 'keyword'},
                                'count': {'type': 'integer'},
                                'weight': {'type': 'float'}
                            }
                        }
                    }
                },
                'total_catch_weight': {'type': 'float'},
                'fishing_effort': {
                    'properties': {
                        'duration_hours': {'type': 'float'},
                        'gear_type': {'type': 'keyword'},
                        'mesh_size': {'type': 'float'}
                    }
                },
                'environmental_conditions': {
                    'properties': {
                        'water_temperature': {'type': 'float'},
                        'depth': {'type': 'float'},
                        'weather_conditions': {'type': 'keyword'}
                    }
                },
                'quota_type': {'type': 'keyword'},
                'created_at': {'type': 'date'}
            }
        }
    
    def _get_research_mapping(self) -> Dict[str, Any]:
        """Get mapping for research index"""
        return {
            'properties': {
                'id': {'type': 'integer'},
                'project_id': {'type': 'keyword'},
                'project_title': {
                    'type': 'text',
                    'analyzer': 'standard',
                    'fields': {'keyword': {'type': 'keyword'}}
                },
                'description': {'type': 'text'},
                'principal_investigator': {
                    'type': 'text',
                    'fields': {'keyword': {'type': 'keyword'}}
                },
                'institution': {
                    'type': 'text',
                    'fields': {'keyword': {'type': 'keyword'}}
                },
                'research_area': {
                    'type': 'text',
                    'analyzer': 'location_analyzer',
                    'fields': {'keyword': {'type': 'keyword'}}
                },
                'study_species': {
                    'type': 'nested',
                    'properties': {
                        'species_id': {'type': 'integer'},
                        'scientific_name': {
                            'type': 'text',
                            'analyzer': 'scientific_name_analyzer'
                        }
                    }
                },
                'research_type': {'type': 'keyword'},
                'methodology': {'type': 'text'},
                'start_date': {'type': 'date'},
                'end_date': {'type': 'date'},
                'status': {'type': 'keyword'},
                'keywords': {'type': 'keyword'},
                'publications': {
                    'type': 'nested',
                    'properties': {
                        'title': {'type': 'text'},
                        'authors': {'type': 'text'},
                        'doi': {'type': 'keyword'},
                        'publication_date': {'type': 'date'}
                    }
                },
                'created_at': {'type': 'date'}
            }
        }
    
    def _get_taxonomy_mapping(self) -> Dict[str, Any]:
        """Get mapping for taxonomy index"""
        return {
            'properties': {
                'id': {'type': 'integer'},
                'taxon_id': {'type': 'keyword'},
                'scientific_name': {
                    'type': 'text',
                    'analyzer': 'scientific_name_analyzer',
                    'fields': {
                        'keyword': {'type': 'keyword'},
                        'suggest': {
                            'type': 'completion',
                            'analyzer': 'scientific_name_analyzer'
                        }
                    }
                },
                'common_names': {
                    'type': 'text',
                    'analyzer': 'standard',
                    'fields': {
                        'keyword': {'type': 'keyword'},
                        'suggest': {
                            'type': 'completion',
                            'analyzer': 'standard'
                        }
                    }
                },
                'rank': {'type': 'keyword'},
                'kingdom': {'type': 'keyword'},
                'phylum': {'type': 'keyword'},
                'class': {'type': 'keyword'},
                'order': {'type': 'keyword'},
                'family': {'type': 'keyword'},
                'genus': {'type': 'keyword'},
                'species': {'type': 'keyword'},
                'parent_taxon_id': {'type': 'keyword'},
                'children_count': {'type': 'integer'},
                'taxonomic_authority': {'type': 'text'},
                'year_described': {'type': 'integer'},
                'synonyms': {
                    'type': 'text',
                    'analyzer': 'scientific_name_analyzer'
                },
                'description': {'type': 'text'},
                'created_at': {'type': 'date'}
            }
        }
    
    def _get_observations_mapping(self) -> Dict[str, Any]:
        """Get mapping for observations index"""
        return {
            'properties': {
                'id': {'type': 'integer'},
                'observer_id': {'type': 'integer'},
                'species_id': {'type': 'integer'},
                'scientific_name': {
                    'type': 'text',
                    'analyzer': 'scientific_name_analyzer',
                    'fields': {'keyword': {'type': 'keyword'}}
                },
                'common_name': {
                    'type': 'text',
                    'fields': {'keyword': {'type': 'keyword'}}
                },
                'location': {'type': 'geo_point'},
                'location_description': {
                    'type': 'text',
                    'analyzer': 'location_analyzer'
                },
                'observation_date': {'type': 'date'},
                'observation_time': {'type': 'date'},
                'abundance': {'type': 'keyword'},
                'behavior': {'type': 'text'},
                'habitat_description': {'type': 'text'},
                'environmental_conditions': {
                    'properties': {
                        'water_temperature': {'type': 'float'},
                        'depth': {'type': 'float'},
                        'visibility': {'type': 'float'},
                        'current_strength': {'type': 'keyword'},
                        'weather': {'type': 'keyword'}
                    }
                },
                'confidence_level': {'type': 'keyword'},
                'verification_status': {'type': 'keyword'},
                'images': {'type': 'keyword'},
                'notes': {'type': 'text'},
                'created_at': {'type': 'date'}
            }
        }
    
    # Indexing operations
    
    async def index_document(self, data_type: str, doc_id: str, document: Dict[str, Any]) -> bool:
        """Index a single document"""
        if not self.is_connected or data_type not in self.index_configs:
            return False
        
        try:
            index_name = self.index_configs[data_type]['index']
            
            # Add timestamp if not present
            if 'created_at' not in document:
                document['created_at'] = datetime.now().isoformat()
            
            result = await self.client.index(
                index=index_name,
                id=doc_id,
                body=document
            )
            
            logger.debug(f"Indexed document {doc_id} in {index_name}")
            return result['result'] in ['created', 'updated']
            
        except Exception as e:
            logger.error(f"Error indexing document {doc_id}: {str(e)}")
            return False
    
    async def bulk_index_documents(self, data_type: str, documents: List[Dict[str, Any]]) -> int:
        """Bulk index multiple documents"""
        if not self.is_connected or data_type not in self.index_configs:
            return 0
        
        try:
            index_name = self.index_configs[data_type]['index']
            
            # Prepare bulk operations
            operations = []
            for doc in documents:
                if 'created_at' not in doc:
                    doc['created_at'] = datetime.now().isoformat()
                
                operations.append({
                    '_index': index_name,
                    '_source': doc
                })
            
            if not operations:
                return 0
            
            result = await self.client.bulk(body=operations)
            
            # Count successful operations
            successful = 0
            for item in result['items']:
                if item.get('index', {}).get('status') in [200, 201]:
                    successful += 1
            
            logger.info(f"Bulk indexed {successful}/{len(documents)} documents in {index_name}")
            return successful
            
        except Exception as e:
            logger.error(f"Error bulk indexing documents: {str(e)}")
            return 0
    
    # Search operations
    
    async def search(self, query: str, data_types: Optional[List[str]] = None, 
                    filters: Optional[Dict[str, Any]] = None, 
                    size: int = 20, from_: int = 0,
                    sort: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Perform general search across data types"""
        if not self.is_connected:
            return {'hits': {'hits': [], 'total': {'value': 0}}}
        
        try:
            # Determine indices to search
            if data_types:
                indices = [self.index_configs[dt]['index'] for dt in data_types if dt in self.index_configs]
            else:
                indices = [config['index'] for config in self.index_configs.values()]
            
            if not indices:
                return {'hits': {'hits': [], 'total': {'value': 0}}}
            
            # Build search body
            search_body = {
                'size': size,
                'from': from_,
                'query': self._build_search_query(query, filters),
                'highlight': self.default_search_config['highlight']
            }
            
            if sort:
                search_body['sort'] = sort
            
            # Execute search
            result = await self.client.search(
                index=','.join(indices),
                body=search_body
            )
            
            logger.debug(f"Search executed: {query} - {result['hits']['total']['value']} results")
            return result
            
        except Exception as e:
            logger.error(f"Error executing search: {str(e)}")
            return {'hits': {'hits': [], 'total': {'value': 0}}}
    
    async def suggest(self, query: str, data_type: str = 'species') -> List[str]:
        """Get search suggestions for autocomplete"""
        if not self.is_connected or data_type not in self.index_configs:
            return []
        
        try:
            index_name = self.index_configs[data_type]['index']
            
            suggest_body = {
                'suggest': {
                    'species_suggest': {
                        'prefix': query,
                        'completion': {
                            'field': 'common_name.suggest' if data_type == 'species' else 'scientific_name.suggest',
                            'size': 10,
                            'skip_duplicates': True
                        }
                    },
                    'scientific_suggest': {
                        'prefix': query,
                        'completion': {
                            'field': 'scientific_name.suggest',
                            'size': 10,
                            'skip_duplicates': True
                        }
                    }
                }
            }
            
            result = await self.client.search(
                index=index_name,
                body=suggest_body
            )
            
            suggestions = []
            for suggest_result in result.get('suggest', {}).values():
                for suggestion in suggest_result:
                    for option in suggestion.get('options', []):
                        suggestions.append(option['text'])
            
            return list(set(suggestions))[:10]  # Remove duplicates and limit
            
        except Exception as e:
            logger.error(f"Error getting suggestions: {str(e)}")
            return []
    
    def _build_search_query(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build Elasticsearch query from search parameters"""
        must_clauses = []
        
        # Main query
        if query.strip():
            must_clauses.append({
                'multi_match': {
                    'query': query,
                    'fields': [
                        'common_name^3',
                        'scientific_name^3',
                        'description^2',
                        'habitat^2',
                        'family',
                        'order',
                        'class',
                        'notes',
                        '*'
                    ],
                    'type': 'best_fields',
                    'fuzziness': 'AUTO'
                }
            })
        else:
            must_clauses.append({'match_all': {}})
        
        # Add filters
        if filters:
            for field, value in filters.items():
                if isinstance(value, list):
                    must_clauses.append({
                        'terms': {f'{field}.keyword': value}
                    })
                elif isinstance(value, dict):
                    if 'range' in value:
                        must_clauses.append({
                            'range': {field: value['range']}
                        })
                    elif 'geo_distance' in value:
                        must_clauses.append({
                            'geo_distance': {
                                'distance': value['geo_distance']['distance'],
                                field: value['geo_distance']['location']
                            }
                        })
                else:
                    must_clauses.append({
                        'term': {f'{field}.keyword': value}
                    })
        
        return {
            'bool': {
                'must': must_clauses
            }
        }
    
    # Analytics and aggregations
    
    async def get_search_analytics(self, data_type: Optional[str] = None, 
                                  time_range: str = '7d') -> Dict[str, Any]:
        """Get search analytics and aggregations"""
        if not self.is_connected:
            return {}
        
        try:
            # Determine indices
            if data_type and data_type in self.index_configs:
                indices = [self.index_configs[data_type]['index']]
            else:
                indices = [config['index'] for config in self.index_configs.values()]
            
            # Build analytics query
            analytics_body = {
                'size': 0,
                'query': {
                    'range': {
                        'created_at': {
                            'gte': f'now-{time_range}'
                        }
                    }
                },
                'aggs': {
                    'total_documents': {
                        'value_count': {
                            'field': '_id'
                        }
                    },
                    'by_type': {
                        'terms': {
                            'field': '_index',
                            'size': 10
                        }
                    },
                    'recent_activity': {
                        'date_histogram': {
                            'field': 'created_at',
                            'calendar_interval': '1d',
                            'min_doc_count': 0
                        }
                    }
                }
            }
            
            # Add specific aggregations based on data type
            if data_type == 'species':
                analytics_body['aggs']['top_families'] = {
                    'terms': {
                        'field': 'family.keyword',
                        'size': 10
                    }
                }
            elif data_type == 'fisheries':
                analytics_body['aggs']['top_fishing_areas'] = {
                    'terms': {
                        'field': 'fishing_area.keyword',
                        'size': 10
                    }
                }
            
            result = await self.client.search(
                index=','.join(indices),
                body=analytics_body
            )
            
            return {
                'total_documents': result['aggregations']['total_documents']['value'],
                'by_type': result['aggregations']['by_type']['buckets'],
                'recent_activity': result['aggregations']['recent_activity']['buckets'],
                'specific_aggs': {k: v for k, v in result['aggregations'].items() 
                                if k not in ['total_documents', 'by_type', 'recent_activity']}
            }
            
        except Exception as e:
            logger.error(f"Error getting search analytics: {str(e)}")
            return {}
    
    # Health and status
    
    async def get_cluster_health(self) -> Dict[str, Any]:
        """Get Elasticsearch cluster health"""
        if not self.is_connected:
            return {'status': 'disconnected'}
        
        try:
            health = await self.client.cluster.health()
            indices_stats = await self.client.cat.indices(format='json')
            
            return {
                'cluster_status': health['status'],
                'number_of_nodes': health['number_of_nodes'],
                'number_of_data_nodes': health['number_of_data_nodes'],
                'active_primary_shards': health['active_primary_shards'],
                'active_shards': health['active_shards'],
                'relocating_shards': health['relocating_shards'],
                'initializing_shards': health['initializing_shards'],
                'unassigned_shards': health['unassigned_shards'],
                'indices': [
                    {
                        'index': idx['index'],
                        'docs_count': int(idx.get('docs.count', 0)),
                        'store_size': idx.get('store.size', '0b'),
                        'health': idx.get('health', 'unknown')
                    }
                    for idx in indices_stats
                    if idx['index'].startswith('ocean_bio_')
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting cluster health: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _get_edna_mapping(self) -> Dict[str, Any]:
        """Get mapping for eDNA index"""
        return {
            'properties': {
                'id': {'type': 'integer'},
                'sample_id': {'type': 'keyword'},
                'sample_name': {'type': 'text', 'fields': {'keyword': {'type': 'keyword'}}},
                'collection_date': {'type': 'date'},
                'collected_by': {'type': 'text'},
                'location': {'type': 'geo_point'},
                'latitude': {'type': 'float'},
                'longitude': {'type': 'float'},
                'depth': {'type': 'float'},
                'water_temperature': {'type': 'float'},
                'salinity': {'type': 'float'},
                'ph': {'type': 'float'},
                'dissolved_oxygen': {'type': 'float'},
                'species_detected': {'type': 'text'},
                'detection_method': {'type': 'keyword'},
                'confidence_score': {'type': 'float'},
                'target_gene': {'type': 'keyword'},
                'sequence_data': {'type': 'text'},
                'processing_status': {'type': 'keyword'},
                'created_at': {'type': 'date'}
            }
        }
    
    def _get_oceanographic_mapping(self) -> Dict[str, Any]:
        """Get mapping for oceanographic index"""
        return {
            'properties': {
                'id': {'type': 'integer'},
                'station_id': {'type': 'keyword'},
                'station_name': {'type': 'text', 'fields': {'keyword': {'type': 'keyword'}}},
                'location': {'type': 'geo_point'},
                'latitude': {'type': 'float'},
                'longitude': {'type': 'float'},
                'depth': {'type': 'float'},
                'measurement_time': {'type': 'date'},
                'parameter_name': {'type': 'keyword'},
                'parameter_value': {'type': 'float'},
                'parameter_unit': {'type': 'keyword'},
                'quality_flag': {'type': 'keyword'},
                'instrument': {'type': 'text'},
                'data_source': {'type': 'keyword'},
                'weather_conditions': {'type': 'text'},
                'sea_state': {'type': 'keyword'},
                'processing_level': {'type': 'keyword'},
                'created_at': {'type': 'date'}
            }
        }
    
    def _get_otolith_mapping(self) -> Dict[str, Any]:
        """Get mapping for otolith index"""
        return {
            'properties': {
                'id': {'type': 'integer'},
                'specimen_id': {'type': 'keyword'},
                'species_id': {'type': 'integer'},
                'scientific_name': {
                    'type': 'text',
                    'analyzer': 'scientific_name_analyzer',
                    'fields': {'keyword': {'type': 'keyword'}}
                },
                'common_name': {'type': 'text', 'fields': {'keyword': {'type': 'keyword'}}},
                'collection_location': {'type': 'geo_point'},
                'collection_date': {'type': 'date'},
                'fish_length': {'type': 'float'},
                'fish_weight': {'type': 'float'},
                'fish_age': {'type': 'integer'},
                'otolith_length': {'type': 'float'},
                'otolith_width': {'type': 'float'},
                'otolith_weight': {'type': 'float'},
                'growth_rings': {'type': 'integer'},
                'morphometric_data': {'type': 'object'},
                'image_analysis_results': {'type': 'object'},
                'preparation_method': {'type': 'keyword'},
                'created_at': {'type': 'date'}
            }
        }
    
    def _get_taxonomy_mapping(self) -> Dict[str, Any]:
        """Get mapping for taxonomy index - updated for broader taxonomy"""
        return {
            'properties': {
                'id': {'type': 'integer'},
                'taxon_id': {'type': 'keyword'},
                'scientific_name': {
                    'type': 'text',
                    'analyzer': 'scientific_name_analyzer',
                    'fields': {
                        'keyword': {'type': 'keyword'},
                        'suggest': {
                            'type': 'completion',
                            'analyzer': 'scientific_name_analyzer'
                        }
                    }
                },
                'common_names': {
                    'type': 'text',
                    'analyzer': 'standard',
                    'fields': {
                        'keyword': {'type': 'keyword'},
                        'suggest': {
                            'type': 'completion',
                            'analyzer': 'standard'
                        }
                    }
                },
                'rank': {'type': 'keyword'},
                'kingdom': {'type': 'keyword'},
                'phylum': {'type': 'keyword'},
                'class': {'type': 'keyword'},
                'order': {'type': 'keyword'},
                'family': {'type': 'keyword'},
                'genus': {'type': 'keyword'},
                'species': {'type': 'keyword'},
                'parent_taxon_id': {'type': 'keyword'},
                'children_count': {'type': 'integer'},
                'taxonomic_authority': {'type': 'text'},
                'year_described': {'type': 'integer'},
                'synonyms': {
                    'type': 'text',
                    'analyzer': 'scientific_name_analyzer'
                },
                'description': {'type': 'text'},
                'habitat': {'type': 'text'},
                'distribution': {'type': 'text'},
                'conservation_status': {'type': 'keyword'},
                'created_at': {'type': 'date'}
            }
        }
    
    def _get_fisheries_mapping(self) -> Dict[str, Any]:
        """Get mapping for fisheries index - updated for comprehensive fisheries data"""
        return {
            'properties': {
                'id': {'type': 'integer'},
                'vessel_id': {'type': 'integer'},
                'vessel_name': {'type': 'text', 'fields': {'keyword': {'type': 'keyword'}}},
                'registration_number': {'type': 'keyword'},
                'vessel_type': {'type': 'keyword'},
                'trip_id': {'type': 'integer'},
                'species_id': {'type': 'integer'},
                'scientific_name': {
                    'type': 'text',
                    'analyzer': 'scientific_name_analyzer',
                    'fields': {'keyword': {'type': 'keyword'}}
                },
                'common_name': {'type': 'text', 'fields': {'keyword': {'type': 'keyword'}}},
                'catch_date': {'type': 'date'},
                'catch_location': {'type': 'geo_point'},
                'fishing_area': {'type': 'keyword'},
                'fishing_method': {'type': 'keyword'},
                'catch_weight': {'type': 'float'},
                'catch_quantity': {'type': 'integer'},
                'average_size': {'type': 'float'},
                'market_grade': {'type': 'keyword'},
                'price_per_kg': {'type': 'float'},
                'total_value': {'type': 'float'},
                'landing_port': {'type': 'keyword'},
                'depth': {'type': 'float'},
                'water_temperature': {'type': 'float'},
                'catch_status': {'type': 'keyword'},
                'quota_impact': {'type': 'float'},
                'sustainability_rating': {'type': 'keyword'},
                'created_at': {'type': 'date'}
            }
        }

# Global Elasticsearch service instance
elasticsearch_service = ElasticsearchService()
