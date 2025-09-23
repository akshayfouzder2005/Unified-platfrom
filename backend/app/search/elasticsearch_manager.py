"""
Elasticsearch Manager for Ocean-Bio Platform
Handles Elasticsearch connections, index management, and search operations
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Elasticsearch imports with fallback
try:
    from elasticsearch import Elasticsearch
    from elasticsearch.exceptions import ConnectionError, NotFoundError, RequestError
    HAS_ELASTICSEARCH = True
except ImportError:
    HAS_ELASTICSEARCH = False

logger = logging.getLogger(__name__)

class ElasticsearchManager:
    """
    Elasticsearch connection and index management
    Provides search functionality with fallback when Elasticsearch is not available
    """
    
    def __init__(self, hosts: List[str] = None):
        self.hosts = hosts or ['localhost:9200']
        self.es_client = None
        self.is_connected = False
        
        # Index configurations
        self.indices = {
            'species': {
                'name': 'ocean_bio_species',
                'mappings': self._get_species_mapping(),
                'settings': self._get_default_settings()
            },
            'research_data': {
                'name': 'ocean_bio_research_data',
                'mappings': self._get_research_data_mapping(),
                'settings': self._get_default_settings()
            },
            'fisheries': {
                'name': 'ocean_bio_fisheries',
                'mappings': self._get_fisheries_mapping(),
                'settings': self._get_default_settings()
            },
            'otoliths': {
                'name': 'ocean_bio_otoliths',
                'mappings': self._get_otolith_mapping(),
                'settings': self._get_default_settings()
            }
        }
        
        # Initialize connection
        self._initialize_elasticsearch()
        
        # Fallback search data when Elasticsearch is not available
        self.fallback_data = self._create_fallback_search_data()
    
    def _initialize_elasticsearch(self):
        """Initialize Elasticsearch connection"""
        if not HAS_ELASTICSEARCH:
            logger.warning("Elasticsearch not available - using fallback search")
            return
        
        try:
            self.es_client = Elasticsearch(
                self.hosts,
                timeout=30,
                max_retries=2,
                retry_on_timeout=True
            )
            
            # Test connection
            if self.es_client.ping():
                self.is_connected = True
                logger.info("Connected to Elasticsearch successfully")
                self._setup_indices()
            else:
                logger.warning("Could not connect to Elasticsearch - using fallback search")
                
        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {str(e)}")
            logger.warning("Using fallback search functionality")
    
    def _setup_indices(self):
        """Set up Elasticsearch indices"""
        try:
            for index_key, index_config in self.indices.items():
                index_name = index_config['name']
                
                if not self.es_client.indices.exists(index=index_name):
                    self.es_client.indices.create(
                        index=index_name,
                        body={
                            'mappings': index_config['mappings'],
                            'settings': index_config['settings']
                        }
                    )
                    logger.info(f"Created index: {index_name}")
                    
                    # Add sample data
                    self._populate_sample_data(index_key, index_name)
                
            logger.info("All Elasticsearch indices are ready")
            
        except Exception as e:
            logger.error(f"Error setting up indices: {str(e)}")
    
    def _get_species_mapping(self) -> Dict[str, Any]:
        """Get mapping configuration for species index"""
        return {
            'properties': {
                'id': {'type': 'integer'},
                'common_name': {
                    'type': 'text',
                    'analyzer': 'standard',
                    'fields': {
                        'keyword': {'type': 'keyword'},
                        'suggest': {'type': 'completion'}
                    }
                },
                'scientific_name': {
                    'type': 'text',
                    'analyzer': 'standard',
                    'fields': {
                        'keyword': {'type': 'keyword'},
                        'suggest': {'type': 'completion'}
                    }
                },
                'taxonomy': {
                    'properties': {
                        'kingdom': {'type': 'keyword'},
                        'phylum': {'type': 'keyword'},
                        'class': {'type': 'keyword'},
                        'order': {'type': 'keyword'},
                        'family': {'type': 'keyword'},
                        'genus': {'type': 'keyword'}
                    }
                },
                'habitat': {'type': 'text'},
                'description': {'type': 'text'},
                'conservation_status': {'type': 'keyword'},
                'geographic_distribution': {'type': 'text'},
                'created_at': {'type': 'date'},
                'updated_at': {'type': 'date'}
            }
        }
    
    def _get_research_data_mapping(self) -> Dict[str, Any]:
        """Get mapping for research data index"""
        return {
            'properties': {
                'id': {'type': 'integer'},
                'title': {
                    'type': 'text',
                    'analyzer': 'standard',
                    'fields': {'suggest': {'type': 'completion'}}
                },
                'description': {'type': 'text'},
                'research_type': {'type': 'keyword'},
                'location': {
                    'properties': {
                        'name': {'type': 'text'},
                        'coordinates': {'type': 'geo_point'},
                        'depth': {'type': 'float'}
                    }
                },
                'sample_data': {'type': 'text'},
                'methodology': {'type': 'text'},
                'findings': {'type': 'text'},
                'researcher': {'type': 'keyword'},
                'institution': {'type': 'keyword'},
                'publication_date': {'type': 'date'},
                'tags': {'type': 'keyword'}
            }
        }
    
    def _get_fisheries_mapping(self) -> Dict[str, Any]:
        """Get mapping for fisheries index"""
        return {
            'properties': {
                'vessel_id': {'type': 'integer'},
                'vessel_name': {'type': 'text'},
                'catch_data': {
                    'properties': {
                        'species': {'type': 'keyword'},
                        'weight': {'type': 'float'},
                        'location': {'type': 'geo_point'},
                        'date': {'type': 'date'}
                    }
                },
                'fishing_method': {'type': 'keyword'},
                'fishing_area': {'type': 'keyword'},
                'quota_info': {'type': 'text'}
            }
        }
    
    def _get_otolith_mapping(self) -> Dict[str, Any]:
        """Get mapping for otolith index"""
        return {
            'properties': {
                'specimen_id': {'type': 'keyword'},
                'species_name': {
                    'type': 'text',
                    'fields': {'suggest': {'type': 'completion'}}
                },
                'morphometric_data': {
                    'properties': {
                        'length': {'type': 'float'},
                        'width': {'type': 'float'},
                        'area': {'type': 'float'},
                        'perimeter': {'type': 'float'}
                    }
                },
                'collection_info': {
                    'properties': {
                        'location': {'type': 'geo_point'},
                        'date': {'type': 'date'},
                        'collector': {'type': 'keyword'}
                    }
                }
            }
        }
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default index settings"""
        return {
            'number_of_shards': 1,
            'number_of_replicas': 0,
            'analysis': {
                'analyzer': {
                    'marine_text_analyzer': {
                        'type': 'standard',
                        'stopwords': '_english_'
                    }
                }
            }
        }
    
    def _populate_sample_data(self, index_key: str, index_name: str):
        """Populate index with sample data"""
        try:
            sample_data = self._get_sample_data(index_key)
            
            for doc in sample_data:
                self.es_client.index(
                    index=index_name,
                    body=doc
                )
            
            # Refresh index to make documents searchable
            self.es_client.indices.refresh(index=index_name)
            
            logger.info(f"Populated {index_name} with {len(sample_data)} sample documents")
            
        except Exception as e:
            logger.error(f"Error populating sample data for {index_name}: {str(e)}")
    
    def _get_sample_data(self, index_key: str) -> List[Dict[str, Any]]:
        """Get sample data for each index type"""
        if index_key == 'species':
            return [
                {
                    'id': 1,
                    'common_name': 'Atlantic Cod',
                    'scientific_name': 'Gadus morhua',
                    'taxonomy': {
                        'kingdom': 'Animalia',
                        'phylum': 'Chordata',
                        'class': 'Actinopterygii',
                        'order': 'Gadiformes',
                        'family': 'Gadidae',
                        'genus': 'Gadus'
                    },
                    'habitat': 'Cold water marine environments',
                    'description': 'Important commercial fish species in North Atlantic',
                    'conservation_status': 'Least Concern',
                    'geographic_distribution': 'North Atlantic Ocean',
                    'created_at': datetime.now().isoformat()
                },
                {
                    'id': 2,
                    'common_name': 'Yellowfin Tuna',
                    'scientific_name': 'Thunnus albacares',
                    'taxonomy': {
                        'kingdom': 'Animalia',
                        'phylum': 'Chordata',
                        'class': 'Actinopterygii',
                        'order': 'Scombriformes',
                        'family': 'Scombridae',
                        'genus': 'Thunnus'
                    },
                    'habitat': 'Pelagic waters of tropical and subtropical oceans',
                    'description': 'Large, fast-swimming tuna species',
                    'conservation_status': 'Near Threatened',
                    'geographic_distribution': 'Worldwide in warm oceans',
                    'created_at': datetime.now().isoformat()
                }
            ]
        elif index_key == 'research_data':
            return [
                {
                    'id': 1,
                    'title': 'Marine Biodiversity Assessment in North Atlantic',
                    'description': 'Comprehensive study of marine species diversity',
                    'research_type': 'biodiversity_survey',
                    'location': {
                        'name': 'North Atlantic',
                        'coordinates': {'lat': 55.0, 'lon': -30.0},
                        'depth': 200.0
                    },
                    'methodology': 'Trawl surveys and eDNA sampling',
                    'findings': 'High biodiversity with several rare species identified',
                    'researcher': 'Dr. Marine Biologist',
                    'institution': 'Ocean Research Institute',
                    'publication_date': datetime.now().isoformat(),
                    'tags': ['biodiversity', 'marine', 'atlantic']
                }
            ]
        
        return []
    
    def _create_fallback_search_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Create fallback search data when Elasticsearch is unavailable"""
        return {
            'species': [
                {
                    'id': 1,
                    'common_name': 'Atlantic Cod',
                    'scientific_name': 'Gadus morhua',
                    'description': 'Important commercial fish species',
                    'habitat': 'Cold water marine environments'
                },
                {
                    'id': 2,
                    'common_name': 'Yellowfin Tuna',
                    'scientific_name': 'Thunnus albacares',
                    'description': 'Large, fast-swimming tuna species',
                    'habitat': 'Pelagic waters of tropical oceans'
                },
                {
                    'id': 3,
                    'common_name': 'Red Snapper',
                    'scientific_name': 'Lutjanus campechanus',
                    'description': 'Popular recreational and commercial fish',
                    'habitat': 'Reef environments in warm waters'
                }
            ],
            'research_data': [
                {
                    'id': 1,
                    'title': 'Marine Biodiversity Assessment',
                    'description': 'Comprehensive study of marine species diversity',
                    'research_type': 'biodiversity_survey'
                }
            ]
        }
    
    def search(self, query: str, index_types: List[str] = None, size: int = 20) -> Dict[str, Any]:
        """
        Perform search across specified indices
        
        Args:
            query: Search query string
            index_types: List of index types to search in
            size: Maximum number of results
            
        Returns:
            Search results
        """
        if self.is_connected and self.es_client:
            return self._elasticsearch_search(query, index_types, size)
        else:
            return self._fallback_search(query, index_types, size)
    
    def _elasticsearch_search(self, query: str, index_types: List[str] = None, size: int = 20) -> Dict[str, Any]:
        """Perform Elasticsearch search"""
        try:
            if not index_types:
                index_types = list(self.indices.keys())
            
            # Build indices list
            indices = [self.indices[idx]['name'] for idx in index_types if idx in self.indices]
            
            # Build search query
            search_body = {
                'query': {
                    'multi_match': {
                        'query': query,
                        'fields': ['common_name^3', 'scientific_name^3', 'title^2', 'description', 'habitat'],
                        'type': 'best_fields',
                        'fuzziness': 'AUTO'
                    }
                },
                'highlight': {
                    'fields': {
                        'common_name': {},
                        'scientific_name': {},
                        'description': {},
                        'title': {}
                    }
                },
                'size': size
            }
            
            # Execute search
            response = self.es_client.search(
                index=','.join(indices),
                body=search_body
            )
            
            # Process results
            results = []
            for hit in response['hits']['hits']:
                result = {
                    'id': hit['_source'].get('id', hit['_id']),
                    'index_type': hit['_index'],
                    'score': hit['_score'],
                    'source': hit['_source'],
                    'highlight': hit.get('highlight', {})
                }
                results.append(result)
            
            return {
                'success': True,
                'total_hits': response['hits']['total']['value'],
                'max_score': response['hits']['max_score'],
                'results': results,
                'took': response['took'],
                'search_type': 'elasticsearch'
            }
            
        except Exception as e:
            logger.error(f"Elasticsearch search error: {str(e)}")
            return self._fallback_search(query, index_types, size)
    
    def _fallback_search(self, query: str, index_types: List[str] = None, size: int = 20) -> Dict[str, Any]:
        """Fallback search when Elasticsearch is unavailable"""
        try:
            if not index_types:
                index_types = list(self.fallback_data.keys())
            
            results = []
            query_lower = query.lower()
            
            for index_type in index_types:
                if index_type not in self.fallback_data:
                    continue
                
                for item in self.fallback_data[index_type]:
                    score = 0
                    
                    # Simple text matching with scoring
                    for field in ['common_name', 'scientific_name', 'title', 'description']:
                        if field in item:
                            field_value = str(item[field]).lower()
                            if query_lower in field_value:
                                if field in ['common_name', 'scientific_name', 'title']:
                                    score += 3  # Higher score for name/title matches
                                else:
                                    score += 1
                    
                    if score > 0:
                        results.append({
                            'id': item.get('id', 'unknown'),
                            'index_type': f'fallback_{index_type}',
                            'score': score,
                            'source': item,
                            'highlight': {}
                        })
            
            # Sort by score
            results.sort(key=lambda x: x['score'], reverse=True)
            results = results[:size]
            
            return {
                'success': True,
                'total_hits': len(results),
                'max_score': max([r['score'] for r in results]) if results else 0,
                'results': results,
                'took': 1,  # Mock timing
                'search_type': 'fallback'
            }
            
        except Exception as e:
            logger.error(f"Fallback search error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'results': [],
                'search_type': 'fallback_error'
            }
    
    def suggest(self, query: str, index_type: str = 'species') -> List[str]:
        """Get search suggestions"""
        if self.is_connected and self.es_client:
            try:
                index_name = self.indices[index_type]['name']
                
                suggest_body = {
                    'suggest': {
                        'text': query,
                        f'{index_type}_suggest': {
                            'completion': {
                                'field': 'common_name.suggest',
                                'size': 10
                            }
                        }
                    }
                }
                
                response = self.es_client.search(
                    index=index_name,
                    body=suggest_body
                )
                
                suggestions = []
                suggest_results = response['suggest'][f'{index_type}_suggest'][0]['options']
                for option in suggest_results:
                    suggestions.append(option['text'])
                
                return suggestions
                
            except Exception as e:
                logger.error(f"Suggestion error: {str(e)}")
        
        # Fallback suggestions
        return [item['common_name'] for item in self.fallback_data.get(index_type, [])
                if query.lower() in item.get('common_name', '').lower()][:10]
    
    def get_status(self) -> Dict[str, Any]:
        """Get search system status"""
        status = {
            'elasticsearch_available': HAS_ELASTICSEARCH,
            'connected': self.is_connected,
            'hosts': self.hosts,
            'search_type': 'elasticsearch' if self.is_connected else 'fallback'
        }
        
        if self.is_connected and self.es_client:
            try:
                cluster_health = self.es_client.cluster.health()
                status.update({
                    'cluster_status': cluster_health['status'],
                    'number_of_nodes': cluster_health['number_of_nodes'],
                    'active_shards': cluster_health['active_shards']
                })
            except Exception as e:
                logger.error(f"Error getting cluster status: {str(e)}")
        
        return status

# Global search manager instance
search_manager = ElasticsearchManager()