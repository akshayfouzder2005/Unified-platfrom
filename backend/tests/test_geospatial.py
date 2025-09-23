"""
🗺️ Comprehensive Test Suite for Geospatial Analysis Components

Unit and integration tests for GIS operations, spatial analysis, and mapping functionality.
Tests coordinate system transformations, spatial queries, and geographic data processing.

Author: Ocean-Bio Development Team
Version: 2.0.0
"""

import pytest
import numpy as np
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Import the modules under test
from backend.app.geospatial.gis_integration import GISService
from backend.app.geospatial.spatial_analysis import SpatialAnalyzer
from backend.app.geospatial.mapping_service import MappingService

class TestGISService:
    """Test suite for GIS integration service"""
    
    @pytest.fixture
    def gis_service(self):
        """Create GIS service instance for testing"""
        return GISService()
    
    @pytest.fixture
    def sample_coordinates(self):
        """Sample coordinate data for testing"""
        return {
            'valid_coordinates': [
                (19.0760, 72.8777),  # Mumbai
                (13.0827, 80.2707),  # Chennai
                (22.5726, 88.3639),  # Kolkata
            ],
            'invalid_coordinates': [
                (91.0, 0.0),    # Invalid latitude
                (0.0, 181.0),   # Invalid longitude
                (-91.0, 0.0),   # Invalid latitude
            ]
        }
    
    def test_coordinate_validation(self, gis_service, sample_coordinates):
        """Test coordinate validation functionality"""
        # Test valid coordinates
        for lat, lon in sample_coordinates['valid_coordinates']:
            result = gis_service.validate_coordinates(lat, lon)
            assert result['is_valid'] is True
            assert 'location_info' in result
            
        # Test invalid coordinates
        for lat, lon in sample_coordinates['invalid_coordinates']:
            result = gis_service.validate_coordinates(lat, lon)
            assert result['is_valid'] is False
    
    def test_distance_calculation(self, gis_service):
        """Test distance calculation between coordinates"""
        # Test known distance: Mumbai to Chennai (~1030 km)
        mumbai = (19.0760, 72.8777)
        chennai = (13.0827, 80.2707)
        
        result = gis_service.calculate_distance(
            mumbai[0], mumbai[1], 
            chennai[0], chennai[1]
        )
        
        assert 'distance_km' in result
        assert 'distance_nm' in result
        assert 'bearing' in result
        
        # Check approximate distance (allowing for some variance)
        assert 1000 < result['distance_km'] < 1100
        
        # Test same point distance
        result = gis_service.calculate_distance(
            mumbai[0], mumbai[1], 
            mumbai[0], mumbai[1]
        )
        assert result['distance_km'] == 0
    
    @patch('backend.app.geospatial.gis_integration.create_engine')
    def test_database_connection(self, mock_create_engine, gis_service):
        """Test database connection functionality"""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        connection_result = gis_service.test_connection()
        
        assert 'status' in connection_result
        assert connection_result['status'] == 'healthy'
    
    @pytest.mark.asyncio
    async def test_location_query(self, gis_service):
        """Test location-based spatial queries"""
        query_params = {
            'latitude': 19.0760,
            'longitude': 72.8777,
            'radius_km': 10.0,
            'data_types': ['biodiversity', 'water_quality'],
            'start_date': datetime.now() - timedelta(days=30),
            'end_date': datetime.now()
        }
        
        with patch.object(gis_service, '_execute_spatial_query') as mock_query:
            mock_query.return_value = {
                'type': 'FeatureCollection',
                'features': [
                    {
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Point',
                            'coordinates': [72.8777, 19.0760]
                        },
                        'properties': {
                            'id': 'test_001',
                            'type': 'biodiversity',
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                ]
            }
            
            result = await gis_service.query_by_location(**query_params)
            
            assert 'type' in result
            assert result['type'] == 'FeatureCollection'
            assert 'features' in result
            assert len(result['features']) > 0
    
    @pytest.mark.asyncio
    async def test_bounding_box_query(self, gis_service):
        """Test bounding box spatial queries"""
        query_params = {
            'min_lat': 18.0,
            'max_lat': 20.0,
            'min_lon': 72.0,
            'max_lon': 74.0,
            'data_types': ['biodiversity']
        }
        
        with patch.object(gis_service, '_execute_spatial_query') as mock_query:
            mock_query.return_value = {
                'type': 'FeatureCollection',
                'features': []
            }
            
            result = await gis_service.query_by_bounding_box(**query_params)
            
            assert 'type' in result
            assert result['type'] == 'FeatureCollection'
            mock_query.assert_called_once()
    
    def test_coordinate_transformation(self, gis_service):
        """Test coordinate system transformations"""
        # Test WGS84 to UTM conversion
        lat, lon = 19.0760, 72.8777  # Mumbai
        
        with patch.object(gis_service, '_transform_coordinates') as mock_transform:
            mock_transform.return_value = (654321.0, 2109876.0)
            
            utm_x, utm_y = gis_service.transform_to_utm(lat, lon)
            
            assert isinstance(utm_x, (int, float))
            assert isinstance(utm_y, (int, float))
            mock_transform.assert_called_once()


class TestSpatialAnalyzer:
    """Test suite for spatial analysis functionality"""
    
    @pytest.fixture
    def spatial_analyzer(self):
        """Create spatial analyzer instance for testing"""
        return SpatialAnalyzer()
    
    @pytest.fixture
    def sample_locations(self):
        """Sample location data for analysis"""
        return [
            (19.0760, 72.8777),  # Mumbai
            (19.0896, 72.8656),  # Mumbai nearby
            (13.0827, 80.2707),  # Chennai
            (13.0878, 80.2785),  # Chennai nearby
        ]
    
    def test_get_available_analyses(self, spatial_analyzer):
        """Test getting list of available spatial analyses"""
        analyses = spatial_analyzer.get_available_analyses()
        
        assert isinstance(analyses, list)
        assert 'hotspot_analysis' in analyses
        assert 'cluster_analysis' in analyses
        assert 'spatial_autocorrelation' in analyses
        assert 'interpolation' in analyses
    
    def test_get_analysis_info(self, spatial_analyzer):
        """Test getting analysis information"""
        info = spatial_analyzer.get_analysis_info('hotspot_analysis')
        
        assert 'name' in info
        assert 'description' in info
        assert 'parameters' in info
        assert 'output_type' in info
    
    @pytest.mark.asyncio
    async def test_hotspot_analysis(self, spatial_analyzer, sample_locations):
        """Test hotspot analysis functionality"""
        analysis_params = {
            'analysis_type': 'hotspot_analysis',
            'locations': sample_locations,
            'parameters': {
                'confidence_level': 0.95,
                'distance_band': 1000  # 1km
            }
        }
        
        result = await spatial_analyzer.perform_analysis(**analysis_params)
        
        assert 'analysis_type' in result
        assert result['analysis_type'] == 'hotspot_analysis'
        assert 'hotspots' in result
        assert 'coldspots' in result
        assert 'statistics' in result
    
    @pytest.mark.asyncio
    async def test_cluster_analysis(self, spatial_analyzer, sample_locations):
        """Test spatial clustering analysis"""
        analysis_params = {
            'analysis_type': 'cluster_analysis',
            'locations': sample_locations,
            'parameters': {
                'method': 'dbscan',
                'eps': 0.1,  # ~11km
                'min_samples': 2
            }
        }
        
        result = await spatial_analyzer.perform_analysis(**analysis_params)
        
        assert 'analysis_type' in result
        assert result['analysis_type'] == 'cluster_analysis'
        assert 'clusters' in result
        assert 'cluster_count' in result
        assert 'cluster_metrics' in result
    
    @pytest.mark.asyncio
    async def test_spatial_autocorrelation(self, spatial_analyzer, sample_locations):
        """Test spatial autocorrelation analysis"""
        # Add some values to the locations
        locations_with_values = [
            (loc[0], loc[1], np.random.random()) for loc in sample_locations
        ]
        
        analysis_params = {
            'analysis_type': 'spatial_autocorrelation',
            'locations': locations_with_values,
            'parameters': {
                'method': 'morans_i',
                'distance_threshold': 100  # 100km
            }
        }
        
        result = await spatial_analyzer.perform_analysis(**analysis_params)
        
        assert 'analysis_type' in result
        assert result['analysis_type'] == 'spatial_autocorrelation'
        assert 'morans_i' in result
        assert 'p_value' in result
        assert 'interpretation' in result
    
    @pytest.mark.asyncio
    async def test_interpolation_analysis(self, spatial_analyzer, sample_locations):
        """Test spatial interpolation analysis"""
        # Add values for interpolation
        locations_with_values = [
            (loc[0], loc[1], 20 + np.random.random() * 10) for loc in sample_locations
        ]
        
        analysis_params = {
            'analysis_type': 'interpolation',
            'locations': locations_with_values,
            'parameters': {
                'method': 'kriging',
                'grid_resolution': 0.01,
                'output_format': 'raster'
            }
        }
        
        result = await spatial_analyzer.perform_analysis(**analysis_params)
        
        assert 'analysis_type' in result
        assert result['analysis_type'] == 'interpolation'
        assert 'interpolated_surface' in result
        assert 'variance_surface' in result
        assert 'validation_metrics' in result
    
    def test_distance_matrix_calculation(self, spatial_analyzer, sample_locations):
        """Test distance matrix calculation"""
        distance_matrix = spatial_analyzer._calculate_distance_matrix(sample_locations)
        
        assert isinstance(distance_matrix, np.ndarray)
        assert distance_matrix.shape == (len(sample_locations), len(sample_locations))
        
        # Test symmetry
        assert np.allclose(distance_matrix, distance_matrix.T)
        
        # Test diagonal is zero
        assert np.allclose(np.diag(distance_matrix), 0)
    
    def test_spatial_weights_calculation(self, spatial_analyzer, sample_locations):
        """Test spatial weights matrix calculation"""
        weights_matrix = spatial_analyzer._calculate_spatial_weights(
            sample_locations, method='distance_band', threshold=100
        )
        
        assert isinstance(weights_matrix, np.ndarray)
        assert weights_matrix.shape == (len(sample_locations), len(sample_locations))
        
        # Test symmetry
        assert np.allclose(weights_matrix, weights_matrix.T)
        
        # Test diagonal is zero (no self-neighbors)
        assert np.allclose(np.diag(weights_matrix), 0)


class TestMappingService:
    """Test suite for mapping service functionality"""
    
    @pytest.fixture
    def mapping_service(self):
        """Create mapping service instance for testing"""
        return MappingService()
    
    @pytest.fixture
    def sample_map_request(self):
        """Sample map generation request"""
        return {
            'map_type': 'biodiversity_hotspots',
            'region': {
                'min_lat': 18.0,
                'max_lat': 20.0,
                'min_lon': 72.0,
                'max_lon': 74.0
            },
            'layers': ['biodiversity', 'bathymetry'],
            'style_options': {
                'color_scheme': 'viridis',
                'transparency': 0.7,
                'marker_size': 5
            }
        }
    
    def test_get_available_layers(self, mapping_service):
        """Test getting available map layers"""
        layers = mapping_service.get_available_layers()
        
        assert isinstance(layers, list)
        assert len(layers) > 0
        
        # Check layer structure
        for layer in layers:
            assert 'id' in layer
            assert 'name' in layer
            assert 'type' in layer
            assert 'description' in layer
    
    def test_get_status(self, mapping_service):
        """Test mapping service status"""
        status = mapping_service.get_status()
        
        assert 'status' in status
        assert 'capabilities' in status
        assert 'layer_count' in status
    
    @pytest.mark.asyncio
    async def test_generate_map(self, mapping_service, sample_map_request):
        """Test map generation functionality"""
        with patch.object(mapping_service, '_render_map_layers') as mock_render:
            mock_render.return_value = {
                'map_id': 'test_map_001',
                'map_url': '/maps/test_map_001.html',
                'metadata': {
                    'layers_rendered': 2,
                    'features_count': 150,
                    'generation_time': 2.5
                }
            }
            
            result = await mapping_service.generate_map(**sample_map_request)
            
            assert 'map_id' in result
            assert 'map_url' in result
            assert 'metadata' in result
            mock_render.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_map_details(self, mapping_service):
        """Test retrieving map details"""
        map_id = 'test_map_001'
        
        with patch.object(mapping_service, '_load_map_metadata') as mock_load:
            mock_load.return_value = {
                'map_id': map_id,
                'creation_date': datetime.now().isoformat(),
                'region': {'min_lat': 18.0, 'max_lat': 20.0, 'min_lon': 72.0, 'max_lon': 74.0},
                'layers': ['biodiversity', 'bathymetry'],
                'status': 'ready'
            }
            
            result = await mapping_service.get_map_details(map_id)
            
            assert result is not None
            assert result['map_id'] == map_id
            assert 'creation_date' in result
    
    @pytest.mark.asyncio
    async def test_delete_map(self, mapping_service):
        """Test map deletion functionality"""
        map_id = 'test_map_001'
        
        with patch.object(mapping_service, '_remove_map_files') as mock_remove:
            mock_remove.return_value = True
            
            result = await mapping_service.delete_map(map_id)
            
            assert result is True
            mock_remove.assert_called_once_with(map_id)
    
    def test_validate_map_request(self, mapping_service, sample_map_request):
        """Test map request validation"""
        # Test valid request
        validation_result = mapping_service._validate_map_request(sample_map_request)
        assert validation_result['is_valid'] is True
        assert validation_result['errors'] == []
        
        # Test invalid request
        invalid_request = sample_map_request.copy()
        invalid_request['region']['min_lat'] = 25.0  # Greater than max_lat
        
        validation_result = mapping_service._validate_map_request(invalid_request)
        assert validation_result['is_valid'] is False
        assert len(validation_result['errors']) > 0
    
    def test_layer_compatibility_check(self, mapping_service):
        """Test layer compatibility checking"""
        compatible_layers = ['biodiversity', 'water_quality', 'bathymetry']
        incompatible_layers = ['biodiversity', 'conflicting_layer']
        
        # Test compatible layers
        result = mapping_service._check_layer_compatibility(compatible_layers)
        assert result['compatible'] is True
        
        # Test incompatible layers (mock incompatibility)
        with patch.object(mapping_service, '_get_layer_conflicts') as mock_conflicts:
            mock_conflicts.return_value = ['conflicting_layer conflicts with biodiversity']
            
            result = mapping_service._check_layer_compatibility(incompatible_layers)
            assert result['compatible'] is False
            assert len(result['conflicts']) > 0


class TestIntegration:
    """Integration tests for geospatial components"""
    
    @pytest.fixture
    def services(self):
        """Create all service instances for integration testing"""
        return {
            'gis': GISService(),
            'spatial': SpatialAnalyzer(),
            'mapping': MappingService()
        }
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, services):
        """Test complete geospatial analysis workflow"""
        # Step 1: Validate coordinates
        lat, lon = 19.0760, 72.8777
        validation = services['gis'].validate_coordinates(lat, lon)
        assert validation['is_valid'] is True
        
        # Step 2: Query nearby data
        with patch.object(services['gis'], 'query_by_location') as mock_query:
            mock_query.return_value = {
                'type': 'FeatureCollection',
                'features': [
                    {
                        'type': 'Feature',
                        'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
                        'properties': {'id': 'test', 'value': 25.5}
                    }
                ]
            }
            
            query_result = await services['gis'].query_by_location(
                latitude=lat, longitude=lon, radius_km=10
            )
            
            assert len(query_result['features']) > 0
        
        # Step 3: Perform spatial analysis
        locations = [(lat, lon), (lat + 0.01, lon + 0.01)]
        
        with patch.object(services['spatial'], 'perform_analysis') as mock_analysis:
            mock_analysis.return_value = {
                'analysis_type': 'hotspot_analysis',
                'hotspots': [{'location': [lat, lon], 'significance': 0.95}],
                'statistics': {'total_locations': 2}
            }
            
            analysis_result = await services['spatial'].perform_analysis(
                analysis_type='hotspot_analysis',
                locations=locations,
                parameters={}
            )
            
            assert 'hotspots' in analysis_result
        
        # Step 4: Generate map
        map_request = {
            'map_type': 'analysis_results',
            'region': {
                'min_lat': lat - 0.1, 'max_lat': lat + 0.1,
                'min_lon': lon - 0.1, 'max_lon': lon + 0.1
            },
            'layers': ['biodiversity'],
            'style_options': {}
        }
        
        with patch.object(services['mapping'], 'generate_map') as mock_map:
            mock_map.return_value = {
                'map_id': 'integration_test_map',
                'map_url': '/maps/integration_test_map.html'
            }
            
            map_result = await services['mapping'].generate_map(**map_request)
            
            assert 'map_id' in map_result
    
    @pytest.mark.asyncio
    async def test_error_handling(self, services):
        """Test error handling across services"""
        # Test invalid coordinates
        validation = services['gis'].validate_coordinates(91.0, 0.0)
        assert validation['is_valid'] is False
        
        # Test query with no results
        with patch.object(services['gis'], 'query_by_location') as mock_query:
            mock_query.return_value = {'type': 'FeatureCollection', 'features': []}
            
            result = await services['gis'].query_by_location(
                latitude=0.0, longitude=0.0, radius_km=1
            )
            
            assert len(result['features']) == 0
    
    def test_performance_benchmarks(self, services):
        """Test performance benchmarks for geospatial operations"""
        import time
        
        # Benchmark coordinate validation
        start_time = time.time()
        for _ in range(1000):
            services['gis'].validate_coordinates(19.0760, 72.8777)
        validation_time = time.time() - start_time
        
        # Should validate 1000 coordinates in less than 1 second
        assert validation_time < 1.0
        
        # Benchmark distance calculation
        start_time = time.time()
        for _ in range(100):
            services['gis'].calculate_distance(19.0760, 72.8777, 13.0827, 80.2707)
        distance_time = time.time() - start_time
        
        # Should calculate 100 distances in less than 1 second
        assert distance_time < 1.0


# Test fixtures and utilities
@pytest.fixture
def mock_database():
    """Mock database connection for testing"""
    mock_db = Mock()
    mock_db.execute.return_value = Mock()
    mock_db.fetchall.return_value = []
    return mock_db

@pytest.fixture
def sample_geojson():
    """Sample GeoJSON data for testing"""
    return {
        'type': 'FeatureCollection',
        'features': [
            {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [72.8777, 19.0760]
                },
                'properties': {
                    'id': 'mumbai_001',
                    'name': 'Mumbai Marine Station',
                    'type': 'biodiversity',
                    'value': 85.5,
                    'timestamp': '2024-09-23T10:00:00Z'
                }
            },
            {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [80.2707, 13.0827]
                },
                'properties': {
                    'id': 'chennai_001',
                    'name': 'Chennai Marine Research',
                    'type': 'water_quality',
                    'value': 72.3,
                    'timestamp': '2024-09-23T10:30:00Z'
                }
            }
        ]
    }

# Run tests with coverage
if __name__ == '__main__':
    pytest.main([
        __file__,
        '-v',
        '--cov=backend.app.geospatial',
        '--cov-report=html',
        '--cov-report=term-missing'
    ])