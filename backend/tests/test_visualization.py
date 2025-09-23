"""
Comprehensive visualization module tests for Ocean-Bio platform.

Tests dashboard statistics, charts, maps, and data visualization endpoints.
"""

import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient


@pytest.mark.visualization
class TestDashboardStatistics:
    """Test dashboard statistics endpoints."""
    
    def test_get_dashboard_stats(self, client: TestClient, researcher_headers: dict, sample_catch_records):
        """Test getting dashboard statistics."""
        response = client.get("/api/v1/visualization/dashboard-stats", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "overview" in data
        overview = data["overview"]
        assert "total_species" in overview
        assert "total_vessels" in overview
        assert "total_catches" in overview
        assert "total_trips" in overview
        assert "recent_catches" in overview
        
        # Check data types
        assert isinstance(overview["total_species"], int)
        assert isinstance(overview["total_vessels"], int)
        assert isinstance(overview["total_catches"], int)
        assert isinstance(overview["total_trips"], int)
        assert isinstance(overview["recent_catches"], int)
    
    def test_dashboard_stats_access_control(self, client: TestClient, viewer_headers: dict):
        """Test dashboard access for different user roles."""
        response = client.get("/api/v1/visualization/dashboard-stats", headers=viewer_headers)
        
        # Viewers should be able to access dashboard
        assert response.status_code == 200
    
    def test_dashboard_stats_without_auth(self, client: TestClient):
        """Test dashboard access without authentication."""
        response = client.get("/api/v1/visualization/dashboard-stats")
        
        assert response.status_code == 401


@pytest.mark.visualization
class TestSpeciesVisualization:
    """Test species-related visualization endpoints."""
    
    def test_get_species_distribution(self, client: TestClient, researcher_headers: dict, sample_catch_records):
        """Test getting species distribution data."""
        response = client.get("/api/v1/visualization/species-distribution", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "chart_type" in data
        assert "title" in data
        assert "data" in data
        assert data["chart_type"] == "pie"
        assert "Species Distribution" in data["title"]
        assert isinstance(data["data"], list)
        
        if data["data"]:  # If there's data
            item = data["data"][0]
            assert "species" in item
            assert "weight" in item
            assert "count" in item
            assert isinstance(item["weight"], (int, float))
            assert isinstance(item["count"], int)
    
    def test_species_distribution_with_limit(self, client: TestClient, researcher_headers: dict):
        """Test species distribution with custom limit."""
        response = client.get("/api/v1/visualization/species-distribution?limit=5", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) <= 5
    
    def test_species_distribution_invalid_limit(self, client: TestClient, researcher_headers: dict):
        """Test species distribution with invalid limit."""
        response = client.get("/api/v1/visualization/species-distribution?limit=100", headers=researcher_headers)
        
        # Should use maximum allowed limit (50)
        assert response.status_code == 422  # Validation error
    
    def test_biodiversity_metrics(self, client: TestClient, researcher_headers: dict, sample_taxonomic_units):
        """Test getting biodiversity metrics."""
        response = client.get("/api/v1/visualization/biodiversity-metrics", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "diversity_hierarchy" in data
        assert "chart_data" in data
        
        hierarchy = data["diversity_hierarchy"]
        assert "kingdoms" in hierarchy
        assert "phyla" in hierarchy
        assert "classes" in hierarchy
        assert "orders" in hierarchy
        assert "families" in hierarchy
        assert "genera" in hierarchy
        assert "species" in hierarchy
        
        chart_data = data["chart_data"]
        assert isinstance(chart_data, list)
        assert len(chart_data) == 7  # 7 taxonomic levels
        
        for level in chart_data:
            assert "level" in level
            assert "count" in level
            assert isinstance(level["count"], int)


@pytest.mark.visualization
class TestCatchTrends:
    """Test catch trends visualization endpoints."""
    
    def test_get_catch_trends_default(self, client: TestClient, researcher_headers: dict, sample_catch_records):
        """Test getting catch trends with default parameters."""
        response = client.get("/api/v1/visualization/catch-trends", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "chart_type" in data
        assert "title" in data
        assert "data" in data
        assert data["chart_type"] == "line"
        assert "Catch Trends" in data["title"]
        assert "30 Days" in data["title"]
        assert isinstance(data["data"], list)
    
    def test_catch_trends_custom_period(self, client: TestClient, researcher_headers: dict):
        """Test catch trends with custom time period."""
        response = client.get("/api/v1/visualization/catch-trends?days=7", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "7 Days" in data["title"]
    
    def test_catch_trends_invalid_period(self, client: TestClient, researcher_headers: dict):
        """Test catch trends with invalid period."""
        response = client.get("/api/v1/visualization/catch-trends?days=400", headers=researcher_headers)
        
        # Should return validation error for > 365 days
        assert response.status_code == 422
    
    def test_monthly_summary(self, client: TestClient, researcher_headers: dict, sample_catch_records):
        """Test getting monthly catch summary."""
        response = client.get("/api/v1/visualization/monthly-summary", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "chart_type" in data
        assert "title" in data
        assert "data" in data
        assert data["chart_type"] == "line"
        assert "Monthly Catch Summary" in data["title"]
        assert isinstance(data["data"], list)
        
        if data["data"]:  # If there's data
            item = data["data"][0]
            assert "period" in item
            assert "weight" in item
            assert "count" in item
            assert "species_count" in item
            
            # Check period format (YYYY-MM)
            assert len(item["period"]) == 7
            assert item["period"][4] == "-"
    
    def test_monthly_summary_custom_months(self, client: TestClient, researcher_headers: dict):
        """Test monthly summary with custom month range."""
        response = client.get("/api/v1/visualization/monthly-summary?months=6", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "6 Months" in data["title"]


@pytest.mark.visualization  
class TestGeospatialVisualization:
    """Test geospatial visualization endpoints."""
    
    def test_get_fishing_areas_data(self, client: TestClient, researcher_headers: dict, sample_catch_records):
        """Test getting fishing areas data."""
        response = client.get("/api/v1/visualization/fishing-areas", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "chart_type" in data
        assert "title" in data
        assert "data" in data
        assert data["chart_type"] == "bar"
        assert "Fishing Areas" in data["title"]
        assert isinstance(data["data"], list)
        
        if data["data"]:  # If there's data
            item = data["data"][0]
            assert "area" in item
            assert "weight" in item
            assert "count" in item
            assert isinstance(item["weight"], (int, float))
            assert isinstance(item["count"], int)
    
    def test_fishing_areas_with_limit(self, client: TestClient, researcher_headers: dict):
        """Test fishing areas with custom limit."""
        response = client.get("/api/v1/visualization/fishing-areas?limit=10", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) <= 10
    
    def test_get_heatmap_data(self, client: TestClient, researcher_headers: dict, sample_catch_records):
        """Test getting heatmap data."""
        response = client.get("/api/v1/visualization/heatmap-data", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "chart_type" in data
        assert "title" in data
        assert "data" in data
        assert data["chart_type"] == "heatmap"
        assert "Heatmap" in data["title"]
        assert isinstance(data["data"], list)
        
        if data["data"]:  # If there's data
            point = data["data"][0]
            assert "lat" in point
            assert "lon" in point
            assert "weight" in point
            assert "count" in point
            assert "intensity" in point
            
            # Check coordinate validity
            assert -90 <= point["lat"] <= 90
            assert -180 <= point["lon"] <= 180
            assert isinstance(point["weight"], (int, float))
            assert isinstance(point["intensity"], (int, float))
            assert 0 <= point["intensity"] <= 1


@pytest.mark.visualization
class TestVesselPerformance:
    """Test vessel performance visualization endpoints."""
    
    def test_get_vessel_performance(self, client: TestClient, researcher_headers: dict, sample_catch_records):
        """Test getting vessel performance data."""
        response = client.get("/api/v1/visualization/vessel-performance", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "chart_type" in data
        assert "title" in data  
        assert "data" in data
        assert data["chart_type"] == "bar"
        assert "Vessel" in data["title"]
        assert isinstance(data["data"], list)
        
        if data["data"]:  # If there's data
            vessel = data["data"][0]
            assert "vessel" in vessel
            assert "total_catch" in vessel
            assert "trip_count" in vessel
            assert "avg_catch" in vessel
            assert isinstance(vessel["total_catch"], (int, float))
            assert isinstance(vessel["trip_count"], int)
            assert isinstance(vessel["avg_catch"], (int, float))
    
    def test_vessel_performance_with_limit(self, client: TestClient, researcher_headers: dict):
        """Test vessel performance with custom limit."""
        response = client.get("/api/v1/visualization/vessel-performance?limit=5", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) <= 5
    
    def test_vessel_performance_invalid_limit(self, client: TestClient, researcher_headers: dict):
        """Test vessel performance with invalid limit."""
        response = client.get("/api/v1/visualization/vessel-performance?limit=50", headers=researcher_headers)
        
        # Should return validation error for > 30
        assert response.status_code == 422


@pytest.mark.visualization
class TestLegacyEndpoints:
    """Test legacy and compatibility endpoints."""
    
    def test_ocean_trends_legacy(self, client: TestClient, researcher_headers: dict):
        """Test legacy ocean trends endpoint."""
        response = client.get("/api/v1/visualization/trends/ocean", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "variable" in data
        assert "units" in data
        assert "series" in data
        assert data["variable"] == "Sea Surface Temperature"
        assert data["units"] == "°C"
        assert isinstance(data["series"], list)
        assert len(data["series"]) > 0
        
        for point in data["series"]:
            assert "date" in point
            assert "value" in point
            assert isinstance(point["value"], (int, float))


@pytest.mark.integration
class TestVisualizationIntegration:
    """Integration tests for visualization workflows."""
    
    def test_complete_dashboard_workflow(self, client: TestClient, admin_headers: dict, researcher_headers: dict, sample_taxonomic_units):
        """Test complete dashboard data workflow."""
        # 1. Create some data (vessel, trip, catch)
        vessel_data = {
            "vessel_name": "Viz Test Vessel",
            "registration_number": "VIZ-001-2024",
            "vessel_type": "TRAWLER",
            "length_meters": 35.0,
            "owner_name": "Viz Owner",
            "home_port": "Viz Port"
        }
        
        vessel_response = client.post("/api/v1/fisheries/vessels", json=vessel_data, headers=admin_headers)
        vessel_id = vessel_response.json()["id"]
        
        # Create trip
        trip_data = {
            "vessel_id": vessel_id,
            "departure_date": (datetime.utcnow() - timedelta(days=1)).isoformat(),
            "fishing_method": "BOTTOM_TRAWLING",
            "departure_port": "Viz Port",
            "trip_purpose": "Commercial fishing",
            "crew_count": 8
        }
        
        trip_response = client.post("/api/v1/fisheries/trips", json=trip_data, headers=admin_headers)
        trip_id = trip_response.json()["id"]
        
        # Create catch
        catch_data = {
            "vessel_id": vessel_id,
            "trip_id": trip_id,
            "species_id": sample_taxonomic_units[0].id,
            "catch_date": datetime.utcnow().isoformat(),
            "catch_weight": 2500.0,
            "catch_count": 125,
            "fishing_area": "Viz Area",
            "coordinates": "20.0,75.0",
            "fishing_method": "BOTTOM_TRAWLING"
        }
        
        client.post("/api/v1/fisheries/catches", json=catch_data, headers=admin_headers)
        
        # 2. Test dashboard reflects the data
        dashboard_response = client.get("/api/v1/visualization/dashboard-stats", headers=researcher_headers)
        assert dashboard_response.status_code == 200
        dashboard_data = dashboard_response.json()
        
        # Should have at least 1 vessel, 1 trip, 1 catch
        overview = dashboard_data["overview"]
        assert overview["total_vessels"] >= 1
        assert overview["total_trips"] >= 1
        assert overview["total_catches"] >= 1
        
        # 3. Test species distribution includes our species
        species_response = client.get("/api/v1/visualization/species-distribution", headers=researcher_headers)
        assert species_response.status_code == 200
        
        # 4. Test heatmap includes our coordinates
        heatmap_response = client.get("/api/v1/visualization/heatmap-data", headers=researcher_headers)
        assert heatmap_response.status_code == 200
        heatmap_data = heatmap_response.json()
        
        # Should have at least one point
        if heatmap_data["data"]:
            assert len(heatmap_data["data"]) >= 1
    
    def test_visualization_data_consistency(self, client: TestClient, researcher_headers: dict, sample_catch_records):
        """Test data consistency across visualization endpoints."""
        # Get data from multiple endpoints
        dashboard_response = client.get("/api/v1/visualization/dashboard-stats", headers=researcher_headers)
        species_response = client.get("/api/v1/visualization/species-distribution", headers=researcher_headers)
        trends_response = client.get("/api/v1/visualization/catch-trends", headers=researcher_headers)
        
        assert dashboard_response.status_code == 200
        assert species_response.status_code == 200
        assert trends_response.status_code == 200
        
        dashboard_data = dashboard_response.json()
        species_data = species_response.json()
        
        # Check consistency
        total_species = dashboard_data["overview"]["total_species"]
        unique_species_in_distribution = len(species_data["data"])
        
        # Species distribution should not show more species than total
        assert unique_species_in_distribution <= total_species


@pytest.mark.unit
class TestVisualizationHelpers:
    """Unit tests for visualization helper functions."""
    
    def test_coordinate_parsing(self):
        """Test coordinate string parsing for heatmap."""
        # This would test coordinate parsing utility functions
        # Implementation depends on specific helper functions
        pass
    
    def test_data_aggregation(self):
        """Test data aggregation for charts."""
        # This would test aggregation logic
        pass
    
    def test_chart_data_formatting(self):
        """Test chart data formatting functions."""
        # This would test data formatting utilities
        pass


@pytest.mark.visualization
class TestVisualizationErrors:
    """Test error handling in visualization endpoints."""
    
    def test_visualization_with_no_data(self, client: TestClient, researcher_headers: dict, db_session):
        """Test visualization endpoints when no data exists."""
        # This test should work with empty database
        
        # Dashboard should still work with zero counts
        response = client.get("/api/v1/visualization/dashboard-stats", headers=researcher_headers)
        assert response.status_code == 200
        data = response.json()
        
        overview = data["overview"]
        assert overview["total_species"] == 0
        assert overview["total_vessels"] == 0
        assert overview["total_catches"] == 0
        
        # Species distribution should return empty data
        response = client.get("/api/v1/visualization/species-distribution", headers=researcher_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["data"] == []
        
        # Heatmap should return empty data
        response = client.get("/api/v1/visualization/heatmap-data", headers=researcher_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["data"] == []
    
    def test_visualization_parameter_validation(self, client: TestClient, researcher_headers: dict):
        """Test parameter validation for visualization endpoints."""
        # Test invalid limit values
        response = client.get("/api/v1/visualization/species-distribution?limit=0", headers=researcher_headers)
        assert response.status_code == 422
        
        response = client.get("/api/v1/visualization/species-distribution?limit=-5", headers=researcher_headers)
        assert response.status_code == 422
        
        # Test invalid days values for trends
        response = client.get("/api/v1/visualization/catch-trends?days=0", headers=researcher_headers)
        assert response.status_code == 422
        
        response = client.get("/api/v1/visualization/catch-trends?days=400", headers=researcher_headers)
        assert response.status_code == 422
        
        # Test invalid months values
        response = client.get("/api/v1/visualization/monthly-summary?months=0", headers=researcher_headers)
        assert response.status_code == 422
        
        response = client.get("/api/v1/visualization/monthly-summary?months=30", headers=researcher_headers)
        assert response.status_code == 422


@pytest.mark.performance
class TestVisualizationPerformance:
    """Performance tests for visualization endpoints."""
    
    def test_dashboard_response_time(self, client: TestClient, researcher_headers: dict):
        """Test dashboard response time."""
        import time
        
        start_time = time.time()
        response = client.get("/api/v1/visualization/dashboard-stats", headers=researcher_headers)
        end_time = time.time()
        
        assert response.status_code == 200
        # Dashboard should respond within 2 seconds
        assert (end_time - start_time) < 2.0
    
    def test_large_dataset_visualization(self, client: TestClient, researcher_headers: dict):
        """Test visualization performance with large datasets."""
        # This would test with a large number of records
        # For now, just ensure it doesn't timeout
        
        response = client.get("/api/v1/visualization/species-distribution?limit=50", headers=researcher_headers)
        assert response.status_code in [200, 404]  # 200 with data or 404 with no data
        
        response = client.get("/api/v1/visualization/heatmap-data", headers=researcher_headers)
        assert response.status_code == 200