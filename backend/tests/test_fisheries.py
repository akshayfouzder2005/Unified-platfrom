"""
Comprehensive fisheries module tests for Ocean-Bio platform.

Tests vessels, trips, catches, quotas, market prices, and analytics.
"""

import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from app.models.fisheries import VesselType, FishingMethod, FishingVessel, FishingTrip, CatchRecord


@pytest.mark.fisheries
class TestVesselsAPI:
    """Test vessel management endpoints."""
    
    def test_create_vessel_success(self, client: TestClient, admin_headers: dict):
        """Test successful vessel creation."""
        vessel_data = {
            "vessel_name": "New Fishing Vessel",
            "registration_number": "IND-NEW-2024",
            "vessel_type": "TRAWLER",
            "length_meters": 42.5,
            "owner_name": "New Owner Co.",
            "home_port": "Kochi",
            "gross_tonnage": 220.0,
            "engine_power_hp": 450,
            "crew_capacity": 10
        }
        
        response = client.post("/api/v1/fisheries/vessels", json=vessel_data, headers=admin_headers)
        
        assert response.status_code == 201
        data = response.json()
        assert data["vessel_name"] == vessel_data["vessel_name"]
        assert data["registration_number"] == vessel_data["registration_number"]
        assert data["vessel_type"] == vessel_data["vessel_type"]
        assert data["length_meters"] == vessel_data["length_meters"]
        assert "id" in data
        assert "registration_date" in data
    
    def test_create_vessel_duplicate_registration(self, client: TestClient, admin_headers: dict, sample_vessels):
        """Test vessel creation with duplicate registration number."""
        vessel_data = {
            "vessel_name": "Different Vessel",
            "registration_number": sample_vessels[0].registration_number,
            "vessel_type": "LONGLINER",
            "length_meters": 35.0,
            "owner_name": "Different Owner",
            "home_port": "Different Port"
        }
        
        response = client.post("/api/v1/fisheries/vessels", json=vessel_data, headers=admin_headers)
        
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"].lower()
    
    def test_get_vessels_list(self, client: TestClient, researcher_headers: dict, sample_vessels):
        """Test getting list of vessels."""
        response = client.get("/api/v1/fisheries/vessels", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "size" in data
        assert len(data["items"]) == len(sample_vessels)
        
        # Check vessel data
        vessel_names = [v["vessel_name"] for v in data["items"]]
        expected_names = [v.vessel_name for v in sample_vessels]
        assert all(name in vessel_names for name in expected_names)
    
    def test_get_vessel_by_id(self, client: TestClient, researcher_headers: dict, sample_vessels):
        """Test getting specific vessel by ID."""
        vessel = sample_vessels[0]
        response = client.get(f"/api/v1/fisheries/vessels/{vessel.id}", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == vessel.id
        assert data["vessel_name"] == vessel.vessel_name
        assert data["registration_number"] == vessel.registration_number
    
    def test_get_nonexistent_vessel(self, client: TestClient, researcher_headers: dict):
        """Test getting nonexistent vessel."""
        response = client.get("/api/v1/fisheries/vessels/999999", headers=researcher_headers)
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_update_vessel(self, client: TestClient, admin_headers: dict, sample_vessels):
        """Test updating vessel information."""
        vessel = sample_vessels[0]
        update_data = {
            "vessel_name": "Updated Vessel Name",
            "home_port": "Updated Port",
            "engine_power_hp": 600
        }
        
        response = client.patch(f"/api/v1/fisheries/vessels/{vessel.id}", json=update_data, headers=admin_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["vessel_name"] == update_data["vessel_name"]
        assert data["home_port"] == update_data["home_port"]
        assert data["engine_power_hp"] == update_data["engine_power_hp"]
        # Other fields should remain unchanged
        assert data["registration_number"] == vessel.registration_number
    
    def test_delete_vessel(self, client: TestClient, admin_headers: dict, sample_vessels):
        """Test deleting vessel."""
        vessel = sample_vessels[0]
        response = client.delete(f"/api/v1/fisheries/vessels/{vessel.id}", headers=admin_headers)
        
        assert response.status_code == 200
        assert "deleted successfully" in response.json()["message"].lower()
        
        # Verify vessel is deleted
        response = client.get(f"/api/v1/fisheries/vessels/{vessel.id}", headers=admin_headers)
        assert response.status_code == 404
    
    def test_vessel_access_control(self, client: TestClient, viewer_headers: dict):
        """Test vessel access control for different user roles."""
        vessel_data = {
            "vessel_name": "Test Vessel",
            "registration_number": "TEST-001",
            "vessel_type": "TRAWLER",
            "length_meters": 30.0,
            "owner_name": "Test Owner",
            "home_port": "Test Port"
        }
        
        # Viewer should not be able to create
        response = client.post("/api/v1/fisheries/vessels", json=vessel_data, headers=viewer_headers)
        assert response.status_code == 403
        
        # Viewer should be able to read
        response = client.get("/api/v1/fisheries/vessels", headers=viewer_headers)
        assert response.status_code == 200


@pytest.mark.fisheries
class TestTripsAPI:
    """Test fishing trip management endpoints."""
    
    def test_create_trip_success(self, client: TestClient, admin_headers: dict, sample_vessels):
        """Test successful trip creation."""
        trip_data = {
            "vessel_id": sample_vessels[0].id,
            "departure_date": (datetime.utcnow() - timedelta(days=1)).isoformat(),
            "fishing_method": "BOTTOM_TRAWLING",
            "departure_port": "Mumbai",
            "trip_purpose": "Commercial fishing",
            "crew_count": 8,
            "fuel_consumed_liters": 2000.0
        }
        
        response = client.post("/api/v1/fisheries/trips", json=trip_data, headers=admin_headers)
        
        assert response.status_code == 201
        data = response.json()
        assert data["vessel_id"] == trip_data["vessel_id"]
        assert data["fishing_method"] == trip_data["fishing_method"]
        assert data["departure_port"] == trip_data["departure_port"]
        assert data["crew_count"] == trip_data["crew_count"]
        assert "id" in data
    
    def test_create_trip_nonexistent_vessel(self, client: TestClient, admin_headers: dict):
        """Test trip creation with nonexistent vessel."""
        trip_data = {
            "vessel_id": 999999,
            "departure_date": datetime.utcnow().isoformat(),
            "fishing_method": "BOTTOM_TRAWLING",
            "departure_port": "Mumbai",
            "trip_purpose": "Commercial fishing",
            "crew_count": 8
        }
        
        response = client.post("/api/v1/fisheries/trips", json=trip_data, headers=admin_headers)
        
        assert response.status_code == 400
        assert "vessel" in response.json()["detail"].lower()
    
    def test_get_trips_list(self, client: TestClient, researcher_headers: dict, sample_fishing_trips):
        """Test getting list of trips."""
        response = client.get("/api/v1/fisheries/trips", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) == len(sample_fishing_trips)
    
    def test_get_trips_by_vessel(self, client: TestClient, researcher_headers: dict, sample_vessels, sample_fishing_trips):
        """Test getting trips filtered by vessel."""
        vessel_id = sample_vessels[0].id
        response = client.get(f"/api/v1/fisheries/trips?vessel_id={vessel_id}", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        # All returned trips should be for the specified vessel
        for trip in data["items"]:
            assert trip["vessel_id"] == vessel_id
    
    def test_update_trip(self, client: TestClient, admin_headers: dict, sample_fishing_trips):
        """Test updating trip information."""
        trip = sample_fishing_trips[0]
        update_data = {
            "return_date": datetime.utcnow().isoformat(),
            "return_port": "Updated Port",
            "total_catch_weight": 18000.0
        }
        
        response = client.patch(f"/api/v1/fisheries/trips/{trip.id}", json=update_data, headers=admin_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["return_port"] == update_data["return_port"]
        assert data["total_catch_weight"] == update_data["total_catch_weight"]
    
    def test_complete_trip(self, client: TestClient, admin_headers: dict, sample_fishing_trips):
        """Test completing a fishing trip."""
        trip = sample_fishing_trips[0]
        completion_data = {
            "return_date": datetime.utcnow().isoformat(),
            "return_port": "Mumbai",
            "total_catch_weight": 16000.0,
            "fuel_consumed_liters": 2800.0
        }
        
        response = client.post(f"/api/v1/fisheries/trips/{trip.id}/complete", json=completion_data, headers=admin_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["return_date"] is not None
        assert data["total_catch_weight"] == completion_data["total_catch_weight"]


@pytest.mark.fisheries
class TestCatchRecordsAPI:
    """Test catch record management endpoints."""
    
    def test_create_catch_record(self, client: TestClient, admin_headers: dict, sample_vessels, sample_fishing_trips, sample_taxonomic_units):
        """Test creating catch record."""
        catch_data = {
            "vessel_id": sample_vessels[0].id,
            "trip_id": sample_fishing_trips[0].id,
            "species_id": sample_taxonomic_units[0].id,
            "catch_date": (datetime.utcnow() - timedelta(days=2)).isoformat(),
            "catch_weight": 3000.0,
            "catch_count": 150,
            "fishing_area": "Arabian Sea - Zone 2",
            "coordinates": "19.0760,72.8777",
            "depth_meters": 35,
            "fishing_method": "BOTTOM_TRAWLING",
            "gear_details": "Bottom trawl net, mesh size 45mm"
        }
        
        response = client.post("/api/v1/fisheries/catches", json=catch_data, headers=admin_headers)
        
        assert response.status_code == 201
        data = response.json()
        assert data["vessel_id"] == catch_data["vessel_id"]
        assert data["species_id"] == catch_data["species_id"]
        assert data["catch_weight"] == catch_data["catch_weight"]
        assert data["fishing_area"] == catch_data["fishing_area"]
    
    def test_get_catch_records(self, client: TestClient, researcher_headers: dict, sample_catch_records):
        """Test getting catch records."""
        response = client.get("/api/v1/fisheries/catches", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) == len(sample_catch_records)
    
    def test_get_catches_by_species(self, client: TestClient, researcher_headers: dict, sample_catch_records, sample_taxonomic_units):
        """Test getting catches filtered by species."""
        species_id = sample_taxonomic_units[0].id
        response = client.get(f"/api/v1/fisheries/catches?species_id={species_id}", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        # All returned catches should be for the specified species
        for catch in data["items"]:
            assert catch["species_id"] == species_id
    
    def test_get_catches_by_date_range(self, client: TestClient, researcher_headers: dict, sample_catch_records):
        """Test getting catches within date range."""
        start_date = (datetime.utcnow() - timedelta(days=15)).date().isoformat()
        end_date = datetime.utcnow().date().isoformat()
        
        response = client.get(f"/api/v1/fisheries/catches?start_date={start_date}&end_date={end_date}", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
    
    def test_update_catch_record(self, client: TestClient, admin_headers: dict, sample_catch_records):
        """Test updating catch record."""
        catch = sample_catch_records[0]
        update_data = {
            "catch_weight": 6000.0,
            "catch_count": 300,
            "gear_details": "Updated gear information"
        }
        
        response = client.patch(f"/api/v1/fisheries/catches/{catch.id}", json=update_data, headers=admin_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["catch_weight"] == update_data["catch_weight"]
        assert data["catch_count"] == update_data["catch_count"]
        assert data["gear_details"] == update_data["gear_details"]


@pytest.mark.fisheries
class TestQuotasAPI:
    """Test fishing quota management endpoints."""
    
    def test_create_quota(self, client: TestClient, admin_headers: dict, sample_taxonomic_units):
        """Test creating fishing quota."""
        quota_data = {
            "species_id": sample_taxonomic_units[0].id,
            "fishing_area": "Arabian Sea",
            "quota_amount": 50000.0,
            "time_period": "2024-Q1",
            "quota_type": "COMMERCIAL",
            "allocated_to": "Western Coast Fisheries",
            "start_date": "2024-01-01",
            "end_date": "2024-03-31"
        }
        
        response = client.post("/api/v1/fisheries/quotas", json=quota_data, headers=admin_headers)
        
        assert response.status_code == 201
        data = response.json()
        assert data["species_id"] == quota_data["species_id"]
        assert data["quota_amount"] == quota_data["quota_amount"]
        assert data["fishing_area"] == quota_data["fishing_area"]
        assert data["quota_type"] == quota_data["quota_type"]
    
    def test_get_quotas(self, client: TestClient, researcher_headers: dict):
        """Test getting quotas list."""
        response = client.get("/api/v1/fisheries/quotas", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
    
    def test_update_quota(self, client: TestClient, admin_headers: dict, sample_taxonomic_units):
        """Test updating quota."""
        # First create a quota
        quota_data = {
            "species_id": sample_taxonomic_units[0].id,
            "fishing_area": "Bay of Bengal",
            "quota_amount": 40000.0,
            "time_period": "2024-Q2",
            "quota_type": "COMMERCIAL"
        }
        
        create_response = client.post("/api/v1/fisheries/quotas", json=quota_data, headers=admin_headers)
        quota_id = create_response.json()["id"]
        
        # Update the quota
        update_data = {
            "quota_amount": 45000.0,
            "used_amount": 15000.0
        }
        
        response = client.patch(f"/api/v1/fisheries/quotas/{quota_id}", json=update_data, headers=admin_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["quota_amount"] == update_data["quota_amount"]
        assert data["used_amount"] == update_data["used_amount"]


@pytest.mark.fisheries
class TestMarketPricesAPI:
    """Test market price management endpoints."""
    
    def test_create_market_price(self, client: TestClient, admin_headers: dict, sample_taxonomic_units):
        """Test creating market price record."""
        price_data = {
            "species_id": sample_taxonomic_units[0].id,
            "market_location": "Mumbai Fish Market",
            "price_per_kg": 450.0,
            "price_date": datetime.utcnow().date().isoformat(),
            "quality_grade": "Grade A",
            "market_conditions": "High demand",
            "recorded_by": "Market Surveyor"
        }
        
        response = client.post("/api/v1/fisheries/market-prices", json=price_data, headers=admin_headers)
        
        assert response.status_code == 201
        data = response.json()
        assert data["species_id"] == price_data["species_id"]
        assert data["market_location"] == price_data["market_location"]
        assert data["price_per_kg"] == price_data["price_per_kg"]
        assert data["quality_grade"] == price_data["quality_grade"]
    
    def test_get_market_prices(self, client: TestClient, researcher_headers: dict):
        """Test getting market prices."""
        response = client.get("/api/v1/fisheries/market-prices", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
    
    def test_get_prices_by_species(self, client: TestClient, researcher_headers: dict, sample_taxonomic_units):
        """Test getting prices for specific species."""
        species_id = sample_taxonomic_units[0].id
        response = client.get(f"/api/v1/fisheries/market-prices?species_id={species_id}", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        # All returned prices should be for the specified species
        for price in data["items"]:
            assert price["species_id"] == species_id
    
    def test_get_latest_prices(self, client: TestClient, researcher_headers: dict):
        """Test getting latest market prices."""
        response = client.get("/api/v1/fisheries/market-prices/latest", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.fisheries
class TestAnalyticsAPI:
    """Test fisheries analytics endpoints."""
    
    def test_get_catch_statistics(self, client: TestClient, researcher_headers: dict, sample_catch_records):
        """Test getting catch statistics."""
        response = client.get("/api/v1/fisheries/analytics/catch-statistics", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "total_catch_weight" in data
        assert "total_catch_count" in data
        assert "unique_species" in data
        assert "active_vessels" in data
        assert "statistics_period" in data
    
    def test_get_species_breakdown(self, client: TestClient, researcher_headers: dict, sample_catch_records):
        """Test getting species breakdown."""
        response = client.get("/api/v1/fisheries/analytics/species-breakdown", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        if data:  # If there's data
            assert "species_name" in data[0]
            assert "total_weight" in data[0]
            assert "total_count" in data[0]
            assert "percentage_of_total" in data[0]
    
    def test_get_catch_trends(self, client: TestClient, researcher_headers: dict, sample_catch_records):
        """Test getting catch trends."""
        response = client.get("/api/v1/fisheries/analytics/catch-trends?period=30", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "period" in data
        assert "trends" in data
        assert isinstance(data["trends"], list)
    
    def test_get_vessel_performance(self, client: TestClient, researcher_headers: dict, sample_catch_records, sample_vessels):
        """Test getting vessel performance metrics."""
        response = client.get("/api/v1/fisheries/analytics/vessel-performance", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        if data:  # If there's data
            assert "vessel_name" in data[0]
            assert "total_catch" in data[0]
            assert "trip_count" in data[0]
            assert "efficiency_score" in data[0]
    
    def test_get_fishing_area_analysis(self, client: TestClient, researcher_headers: dict, sample_catch_records):
        """Test getting fishing area analysis."""
        response = client.get("/api/v1/fisheries/analytics/fishing-areas", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        if data:  # If there's data
            assert "fishing_area" in data[0]
            assert "total_catch" in data[0]
            assert "species_diversity" in data[0]


@pytest.mark.fisheries
class TestUtilityEndpoints:
    """Test utility endpoints for fisheries module."""
    
    def test_get_vessel_types(self, client: TestClient, researcher_headers: dict):
        """Test getting vessel types enum."""
        response = client.get("/api/v1/fisheries/vessel-types", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        # Check that common vessel types are included
        vessel_types = [item["value"] for item in data]
        assert "TRAWLER" in vessel_types
        assert "PURSE_SEINER" in vessel_types
    
    def test_get_fishing_methods(self, client: TestClient, researcher_headers: dict):
        """Test getting fishing methods enum."""
        response = client.get("/api/v1/fisheries/fishing-methods", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        # Check that common fishing methods are included
        methods = [item["value"] for item in data]
        assert "BOTTOM_TRAWLING" in methods
        assert "PURSE_SEINING" in methods
    
    def test_get_ports(self, client: TestClient, researcher_headers: dict):
        """Test getting available ports."""
        response = client.get("/api/v1/fisheries/ports", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_fishing_areas(self, client: TestClient, researcher_headers: dict):
        """Test getting fishing areas."""
        response = client.get("/api/v1/fisheries/fishing-areas", headers=researcher_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.integration
class TestFisheriesIntegration:
    """Integration tests for fisheries module workflow."""
    
    def test_complete_fishing_workflow(self, client: TestClient, admin_headers: dict, researcher_headers: dict, sample_taxonomic_units):
        """Test complete fishing workflow from vessel to catch."""
        # 1. Create vessel
        vessel_data = {
            "vessel_name": "Integration Vessel",
            "registration_number": "INT-001-2024",
            "vessel_type": "TRAWLER",
            "length_meters": 40.0,
            "owner_name": "Integration Owner",
            "home_port": "Integration Port"
        }
        
        vessel_response = client.post("/api/v1/fisheries/vessels", json=vessel_data, headers=admin_headers)
        assert vessel_response.status_code == 201
        vessel_id = vessel_response.json()["id"]
        
        # 2. Create fishing trip
        trip_data = {
            "vessel_id": vessel_id,
            "departure_date": (datetime.utcnow() - timedelta(days=2)).isoformat(),
            "fishing_method": "BOTTOM_TRAWLING",
            "departure_port": "Integration Port",
            "trip_purpose": "Commercial fishing",
            "crew_count": 10
        }
        
        trip_response = client.post("/api/v1/fisheries/trips", json=trip_data, headers=admin_headers)
        assert trip_response.status_code == 201
        trip_id = trip_response.json()["id"]
        
        # 3. Record catch
        catch_data = {
            "vessel_id": vessel_id,
            "trip_id": trip_id,
            "species_id": sample_taxonomic_units[0].id,
            "catch_date": (datetime.utcnow() - timedelta(days=1)).isoformat(),
            "catch_weight": 4000.0,
            "catch_count": 200,
            "fishing_area": "Integration Area",
            "fishing_method": "BOTTOM_TRAWLING"
        }
        
        catch_response = client.post("/api/v1/fisheries/catches", json=catch_data, headers=admin_headers)
        assert catch_response.status_code == 201
        
        # 4. Complete trip
        completion_data = {
            "return_date": datetime.utcnow().isoformat(),
            "return_port": "Integration Port",
            "total_catch_weight": 4000.0
        }
        
        complete_response = client.post(f"/api/v1/fisheries/trips/{trip_id}/complete", json=completion_data, headers=admin_headers)
        assert complete_response.status_code == 200
        
        # 5. Verify statistics are updated
        stats_response = client.get("/api/v1/fisheries/analytics/catch-statistics", headers=researcher_headers)
        assert stats_response.status_code == 200
        stats = stats_response.json()
        assert stats["total_catch_weight"] >= 4000.0
    
    def test_quota_tracking_workflow(self, client: TestClient, admin_headers: dict, sample_taxonomic_units):
        """Test quota tracking workflow."""
        # 1. Create quota
        quota_data = {
            "species_id": sample_taxonomic_units[0].id,
            "fishing_area": "Test Area",
            "quota_amount": 10000.0,
            "time_period": "2024-Q3",
            "quota_type": "COMMERCIAL"
        }
        
        quota_response = client.post("/api/v1/fisheries/quotas", json=quota_data, headers=admin_headers)
        assert quota_response.status_code == 201
        quota_id = quota_response.json()["id"]
        
        # 2. Update quota usage
        update_data = {"used_amount": 3000.0}
        
        update_response = client.patch(f"/api/v1/fisheries/quotas/{quota_id}", json=update_data, headers=admin_headers)
        assert update_response.status_code == 200
        
        # 3. Verify quota status
        get_response = client.get(f"/api/v1/fisheries/quotas/{quota_id}", headers=admin_headers)
        assert get_response.status_code == 200
        quota = get_response.json()
        assert quota["used_amount"] == 3000.0
        assert quota["remaining_amount"] == 7000.0


@pytest.mark.unit
class TestFisheriesBusinessLogic:
    """Unit tests for fisheries business logic."""
    
    def test_vessel_efficiency_calculation(self, sample_vessels, sample_catch_records):
        """Test vessel efficiency calculation logic."""
        # This would test business logic functions
        # Implementation depends on specific business logic in crud module
        pass
    
    def test_quota_validation(self):
        """Test quota validation logic."""
        # This would test quota validation business rules
        pass
    
    def test_catch_statistics_calculation(self):
        """Test catch statistics calculation."""
        # This would test statistics calculation functions
        pass