"""
Test configuration and fixtures for the Ocean-Bio backend platform.

Provides pytest fixtures for database, authentication, and test data.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock
from typing import Generator, Dict, Any
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app
from app.core.database import get_db, Base
from app.core.security import get_password_hash, create_access_token
from app.models.user import User, UserRole, UserStatus
from app.models.taxonomy import TaxonomicUnit
from app.models.fisheries import (
    VesselType, FishingMethod, FishingVessel, 
    FishingTrip, CatchRecord, FishingQuota, MarketPrice
)
from app.models.otolith import (
    OtolithSpecimen, OtolithMeasurement, 
    OtolithImage, OtolithClassification, OtolithStudy
)

# Test database URL (SQLite in-memory for fast tests)
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_oceanbio.db"

# Create test engine and session
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False},
    echo=False
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database session for each test."""
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Create session
    session = TestingSessionLocal()
    
    try:
        yield session
    finally:
        session.close()
        # Drop all tables after test
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    """Create a test client with overridden database dependency."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
async def async_client(db_session):
    """Create an async test client."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as async_test_client:
        yield async_test_client
    
    app.dependency_overrides.clear()


@pytest.fixture
def admin_user(db_session) -> User:
    """Create an admin user for testing."""
    user = User(
        username="admin_test",
        email="admin@test.com",
        hashed_password=get_password_hash("testpassword123"),
        full_name="Test Admin",
        role=UserRole.ADMIN,
        status=UserStatus.ACTIVE,
        created_at=datetime.utcnow()
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def researcher_user(db_session) -> User:
    """Create a researcher user for testing."""
    user = User(
        username="researcher_test",
        email="researcher@test.com",
        hashed_password=get_password_hash("testpassword123"),
        full_name="Test Researcher",
        role=UserRole.RESEARCHER,
        status=UserStatus.ACTIVE,
        created_at=datetime.utcnow()
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def viewer_user(db_session) -> User:
    """Create a viewer user for testing."""
    user = User(
        username="viewer_test",
        email="viewer@test.com",
        hashed_password=get_password_hash("testpassword123"),
        full_name="Test Viewer",
        role=UserRole.VIEWER,
        status=UserStatus.ACTIVE,
        created_at=datetime.utcnow()
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def admin_headers(admin_user) -> Dict[str, str]:
    """Create authorization headers for admin user."""
    access_token = create_access_token(
        data={"sub": admin_user.username}
    )
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture
def researcher_headers(researcher_user) -> Dict[str, str]:
    """Create authorization headers for researcher user."""
    access_token = create_access_token(
        data={"sub": researcher_user.username}
    )
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture
def viewer_headers(viewer_user) -> Dict[str, str]:
    """Create authorization headers for viewer user."""
    access_token = create_access_token(
        data={"sub": viewer_user.username}
    )
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture
def sample_taxonomic_units(db_session) -> list[TaxonomicUnit]:
    """Create sample taxonomic units for testing."""
    units = [
        TaxonomicUnit(
            kingdom="Animalia",
            phylum="Chordata",
            class_name="Actinopterygii",
            order_name="Perciformes",
            family="Scombridae",
            genus="Thunnus",
            species="thynnus",
            scientific_name="Thunnus thynnus",
            common_name="Atlantic Bluefin Tuna"
        ),
        TaxonomicUnit(
            kingdom="Animalia",
            phylum="Chordata", 
            class_name="Actinopterygii",
            order_name="Gadiformes",
            family="Gadidae",
            genus="Gadus",
            species="morhua",
            scientific_name="Gadus morhua",
            common_name="Atlantic Cod"
        ),
        TaxonomicUnit(
            kingdom="Animalia",
            phylum="Chordata",
            class_name="Actinopterygii", 
            order_name="Clupeiformes",
            family="Clupeidae",
            genus="Clupea",
            species="harengus",
            scientific_name="Clupea harengus",
            common_name="Atlantic Herring"
        )
    ]
    
    for unit in units:
        db_session.add(unit)
    db_session.commit()
    
    for unit in units:
        db_session.refresh(unit)
    
    return units


@pytest.fixture
def sample_vessels(db_session) -> list[FishingVessel]:
    """Create sample fishing vessels for testing."""
    vessels = [
        FishingVessel(
            vessel_name="Ocean Explorer",
            registration_number="IND-001-2024",
            vessel_type=VesselType.TRAWLER,
            length_meters=45.5,
            owner_name="Marine Fisheries Co.",
            home_port="Mumbai",
            gross_tonnage=250.0,
            engine_power_hp=500,
            crew_capacity=12,
            registration_date=datetime.utcnow() - timedelta(days=365)
        ),
        FishingVessel(
            vessel_name="Coastal Navigator",
            registration_number="IND-002-2024",
            vessel_type=VesselType.PURSE_SEINER,
            length_meters=32.8,
            owner_name="Coastal Fishing Ltd.",
            home_port="Chennai",
            gross_tonnage=180.0,
            engine_power_hp=350,
            crew_capacity=8,
            registration_date=datetime.utcnow() - timedelta(days=200)
        )
    ]
    
    for vessel in vessels:
        db_session.add(vessel)
    db_session.commit()
    
    for vessel in vessels:
        db_session.refresh(vessel)
    
    return vessels


@pytest.fixture
def sample_fishing_trips(db_session, sample_vessels) -> list[FishingTrip]:
    """Create sample fishing trips for testing."""
    trips = [
        FishingTrip(
            vessel_id=sample_vessels[0].id,
            departure_date=datetime.utcnow() - timedelta(days=10),
            return_date=datetime.utcnow() - timedelta(days=7),
            fishing_method=FishingMethod.BOTTOM_TRAWLING,
            departure_port="Mumbai",
            return_port="Mumbai",
            trip_purpose="Commercial fishing",
            crew_count=10,
            fuel_consumed_liters=2500.0,
            total_catch_weight=15000.0
        ),
        FishingTrip(
            vessel_id=sample_vessels[1].id,
            departure_date=datetime.utcnow() - timedelta(days=5),
            return_date=datetime.utcnow() - timedelta(days=3),
            fishing_method=FishingMethod.PURSE_SEINING,
            departure_port="Chennai",
            return_port="Chennai", 
            trip_purpose="Commercial fishing",
            crew_count=6,
            fuel_consumed_liters=1800.0,
            total_catch_weight=12000.0
        )
    ]
    
    for trip in trips:
        db_session.add(trip)
    db_session.commit()
    
    for trip in trips:
        db_session.refresh(trip)
    
    return trips


@pytest.fixture
def sample_catch_records(db_session, sample_vessels, sample_fishing_trips, sample_taxonomic_units) -> list[CatchRecord]:
    """Create sample catch records for testing."""
    catches = [
        CatchRecord(
            vessel_id=sample_vessels[0].id,
            trip_id=sample_fishing_trips[0].id,
            species_id=sample_taxonomic_units[0].id,
            catch_date=datetime.utcnow() - timedelta(days=8),
            catch_weight=5000.0,
            catch_count=250,
            fishing_area="Arabian Sea - Zone 1",
            coordinates="18.5204,72.8492",
            depth_meters=45,
            fishing_method=FishingMethod.BOTTOM_TRAWLING,
            gear_details="Bottom trawl net, mesh size 50mm"
        ),
        CatchRecord(
            vessel_id=sample_vessels[1].id,
            trip_id=sample_fishing_trips[1].id,
            species_id=sample_taxonomic_units[2].id,
            catch_date=datetime.utcnow() - timedelta(days=4),
            catch_weight=8000.0,
            catch_count=1600,
            fishing_area="Bay of Bengal - Zone 3",
            coordinates="13.0827,80.2707",
            depth_meters=25,
            fishing_method=FishingMethod.PURSE_SEINING,
            gear_details="Purse seine net, mesh size 30mm"
        )
    ]
    
    for catch in catches:
        db_session.add(catch)
    db_session.commit()
    
    for catch in catches:
        db_session.refresh(catch)
    
    return catches


@pytest.fixture
def sample_otolith_specimens(db_session, sample_taxonomic_units) -> list[OtolithSpecimen]:
    """Create sample otolith specimens for testing."""
    specimens = [
        OtolithSpecimen(
            specimen_id="OTO-001-2024",
            species_id=sample_taxonomic_units[0].id,
            collection_date=datetime.utcnow() - timedelta(days=30),
            collection_location="Arabian Sea",
            collection_coordinates="18.5204,72.8492",
            fish_length_mm=750.0,
            fish_weight_g=15000.0,
            fish_age_years=5,
            sex="M",
            maturity_stage="Adult",
            otolith_type="Sagitta",
            preservation_method="Dry",
            collector_name="Dr. Marine Researcher",
            collection_depth_m=50
        ),
        OtolithSpecimen(
            specimen_id="OTO-002-2024",
            species_id=sample_taxonomic_units[1].id,
            collection_date=datetime.utcnow() - timedelta(days=20),
            collection_location="Bay of Bengal",
            collection_coordinates="13.0827,80.2707",
            fish_length_mm=450.0,
            fish_weight_g=2500.0,
            fish_age_years=3,
            sex="F",
            maturity_stage="Adult",
            otolith_type="Sagitta",
            preservation_method="Alcohol",
            collector_name="Dr. Ocean Scientist",
            collection_depth_m=75
        )
    ]
    
    for specimen in specimens:
        db_session.add(specimen)
    db_session.commit()
    
    for specimen in specimens:
        db_session.refresh(specimen)
    
    return specimens


@pytest.fixture
def mock_file_upload():
    """Mock file upload for testing."""
    mock_file = Mock()
    mock_file.filename = "test_otolith.jpg"
    mock_file.content_type = "image/jpeg"
    mock_file.size = 1024000  # 1MB
    mock_file.read.return_value = b"fake_image_data"
    return mock_file


# Pytest configuration
pytest_plugins = []


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "auth: marks tests related to authentication"
    )
    config.addinivalue_line(
        "markers", "fisheries: marks tests related to fisheries module"
    )
    config.addinivalue_line(
        "markers", "otolith: marks tests related to otolith module"
    )
    config.addinivalue_line(
        "markers", "visualization: marks tests related to visualization module"
    )


# Test data constants
TEST_USER_PASSWORD = "testpassword123"
TEST_ADMIN_USERNAME = "admin_test"
TEST_RESEARCHER_USERNAME = "researcher_test"
TEST_VIEWER_USERNAME = "viewer_test"