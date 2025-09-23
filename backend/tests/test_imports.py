"""
Simple import validation tests for Ocean-Bio platform.

These tests validate that all critical modules can be imported
without runtime errors, ensuring the basic platform structure is sound.
"""

def test_core_exceptions_import():
    """Test that all exception classes can be imported."""
    from app.core.exceptions import (
        ValidationError,
        DataValidationError,
        DatabaseError,
        AuthenticationError,
        AuthorizationError,
        IngestionError,
        DatabaseConnectionError
    )
    
    assert ValidationError is not None
    assert DataValidationError is not None
    assert DatabaseError is not None
    assert AuthenticationError is not None
    assert AuthorizationError is not None
    assert IngestionError is not None
    assert DatabaseConnectionError is not None


def test_security_module_import():
    """Test that security utilities can be imported."""
    from app.core.security import (
        get_password_hash,
        verify_password,
        create_access_token,
        verify_token
    )
    
    assert get_password_hash is not None
    assert verify_password is not None
    assert create_access_token is not None
    assert verify_token is not None


def test_user_schemas_import():
    """Test that user schemas can be imported."""
    from app.schemas.user import (
        UserCreate,
        UserResponse,
        UserUpdate,
        Token,
        PasswordChange
    )
    
    assert UserCreate is not None
    assert UserResponse is not None
    assert UserUpdate is not None
    assert Token is not None
    assert PasswordChange is not None


def test_fisheries_schemas_import():
    """Test that fisheries schemas can be imported."""
    from app.schemas.fisheries import (
        VesselCreate,
        CatchRecordCreate,
        FishingVesselCreate,
        FishingTripCreate
    )
    
    assert VesselCreate is not None
    assert CatchRecordCreate is not None
    assert FishingVesselCreate is not None
    assert FishingTripCreate is not None


def test_password_hashing_functionality():
    """Test basic password hashing functionality."""
    from app.core.security import get_password_hash, verify_password
    
    password = "test_password_123"
    hashed = get_password_hash(password)
    
    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("wrong_password", hashed) is False


def test_exception_creation():
    """Test that custom exceptions can be created and raised."""
    from app.core.exceptions import (
        ValidationError,
        DatabaseError,
        AuthenticationError
    )
    
    # Test ValidationError
    try:
        raise ValidationError("Test validation error", field="test_field")
    except ValidationError as e:
        assert e.message == "Test validation error"
        assert e.field == "test_field"
    
    # Test DatabaseError
    try:
        raise DatabaseError("Test database error", operation="test_operation")
    except DatabaseError as e:
        assert e.message == "Test database error"
        assert e.operation == "test_operation"
    
    # Test AuthenticationError
    try:
        raise AuthenticationError("Test auth error")
    except AuthenticationError as e:
        assert e.message == "Test auth error"


def test_basic_platform_health():
    """Test basic platform health without full application setup."""
    # Import core modules
    import app.core.config
    import app.core.database
    import app.core.security
    import app.core.exceptions
    
    # Import models
    import app.models.user
    import app.models.taxonomy
    import app.models.fisheries
    import app.models.otolith
    
    # Import schemas
    import app.schemas.user
    import app.schemas.fisheries
    import app.schemas.otolith
    
    # All imports successful means basic platform structure is intact
    assert True, "All critical modules imported successfully"