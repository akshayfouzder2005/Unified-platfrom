from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import logging

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Base validation error class"""
    def __init__(self, message: str, field: str = None):
        self.message = message
        self.field = field
        super().__init__(self.message)

class DataValidationError(ValidationError):
    """Custom exception for data validation errors"""
    def __init__(self, message: str, field: str = None, value: any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message, field)

class DatabaseConnectionError(Exception):
    """Custom exception for database connection issues"""
    pass


class DatabaseError(Exception):
    """Custom exception for general database errors"""
    def __init__(self, message: str, operation: str = None):
        self.message = message
        self.operation = operation
        super().__init__(self.message)


class AuthenticationError(Exception):
    """Custom exception for authentication errors"""
    def __init__(self, message: str = "Authentication failed"):
        self.message = message
        super().__init__(self.message)


class AuthorizationError(Exception):
    """Custom exception for authorization errors"""
    def __init__(self, message: str = "Access denied"):
        self.message = message
        super().__init__(self.message)

class IngestionError(Exception):
    """Custom exception for data ingestion failures"""
    def __init__(self, message: str, row: int = None, errors: list = None):
        self.message = message
        self.row = row
        self.errors = errors or []
        super().__init__(self.message)

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle FastAPI validation errors with detailed information"""
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(x) for x in error["loc"])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"],
            "input": error.get("input")
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "The provided data did not pass validation",
            "details": errors,
            "timestamp": request.url.path
        }
    )

async def data_validation_exception_handler(request: Request, exc: DataValidationError):
    """Handle custom data validation errors"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Data Validation Error",
            "message": exc.message,
            "field": exc.field,
            "value": exc.value,
            "timestamp": request.url.path
        }
    )

async def database_exception_handler(request: Request, exc: SQLAlchemyError):
    """Handle database errors"""
    logger.error(f"Database error: {str(exc)}")
    
    if isinstance(exc, IntegrityError):
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "error": "Database Integrity Error",
                "message": "The operation violates a database constraint",
                "details": str(exc.orig) if hasattr(exc, 'orig') else str(exc),
                "timestamp": request.url.path
            }
        )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Database Error",
            "message": "An error occurred while accessing the database",
            "timestamp": request.url.path
        }
    )

async def database_connection_exception_handler(request: Request, exc: DatabaseConnectionError):
    """Handle database connection errors"""
    logger.error(f"Database connection error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "Database Connection Error",
            "message": "Cannot connect to the database. Please try again later.",
            "timestamp": request.url.path
        }
    )

async def ingestion_exception_handler(request: Request, exc: IngestionError):
    """Handle data ingestion errors"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Data Ingestion Error",
            "message": exc.message,
            "row": exc.row,
            "errors": exc.errors,
            "timestamp": request.url.path
        }
    )

async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": request.url.path
        }
    )