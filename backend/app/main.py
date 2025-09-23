from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError
from .core.config import settings
from .core.exceptions import (
    DataValidationError, DatabaseConnectionError, IngestionError,
    validation_exception_handler, data_validation_exception_handler,
    database_exception_handler, database_connection_exception_handler,
    ingestion_exception_handler, general_exception_handler
)
from .modules.taxonomy.router import router as taxonomy_router
from .modules.otolith.router import router as otolith_router
from .modules.edna.router import router as edna_router
from .modules.ingestion.router import router as ingestion_router
from .visualization.router import router as viz_router

app = FastAPI(
    title="AI-Driven Unified Data Platform",
    version="0.1.0",
    description="APIs for oceanographic, fisheries, and molecular biodiversity data",
)

# Register exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(DataValidationError, data_validation_exception_handler)
app.add_exception_handler(SQLAlchemyError, database_exception_handler)
app.add_exception_handler(DatabaseConnectionError, database_connection_exception_handler)
app.add_exception_handler(IngestionError, ingestion_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)


@app.get("/api/health")
def health():
    return {"status": "ok", "app": settings.app_name, "env": settings.environment}


# Register module routers
app.include_router(taxonomy_router, prefix="/api/taxonomy", tags=["taxonomy"])
app.include_router(otolith_router, prefix="/api/otolith", tags=["otolith"])
app.include_router(edna_router, prefix="/api/edna", tags=["edna"])
app.include_router(ingestion_router, prefix="/api/ingestion", tags=["data-ingestion"])
app.include_router(viz_router, prefix="/api/visualization", tags=["visualization"])
