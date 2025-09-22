from fastapi import FastAPI
from .core.config import settings
from .modules.taxonomy.router import router as taxonomy_router
from .modules.otolith.router import router as otolith_router
from .modules.edna.router import router as edna_router
from .visualization.router import router as viz_router

app = FastAPI(
    title="AI-Driven Unified Data Platform",
    version="0.1.0",
    description="APIs for oceanographic, fisheries, and molecular biodiversity data",
)


@app.get("/api/health")
def health():
    return {"status": "ok", "app": settings.app_name, "env": settings.environment}


# Register module routers
app.include_router(taxonomy_router, prefix="/api/taxonomy", tags=["taxonomy"])
app.include_router(otolith_router, prefix="/api/otolith", tags=["otolith"])
app.include_router(edna_router, prefix="/api/edna", tags=["edna"])
app.include_router(viz_router, prefix="/api/visualization", tags=["visualization"])
