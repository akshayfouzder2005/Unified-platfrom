from fastapi import APIRouter

router = APIRouter()


@router.get("/samples")
def list_edna_samples():
    """List eDNA samples (placeholder)."""
    return {"items": []}
