from fastapi import APIRouter

router = APIRouter()


@router.get("/morphometrics")
def list_morphometrics():
    """List otolith morphology metrics (placeholder)."""
    return {"items": []}
