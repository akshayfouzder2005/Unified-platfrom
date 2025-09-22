from fastapi import APIRouter

router = APIRouter()


@router.get("/trends/ocean")
def ocean_trends():
    """Return placeholder trend data for oceanographic variables."""
    return {
        "variable": "SST",
        "units": "C",
        "series": [
            {"t": "2025-01-01", "value": 20.1},
            {"t": "2025-02-01", "value": 20.4},
            {"t": "2025-03-01", "value": 20.2},
        ],
    }
