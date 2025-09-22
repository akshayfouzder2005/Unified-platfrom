from fastapi import APIRouter
from typing import Optional

router = APIRouter()


@router.get("/species")
def list_species(q: Optional[str] = None):
    """List species with optional search query.
    TODO: Integrate taxonomy database/models.
    """
    return {"items": [], "query": q}


@router.get("/taxonomy-tree")
def taxonomy_tree():
    """Return a placeholder taxonomy tree structure."""
    return {"root": "Life", "children": []}
