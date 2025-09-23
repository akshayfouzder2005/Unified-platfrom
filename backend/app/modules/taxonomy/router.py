from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.orm import Session
from typing import Optional, List
from ...core.database import get_db
from ...crud import taxonomy as crud
from ...schemas.taxonomy import (
    TaxonomicRankCreate, TaxonomicRankUpdate, TaxonomicRankResponse,
    TaxonomicUnitCreate, TaxonomicUnitUpdate, TaxonomicUnitResponse,
    TaxonomicSynonymCreate, TaxonomicSynonymUpdate, TaxonomicSynonymResponse,
    TaxonomySearchResponse
)

router = APIRouter()

# Taxonomic Rank endpoints
@router.post("/ranks", response_model=TaxonomicRankResponse, status_code=201)
def create_rank(
    rank_data: TaxonomicRankCreate,
    db: Session = Depends(get_db)
):
    """Create a new taxonomic rank."""
    # Check if rank already exists
    existing = crud.get_taxonomic_rank_by_name(db, rank_data.name)
    if existing:
        raise HTTPException(status_code=400, detail="Taxonomic rank with this name already exists")
    
    return crud.create_taxonomic_rank(db, rank_data)

@router.get("/ranks", response_model=List[TaxonomicRankResponse])
def list_ranks(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, le=1000),
    db: Session = Depends(get_db)
):
    """Get all taxonomic ranks."""
    return crud.get_taxonomic_ranks(db, skip=skip, limit=limit)

@router.get("/ranks/{rank_id}", response_model=TaxonomicRankResponse)
def get_rank(
    rank_id: int = Path(..., gt=0),
    db: Session = Depends(get_db)
):
    """Get a specific taxonomic rank."""
    rank = crud.get_taxonomic_rank(db, rank_id)
    if not rank:
        raise HTTPException(status_code=404, detail="Taxonomic rank not found")
    return rank

@router.put("/ranks/{rank_id}", response_model=TaxonomicRankResponse)
def update_rank(
    rank_id: int = Path(..., gt=0),
    rank_data: TaxonomicRankUpdate = ...,
    db: Session = Depends(get_db)
):
    """Update a taxonomic rank."""
    updated_rank = crud.update_taxonomic_rank(db, rank_id, rank_data)
    if not updated_rank:
        raise HTTPException(status_code=404, detail="Taxonomic rank not found")
    return updated_rank

@router.delete("/ranks/{rank_id}", status_code=204)
def delete_rank(
    rank_id: int = Path(..., gt=0),
    db: Session = Depends(get_db)
):
    """Delete a taxonomic rank."""
    success = crud.delete_taxonomic_rank(db, rank_id)
    if not success:
        raise HTTPException(status_code=404, detail="Taxonomic rank not found")

# Taxonomic Unit (Species/Taxa) endpoints
@router.post("/species", response_model=TaxonomicUnitResponse, status_code=201)
def create_species(
    species_data: TaxonomicUnitCreate,
    db: Session = Depends(get_db)
):
    """Create a new taxonomic unit (species/taxon)."""
    # Check if species already exists
    existing = crud.get_taxonomic_unit_by_name(db, species_data.scientific_name)
    if existing:
        raise HTTPException(status_code=400, detail="Species with this scientific name already exists")
    
    # Validate rank exists
    rank = crud.get_taxonomic_rank(db, species_data.rank_id)
    if not rank:
        raise HTTPException(status_code=400, detail="Invalid rank_id")
    
    # Validate parent exists if provided
    if species_data.parent_id:
        parent = crud.get_taxonomic_unit(db, species_data.parent_id)
        if not parent:
            raise HTTPException(status_code=400, detail="Invalid parent_id")
    
    return crud.create_taxonomic_unit(db, species_data)

@router.get("/species", response_model=TaxonomySearchResponse)
def search_species(
    q: Optional[str] = Query(None, description="Search query for scientific or common names"),
    rank: Optional[str] = Query(None, description="Filter by taxonomic rank name"),
    is_valid: Optional[bool] = Query(None, description="Filter by validity status"),
    is_marine: Optional[bool] = Query(None, description="Filter by marine habitat"),
    parent_id: Optional[int] = Query(None, description="Filter by parent taxon ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, le=1000),
    db: Session = Depends(get_db)
):
    """Search taxonomic units with various filters."""
    species, total = crud.search_taxonomic_units(
        db,
        query=q,
        rank_name=rank,
        is_valid=is_valid,
        is_marine=is_marine,
        parent_id=parent_id,
        skip=skip,
        limit=limit
    )
    
    return {
        "items": species,
        "total": total,
        "limit": limit,
        "offset": skip,
        "has_more": skip + limit < total
    }

@router.get("/species/{species_id}", response_model=TaxonomicUnitResponse)
def get_species(
    species_id: int = Path(..., gt=0),
    db: Session = Depends(get_db)
):
    """Get a specific taxonomic unit."""
    species = crud.get_taxonomic_unit(db, species_id)
    if not species:
        raise HTTPException(status_code=404, detail="Species not found")
    return species

@router.put("/species/{species_id}", response_model=TaxonomicUnitResponse)
def update_species(
    species_id: int = Path(..., gt=0),
    species_data: TaxonomicUnitUpdate = ...,
    db: Session = Depends(get_db)
):
    """Update a taxonomic unit."""
    # Validate rank exists if provided
    if species_data.rank_id:
        rank = crud.get_taxonomic_rank(db, species_data.rank_id)
        if not rank:
            raise HTTPException(status_code=400, detail="Invalid rank_id")
    
    # Validate parent exists if provided
    if species_data.parent_id:
        parent = crud.get_taxonomic_unit(db, species_data.parent_id)
        if not parent:
            raise HTTPException(status_code=400, detail="Invalid parent_id")
    
    updated_species = crud.update_taxonomic_unit(db, species_id, species_data)
    if not updated_species:
        raise HTTPException(status_code=404, detail="Species not found")
    return updated_species

@router.delete("/species/{species_id}", status_code=204)
def delete_species(
    species_id: int = Path(..., gt=0),
    cascade: bool = Query(False, description="Delete children as well"),
    db: Session = Depends(get_db)
):
    """Delete a taxonomic unit."""
    try:
        success = crud.delete_taxonomic_unit(db, species_id, cascade=cascade)
        if not success:
            raise HTTPException(status_code=404, detail="Species not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/species/{species_id}/children", response_model=List[TaxonomicUnitResponse])
def get_species_children(
    species_id: int = Path(..., gt=0),
    db: Session = Depends(get_db)
):
    """Get direct children of a taxonomic unit."""
    # Verify parent exists
    parent = crud.get_taxonomic_unit(db, species_id)
    if not parent:
        raise HTTPException(status_code=404, detail="Parent species not found")
    
    return crud.get_taxonomic_children(db, species_id)

@router.get("/taxonomy-tree")
def get_taxonomy_tree(
    root_id: Optional[int] = Query(None, description="Root taxonomic unit ID"),
    db: Session = Depends(get_db)
):
    """Get taxonomic tree structure."""
    return crud.get_taxonomic_tree(db, root_id)

# Synonyms endpoints
@router.post("/species/{species_id}/synonyms", response_model=TaxonomicSynonymResponse, status_code=201)
def create_synonym(
    species_id: int = Path(..., gt=0),
    synonym_data: TaxonomicSynonymCreate = ...,
    db: Session = Depends(get_db)
):
    """Add a synonym to a taxonomic unit."""
    # Verify species exists
    species = crud.get_taxonomic_unit(db, species_id)
    if not species:
        raise HTTPException(status_code=404, detail="Species not found")
    
    # Override the valid_taxon_id with the path parameter
    synonym_data.valid_taxon_id = species_id
    return crud.create_taxonomic_synonym(db, synonym_data)

@router.get("/species/{species_id}/synonyms", response_model=List[TaxonomicSynonymResponse])
def get_species_synonyms(
    species_id: int = Path(..., gt=0),
    db: Session = Depends(get_db)
):
    """Get all synonyms for a taxonomic unit."""
    # Verify species exists
    species = crud.get_taxonomic_unit(db, species_id)
    if not species:
        raise HTTPException(status_code=404, detail="Species not found")
    
    return crud.get_taxonomic_synonyms(db, species_id)

@router.delete("/synonyms/{synonym_id}", status_code=204)
def delete_synonym(
    synonym_id: int = Path(..., gt=0),
    db: Session = Depends(get_db)
):
    """Delete a taxonomic synonym."""
    success = crud.delete_taxonomic_synonym(db, synonym_id)
    if not success:
        raise HTTPException(status_code=404, detail="Synonym not found")

# Utility endpoints
@router.get("/stats")
def get_taxonomy_stats(db: Session = Depends(get_db)):
    """Get taxonomy database statistics."""
    # This would be implemented with actual database queries
    return {
        "total_species": 0,
        "total_ranks": 0,
        "total_synonyms": 0,
        "marine_species": 0,
        "valid_species": 0,
        "message": "Database connection needed for real stats"
    }