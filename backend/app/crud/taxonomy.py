from typing import List, Optional
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, and_, or_
from ..models.taxonomy import TaxonomicRank, TaxonomicUnit, TaxonomicSynonym
from ..schemas.taxonomy import (
    TaxonomicRankCreate, TaxonomicRankUpdate,
    TaxonomicUnitCreate, TaxonomicUnitUpdate,
    TaxonomicSynonymCreate, TaxonomicSynonymUpdate
)

# Taxonomic Rank CRUD
def create_taxonomic_rank(db: Session, rank_data: TaxonomicRankCreate, created_by: Optional[str] = None) -> TaxonomicRank:
    """Create a new taxonomic rank."""
    db_rank = TaxonomicRank(
        **rank_data.model_dump(),
        created_by=created_by
    )
    db.add(db_rank)
    db.commit()
    db.refresh(db_rank)
    return db_rank

def get_taxonomic_rank(db: Session, rank_id: int) -> Optional[TaxonomicRank]:
    """Get a taxonomic rank by ID."""
    return db.query(TaxonomicRank).filter(TaxonomicRank.id == rank_id).first()

def get_taxonomic_rank_by_name(db: Session, name: str) -> Optional[TaxonomicRank]:
    """Get a taxonomic rank by name."""
    return db.query(TaxonomicRank).filter(TaxonomicRank.name == name).first()

def get_taxonomic_ranks(db: Session, skip: int = 0, limit: int = 100) -> List[TaxonomicRank]:
    """Get all taxonomic ranks with pagination."""
    return db.query(TaxonomicRank).order_by(TaxonomicRank.level).offset(skip).limit(limit).all()

def update_taxonomic_rank(db: Session, rank_id: int, rank_data: TaxonomicRankUpdate) -> Optional[TaxonomicRank]:
    """Update a taxonomic rank."""
    db_rank = get_taxonomic_rank(db, rank_id)
    if not db_rank:
        return None
    
    update_data = rank_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_rank, field, value)
    
    db.commit()
    db.refresh(db_rank)
    return db_rank

def delete_taxonomic_rank(db: Session, rank_id: int) -> bool:
    """Delete a taxonomic rank."""
    db_rank = get_taxonomic_rank(db, rank_id)
    if not db_rank:
        return False
    
    db.delete(db_rank)
    db.commit()
    return True

# Taxonomic Unit CRUD
def create_taxonomic_unit(db: Session, unit_data: TaxonomicUnitCreate, created_by: Optional[str] = None) -> TaxonomicUnit:
    """Create a new taxonomic unit."""
    db_unit = TaxonomicUnit(
        **unit_data.model_dump(),
        created_by=created_by
    )
    db.add(db_unit)
    db.commit()
    db.refresh(db_unit)
    return db_unit

def get_taxonomic_unit(db: Session, unit_id: int, include_rank: bool = True) -> Optional[TaxonomicUnit]:
    """Get a taxonomic unit by ID."""
    query = db.query(TaxonomicUnit)
    if include_rank:
        query = query.options(joinedload(TaxonomicUnit.rank))
    return query.filter(TaxonomicUnit.id == unit_id).first()

def get_taxonomic_unit_by_name(db: Session, scientific_name: str) -> Optional[TaxonomicUnit]:
    """Get a taxonomic unit by scientific name."""
    return db.query(TaxonomicUnit).filter(TaxonomicUnit.scientific_name == scientific_name).first()

def search_taxonomic_units(
    db: Session,
    query: Optional[str] = None,
    rank_name: Optional[str] = None,
    is_valid: Optional[bool] = None,
    is_marine: Optional[bool] = None,
    parent_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100
) -> tuple[List[TaxonomicUnit], int]:
    """Search taxonomic units with filters."""
    db_query = db.query(TaxonomicUnit).options(joinedload(TaxonomicUnit.rank))
    
    # Apply filters
    if query:
        search_filter = or_(
            TaxonomicUnit.scientific_name.ilike(f"%{query}%"),
            TaxonomicUnit.common_name.ilike(f"%{query}%")
        )
        db_query = db_query.filter(search_filter)
    
    if rank_name:
        db_query = db_query.join(TaxonomicRank).filter(TaxonomicRank.name == rank_name)
    
    if is_valid is not None:
        db_query = db_query.filter(TaxonomicUnit.is_valid == is_valid)
    
    if is_marine is not None:
        db_query = db_query.filter(TaxonomicUnit.is_marine == is_marine)
    
    if parent_id is not None:
        db_query = db_query.filter(TaxonomicUnit.parent_id == parent_id)
    
    # Get total count
    total = db_query.count()
    
    # Apply pagination
    results = db_query.offset(skip).limit(limit).all()
    
    return results, total

def get_taxonomic_children(db: Session, parent_id: int) -> List[TaxonomicUnit]:
    """Get direct children of a taxonomic unit."""
    return db.query(TaxonomicUnit).filter(TaxonomicUnit.parent_id == parent_id).all()

def get_taxonomic_tree(db: Session, root_id: Optional[int] = None, max_depth: int = 5) -> dict:
    """Get taxonomic tree structure starting from a root node."""
    # This is a simplified version - a full implementation would use recursive queries
    if root_id:
        root = get_taxonomic_unit(db, root_id)
        if not root:
            return {}
    else:
        # Get kingdoms (assuming level 1 ranks are kingdoms)
        kingdoms = db.query(TaxonomicUnit).join(TaxonomicRank).filter(TaxonomicRank.level == 1).all()
        return {
            "roots": [{"id": k.id, "name": k.scientific_name, "rank": k.rank.name} for k in kingdoms]
        }
    
    children = get_taxonomic_children(db, root_id)
    return {
        "id": root.id,
        "name": root.scientific_name,
        "rank": root.rank.name if root.rank else None,
        "children": [{"id": c.id, "name": c.scientific_name, "rank": c.rank.name if c.rank else None} for c in children]
    }

def update_taxonomic_unit(db: Session, unit_id: int, unit_data: TaxonomicUnitUpdate) -> Optional[TaxonomicUnit]:
    """Update a taxonomic unit."""
    db_unit = get_taxonomic_unit(db, unit_id)
    if not db_unit:
        return None
    
    update_data = unit_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_unit, field, value)
    
    db.commit()
    db.refresh(db_unit)
    return db_unit

def delete_taxonomic_unit(db: Session, unit_id: int, cascade: bool = False) -> bool:
    """Delete a taxonomic unit."""
    db_unit = get_taxonomic_unit(db, unit_id)
    if not db_unit:
        return False
    
    # Check if it has children
    children_count = db.query(TaxonomicUnit).filter(TaxonomicUnit.parent_id == unit_id).count()
    if children_count > 0 and not cascade:
        raise ValueError(f"Cannot delete taxonomic unit with {children_count} children. Use cascade=True to force deletion.")
    
    if cascade:
        # Delete children first (recursive deletion)
        children = get_taxonomic_children(db, unit_id)
        for child in children:
            delete_taxonomic_unit(db, child.id, cascade=True)
    
    db.delete(db_unit)
    db.commit()
    return True

# Taxonomic Synonym CRUD
def create_taxonomic_synonym(db: Session, synonym_data: TaxonomicSynonymCreate, created_by: Optional[str] = None) -> TaxonomicSynonym:
    """Create a new taxonomic synonym."""
    db_synonym = TaxonomicSynonym(
        **synonym_data.model_dump(),
        created_by=created_by
    )
    db.add(db_synonym)
    db.commit()
    db.refresh(db_synonym)
    return db_synonym

def get_taxonomic_synonyms(db: Session, valid_taxon_id: int) -> List[TaxonomicSynonym]:
    """Get all synonyms for a valid taxon."""
    return db.query(TaxonomicSynonym).filter(TaxonomicSynonym.valid_taxon_id == valid_taxon_id).all()

def delete_taxonomic_synonym(db: Session, synonym_id: int) -> bool:
    """Delete a taxonomic synonym."""
    db_synonym = db.query(TaxonomicSynonym).filter(TaxonomicSynonym.id == synonym_id).first()
    if not db_synonym:
        return False
    
    db.delete(db_synonym)
    db.commit()
    return True