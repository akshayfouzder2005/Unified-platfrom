from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field

# Base schemas
class TaxonomicRankBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    level: int = Field(..., gt=0, description="Hierarchical level (1=Kingdom, 2=Phylum, etc.)")
    description: Optional[str] = None

class TaxonomicRankCreate(TaxonomicRankBase):
    pass

class TaxonomicRankUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=50)
    level: Optional[int] = Field(None, gt=0)
    description: Optional[str] = None

class TaxonomicRankResponse(TaxonomicRankBase):
    id: int
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str]
    
    class Config:
        from_attributes = True

# Taxonomic Unit schemas
class TaxonomicUnitBase(BaseModel):
    scientific_name: str = Field(..., min_length=1, max_length=200)
    common_name: Optional[str] = Field(None, max_length=200)
    author: Optional[str] = Field(None, max_length=200)
    year_described: Optional[int] = Field(None, ge=1758, le=2100)  # Linnean system started 1758
    rank_id: int
    parent_id: Optional[int] = None
    is_valid: bool = True
    is_marine: bool = True
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    description: Optional[str] = None
    habitat_notes: Optional[str] = None
    distribution_notes: Optional[str] = None
    conservation_status: Optional[str] = Field(None, max_length=50)
    worms_id: Optional[str] = Field(None, max_length=50)
    ncbi_taxid: Optional[str] = Field(None, max_length=50)
    gbif_id: Optional[str] = Field(None, max_length=50)

class TaxonomicUnitCreate(TaxonomicUnitBase):
    pass

class TaxonomicUnitUpdate(BaseModel):
    scientific_name: Optional[str] = Field(None, min_length=1, max_length=200)
    common_name: Optional[str] = Field(None, max_length=200)
    author: Optional[str] = Field(None, max_length=200)
    year_described: Optional[int] = Field(None, ge=1758, le=2100)
    rank_id: Optional[int] = None
    parent_id: Optional[int] = None
    is_valid: Optional[bool] = None
    is_marine: Optional[bool] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    description: Optional[str] = None
    habitat_notes: Optional[str] = None
    distribution_notes: Optional[str] = None
    conservation_status: Optional[str] = Field(None, max_length=50)
    worms_id: Optional[str] = Field(None, max_length=50)
    ncbi_taxid: Optional[str] = Field(None, max_length=50)
    gbif_id: Optional[str] = Field(None, max_length=50)

class TaxonomicUnitResponse(TaxonomicUnitBase):
    id: int
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str]
    rank: Optional[TaxonomicRankResponse] = None
    
    class Config:
        from_attributes = True

# Synonym schemas
class TaxonomicSynonymBase(BaseModel):
    valid_taxon_id: int
    synonym_name: str = Field(..., min_length=1, max_length=200)
    synonym_author: Optional[str] = Field(None, max_length=200)
    synonym_year: Optional[int] = Field(None, ge=1758, le=2100)
    synonym_type: Optional[str] = Field(None, max_length=50)

class TaxonomicSynonymCreate(TaxonomicSynonymBase):
    pass

class TaxonomicSynonymUpdate(BaseModel):
    valid_taxon_id: Optional[int] = None
    synonym_name: Optional[str] = Field(None, min_length=1, max_length=200)
    synonym_author: Optional[str] = Field(None, max_length=200)
    synonym_year: Optional[int] = Field(None, ge=1758, le=2100)
    synonym_type: Optional[str] = Field(None, max_length=50)

class TaxonomicSynonymResponse(TaxonomicSynonymBase):
    id: int
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str]
    valid_taxon: Optional[TaxonomicUnitResponse] = None
    
    class Config:
        from_attributes = True

# Search and filter schemas
class TaxonomySearchParams(BaseModel):
    q: Optional[str] = None
    rank: Optional[str] = None
    is_valid: Optional[bool] = None
    is_marine: Optional[bool] = None
    parent_id: Optional[int] = None
    limit: int = Field(100, le=1000)
    offset: int = Field(0, ge=0)

class TaxonomySearchResponse(BaseModel):
    items: List[TaxonomicUnitResponse]
    total: int
    limit: int
    offset: int
    has_more: bool