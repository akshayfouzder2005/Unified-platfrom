from sqlalchemy import Column, String, Integer, Text, ForeignKey, Boolean, Float
from sqlalchemy.orm import relationship
from .base import BaseModel

class TaxonomicRank(BaseModel):
    """Taxonomic rank definitions (Kingdom, Phylum, Class, etc.)"""
    __tablename__ = "taxonomic_ranks"
    
    name = Column(String(50), nullable=False, unique=True)  # e.g., "Kingdom", "Phylum"
    level = Column(Integer, nullable=False)  # Hierarchical level (1=Kingdom, 2=Phylum, etc.)
    description = Column(Text, nullable=True)

class TaxonomicUnit(BaseModel):
    """Individual taxonomic units (species, genera, families, etc.)"""
    __tablename__ = "taxonomic_units"
    
    scientific_name = Column(String(200), nullable=False, index=True)
    common_name = Column(String(200), nullable=True)
    author = Column(String(200), nullable=True)  # Taxonomic authority
    year_described = Column(Integer, nullable=True)
    
    # Taxonomic hierarchy
    rank_id = Column(Integer, ForeignKey("taxonomic_ranks.id"), nullable=False)
    parent_id = Column(Integer, ForeignKey("taxonomic_units.id"), nullable=True)
    
    # Status and validation
    is_valid = Column(Boolean, default=True)
    is_marine = Column(Boolean, default=True)
    confidence_score = Column(Float, nullable=True)  # For AI-assisted classifications
    
    # Additional metadata
    description = Column(Text, nullable=True)
    habitat_notes = Column(Text, nullable=True)
    distribution_notes = Column(Text, nullable=True)
    conservation_status = Column(String(50), nullable=True)
    
    # External references
    worms_id = Column(String(50), nullable=True)  # World Register of Marine Species ID
    ncbi_taxid = Column(String(50), nullable=True)  # NCBI Taxonomy ID
    gbif_id = Column(String(50), nullable=True)  # GBIF species ID
    
    # Relationships
    rank = relationship("TaxonomicRank", back_populates="taxonomic_units")
    parent = relationship("TaxonomicUnit", remote_side=[BaseModel.id], back_populates="children")
    children = relationship("TaxonomicUnit", back_populates="parent")
    
    def __str__(self):
        return f"{self.scientific_name} ({self.rank.name if self.rank else 'Unknown'})"

# Add back reference to TaxonomicRank
TaxonomicRank.taxonomic_units = relationship("TaxonomicUnit", back_populates="rank")

class TaxonomicSynonym(BaseModel):
    """Taxonomic synonyms and alternative names"""
    __tablename__ = "taxonomic_synonyms"
    
    valid_taxon_id = Column(Integer, ForeignKey("taxonomic_units.id"), nullable=False)
    synonym_name = Column(String(200), nullable=False, index=True)
    synonym_author = Column(String(200), nullable=True)
    synonym_year = Column(Integer, nullable=True)
    synonym_type = Column(String(50), nullable=True)  # "junior synonym", "homonym", etc.
    
    # Relationship
    valid_taxon = relationship("TaxonomicUnit")

class TaxonomicReference(BaseModel):
    """Scientific references for taxonomic information"""
    __tablename__ = "taxonomic_references"
    
    taxon_id = Column(Integer, ForeignKey("taxonomic_units.id"), nullable=False)
    reference_type = Column(String(50), nullable=False)  # "original_description", "revision", etc.
    title = Column(Text, nullable=False)
    authors = Column(Text, nullable=False)
    journal = Column(String(200), nullable=True)
    year = Column(Integer, nullable=True)
    volume = Column(String(20), nullable=True)
    pages = Column(String(50), nullable=True)
    doi = Column(String(100), nullable=True)
    url = Column(Text, nullable=True)
    
    # Relationship
    taxon = relationship("TaxonomicUnit")