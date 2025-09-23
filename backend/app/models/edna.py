from sqlalchemy import Column, String, Float, DateTime, Integer, Text, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from .base import BaseModel

class EdnaSample(BaseModel):
    """eDNA sample collection information"""
    __tablename__ = "edna_samples"
    
    sample_id = Column(String(50), nullable=False, unique=True, index=True)
    sample_name = Column(String(200), nullable=True)
    
    # Collection details
    collection_date = Column(DateTime, nullable=False, index=True)
    collected_by = Column(String(100), nullable=True)
    collection_method = Column(String(100), nullable=True)
    
    # Location
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    depth = Column(Float, nullable=True)
    
    # Environmental context
    water_temperature = Column(Float, nullable=True)
    salinity = Column(Float, nullable=True)
    ph = Column(Float, nullable=True)
    dissolved_oxygen = Column(Float, nullable=True)
    turbidity = Column(Float, nullable=True)
    
    # Sample characteristics
    volume_filtered = Column(Float, nullable=True)  # Liters
    filter_type = Column(String(50), nullable=True)
    filter_pore_size = Column(Float, nullable=True)  # Micrometers
    preservation_method = Column(String(100), nullable=True)
    storage_temperature = Column(Float, nullable=True)
    
    # Geographic/administrative
    country = Column(String(100), nullable=True)
    region = Column(String(100), nullable=True)
    water_body = Column(String(200), nullable=True)
    habitat_type = Column(String(100), nullable=True)
    
    # Quality and status
    sample_quality = Column(String(20), nullable=True)  # "excellent", "good", "fair", "poor"
    processing_status = Column(String(50), default="collected")  # "collected", "extracted", "sequenced", "analyzed"
    
    # Metadata
    notes = Column(Text, nullable=True)
    project_name = Column(String(200), nullable=True)
    
    # Relationships
    extractions = relationship("EdnaExtraction", back_populates="sample")
    detections = relationship("EdnaDetection", back_populates="sample")

class EdnaExtraction(BaseModel):
    """DNA extraction from eDNA samples"""
    __tablename__ = "edna_extractions"
    
    sample_id = Column(Integer, ForeignKey("edna_samples.id"), nullable=False, index=True)
    extraction_id = Column(String(50), nullable=False, unique=True)
    
    # Extraction details
    extraction_date = Column(DateTime, nullable=False)
    extraction_method = Column(String(100), nullable=False)
    extraction_kit = Column(String(100), nullable=True)
    extracted_by = Column(String(100), nullable=True)
    
    # Quality metrics
    dna_concentration = Column(Float, nullable=True)  # ng/μL
    dna_purity_260_280 = Column(Float, nullable=True)
    dna_purity_260_230 = Column(Float, nullable=True)
    extraction_volume = Column(Float, nullable=True)  # μL
    
    # Storage
    storage_location = Column(String(100), nullable=True)
    storage_temperature = Column(Float, nullable=True)
    
    notes = Column(Text, nullable=True)
    
    # Relationships
    sample = relationship("EdnaSample", back_populates="extractions")
    pcr_reactions = relationship("PcrReaction", back_populates="extraction")

class PcrReaction(BaseModel):
    """PCR amplification reactions"""
    __tablename__ = "pcr_reactions"
    
    extraction_id = Column(Integer, ForeignKey("edna_extractions.id"), nullable=False, index=True)
    reaction_id = Column(String(50), nullable=False, unique=True)
    
    # PCR details
    pcr_date = Column(DateTime, nullable=False)
    target_gene = Column(String(50), nullable=False, index=True)  # "COI", "12S", "16S", "18S"
    primer_set = Column(String(100), nullable=False)
    forward_primer = Column(String(100), nullable=True)
    reverse_primer = Column(String(100), nullable=True)
    
    # PCR conditions
    annealing_temperature = Column(Float, nullable=True)
    cycles = Column(Integer, nullable=True)
    reaction_volume = Column(Float, nullable=True)
    template_volume = Column(Float, nullable=True)
    
    # Quality control
    positive_control = Column(Boolean, default=False)
    negative_control = Column(Boolean, default=False)
    pcr_success = Column(Boolean, nullable=True)
    band_intensity = Column(String(20), nullable=True)  # "strong", "medium", "weak", "none"
    
    notes = Column(Text, nullable=True)
    
    # Relationships
    extraction = relationship("EdnaExtraction", back_populates="pcr_reactions")
    sequences = relationship("DnaSequence", back_populates="pcr_reaction")

class DnaSequence(BaseModel):
    """DNA sequences obtained from PCR products"""
    __tablename__ = "dna_sequences"
    
    pcr_reaction_id = Column(Integer, ForeignKey("pcr_reactions.id"), nullable=False, index=True)
    sequence_id = Column(String(50), nullable=False, unique=True)
    
    # Sequencing details
    sequencing_date = Column(DateTime, nullable=True)
    sequencing_platform = Column(String(50), nullable=True)  # "Illumina", "Oxford Nanopore", "Sanger"
    sequencing_method = Column(String(100), nullable=True)
    
    # Sequence data
    raw_sequence = Column(Text, nullable=False)
    quality_scores = Column(Text, nullable=True)
    sequence_length = Column(Integer, nullable=True)
    
    # Quality metrics
    quality_score = Column(Float, nullable=True)
    gc_content = Column(Float, nullable=True)
    n_content = Column(Float, nullable=True)  # Percentage of N bases
    
    # Processing status
    is_quality_filtered = Column(Boolean, default=False)
    is_chimera_checked = Column(Boolean, default=False)
    is_taxonomically_assigned = Column(Boolean, default=False)
    
    notes = Column(Text, nullable=True)
    
    # Relationships
    pcr_reaction = relationship("PcrReaction", back_populates="sequences")
    taxonomic_assignments = relationship("TaxonomicAssignment", back_populates="sequence")

class TaxonomicAssignment(BaseModel):
    """Taxonomic assignment of DNA sequences"""
    __tablename__ = "taxonomic_assignments"
    
    sequence_id = Column(Integer, ForeignKey("dna_sequences.id"), nullable=False, index=True)
    taxon_id = Column(Integer, ForeignKey("taxonomic_units.id"), nullable=True)  # Links to taxonomy model
    
    # Assignment details
    assignment_method = Column(String(50), nullable=False)  # "BLAST", "SINTAX", "RDP", "custom"
    database_used = Column(String(100), nullable=False)  # "NCBI", "SILVA", "BOLD", "custom"
    database_version = Column(String(50), nullable=True)
    assignment_date = Column(DateTime, nullable=False)
    
    # Confidence metrics
    confidence_score = Column(Float, nullable=False)  # 0-1
    identity_percentage = Column(Float, nullable=True)  # Sequence similarity %
    coverage_percentage = Column(Float, nullable=True)  # Query coverage %
    e_value = Column(Float, nullable=True)
    bit_score = Column(Float, nullable=True)
    
    # Taxonomic hierarchy from assignment
    kingdom = Column(String(100), nullable=True)
    phylum = Column(String(100), nullable=True)
    class_name = Column(String(100), nullable=True)  # 'class' is reserved
    order_name = Column(String(100), nullable=True)  # 'order' is reserved
    family = Column(String(100), nullable=True)
    genus = Column(String(100), nullable=True)
    species = Column(String(200), nullable=True)
    
    # Assignment metadata
    is_verified = Column(Boolean, default=False)
    verified_by = Column(String(100), nullable=True)
    verification_notes = Column(Text, nullable=True)
    
    # Relationships
    sequence = relationship("DnaSequence", back_populates="taxonomic_assignments")

class EdnaDetection(BaseModel):
    """Species detection events from eDNA samples"""
    __tablename__ = "edna_detections"
    
    sample_id = Column(Integer, ForeignKey("edna_samples.id"), nullable=False, index=True)
    taxon_id = Column(Integer, ForeignKey("taxonomic_units.id"), nullable=False, index=True)
    
    # Detection details
    detection_method = Column(String(50), nullable=False)  # "metabarcoding", "qPCR", "ddPCR"
    target_gene = Column(String(50), nullable=False)
    
    # Quantification
    read_count = Column(Integer, nullable=True)  # For metabarcoding
    relative_abundance = Column(Float, nullable=True)  # 0-1
    ct_value = Column(Float, nullable=True)  # For qPCR
    concentration = Column(Float, nullable=True)  # Estimated concentration
    
    # Confidence and validation
    detection_confidence = Column(Float, nullable=False)  # 0-1
    is_validated = Column(Boolean, default=False)
    validation_method = Column(String(100), nullable=True)
    false_positive_risk = Column(String(20), nullable=True)  # "low", "medium", "high"
    
    # Environmental context
    detection_context = Column(Text, nullable=True)
    habitat_suitability = Column(Float, nullable=True)  # 0-1
    season = Column(String(20), nullable=True)
    
    # Metadata
    notes = Column(Text, nullable=True)
    detected_by = Column(String(100), nullable=True)
    detection_date = Column(DateTime, nullable=False)
    
    # Relationships
    sample = relationship("EdnaSample", back_populates="detections")

class EdnaStudy(BaseModel):
    """eDNA study/project metadata"""
    __tablename__ = "edna_studies"
    
    study_name = Column(String(200), nullable=False)
    study_code = Column(String(50), nullable=True, unique=True)
    description = Column(Text, nullable=True)
    
    # Study scope
    objective = Column(Text, nullable=True)
    target_taxa = Column(Text, nullable=True)
    study_area = Column(String(200), nullable=True)
    
    # Temporal scope
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    
    # Personnel
    principal_investigator = Column(String(100), nullable=True)
    organization = Column(String(200), nullable=True)
    collaborators = Column(Text, nullable=True)
    
    # Publication info
    publication_status = Column(String(50), nullable=True)  # "draft", "submitted", "published"
    doi = Column(String(100), nullable=True)
    citation = Column(Text, nullable=True)
    
    # Data sharing
    data_availability = Column(String(50), nullable=True)  # "public", "restricted", "private"
    license = Column(String(100), nullable=True)
    
    notes = Column(Text, nullable=True)