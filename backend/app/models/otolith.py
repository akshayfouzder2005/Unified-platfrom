from sqlalchemy import Column, String, Float, DateTime, Integer, Text, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from .base import BaseModel

class OtolithSpecimen(BaseModel):
    """Individual otolith specimens"""
    __tablename__ = "otolith_specimens"
    
    specimen_id = Column(String(50), nullable=False, unique=True, index=True)
    specimen_name = Column(String(200), nullable=True)
    
    # Fish information
    fish_species_id = Column(Integer, ForeignKey("taxonomic_units.id"), nullable=True)
    fish_length = Column(Float, nullable=True)  # Total length in mm
    fish_weight = Column(Float, nullable=True)  # Weight in grams
    fish_age = Column(Integer, nullable=True)  # Age in years
    fish_sex = Column(String(10), nullable=True)  # "male", "female", "unknown"
    
    # Collection details
    collection_date = Column(DateTime, nullable=True)
    collected_by = Column(String(100), nullable=True)
    collection_method = Column(String(100), nullable=True)
    
    # Location
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    depth = Column(Float, nullable=True)
    location_description = Column(Text, nullable=True)
    
    # Otolith details
    otolith_type = Column(String(20), nullable=False)  # "sagitta", "lapillus", "asteriscus"
    otolith_side = Column(String(10), nullable=True)  # "left", "right", "unknown"
    preservation_method = Column(String(100), nullable=True)
    storage_location = Column(String(200), nullable=True)
    
    # Quality and condition
    condition = Column(String(20), nullable=True)  # "excellent", "good", "fair", "poor", "damaged"
    completeness = Column(Float, nullable=True)  # 0-1, percentage complete
    
    # Metadata
    notes = Column(Text, nullable=True)
    project_name = Column(String(200), nullable=True)
    
    # Relationships
    measurements = relationship("OtolithMeasurement", back_populates="specimen")
    images = relationship("OtolithImage", back_populates="specimen")

class OtolithMeasurement(BaseModel):
    """Morphometric measurements of otoliths"""
    __tablename__ = "otolith_measurements"
    
    specimen_id = Column(Integer, ForeignKey("otolith_specimens.id"), nullable=False, index=True)
    measurement_type = Column(String(50), nullable=False, index=True)
    
    # Basic measurements
    length = Column(Float, nullable=True)  # Maximum length (mm)
    width = Column(Float, nullable=True)  # Maximum width (mm)
    height = Column(Float, nullable=True)  # Maximum height/thickness (mm)
    weight = Column(Float, nullable=True)  # Weight (mg)
    area = Column(Float, nullable=True)  # Surface area (mm²)
    perimeter = Column(Float, nullable=True)  # Perimeter (mm)
    
    # Shape indices
    roundness = Column(Float, nullable=True)  # Shape index
    rectangularity = Column(Float, nullable=True)
    ellipticity = Column(Float, nullable=True)
    form_factor = Column(Float, nullable=True)
    aspect_ratio = Column(Float, nullable=True)
    
    # Fourier descriptors (for detailed shape analysis)
    fourier_descriptors = Column(JSON, nullable=True)  # Array of Fourier coefficients
    
    # Edge and surface features
    edge_roughness = Column(Float, nullable=True)
    surface_texture = Column(Float, nullable=True)
    opacity_index = Column(Float, nullable=True)
    
    # Measurement metadata
    measurement_date = Column(DateTime, nullable=False)
    measured_by = Column(String(100), nullable=True)
    measurement_method = Column(String(100), nullable=True)  # "manual", "image_analysis", "3D_scan"
    measurement_precision = Column(Float, nullable=True)  # Measurement uncertainty
    
    # Equipment used
    equipment = Column(String(200), nullable=True)
    software = Column(String(100), nullable=True)
    calibration_factor = Column(Float, nullable=True)
    
    notes = Column(Text, nullable=True)
    
    # Relationships
    specimen = relationship("OtolithSpecimen", back_populates="measurements")

class OtolithImage(BaseModel):
    """Images of otolith specimens"""
    __tablename__ = "otolith_images"
    
    specimen_id = Column(Integer, ForeignKey("otolith_specimens.id"), nullable=False, index=True)
    image_filename = Column(String(500), nullable=False)
    image_path = Column(String(1000), nullable=False)
    
    # Image metadata
    image_type = Column(String(50), nullable=False)  # "dorsal", "ventral", "lateral", "3D", "microscopy"
    capture_date = Column(DateTime, nullable=True)
    captured_by = Column(String(100), nullable=True)
    
    # Technical details
    resolution_x = Column(Integer, nullable=True)  # pixels
    resolution_y = Column(Integer, nullable=True)  # pixels
    pixel_size = Column(Float, nullable=True)  # μm/pixel
    magnification = Column(Float, nullable=True)
    file_format = Column(String(20), nullable=True)  # "JPEG", "TIFF", "PNG"
    file_size = Column(Integer, nullable=True)  # bytes
    
    # Camera/equipment details
    camera_model = Column(String(100), nullable=True)
    lens_model = Column(String(100), nullable=True)
    lighting_conditions = Column(String(200), nullable=True)
    background_type = Column(String(50), nullable=True)
    
    # Processing information
    is_processed = Column(Boolean, default=False)
    processing_software = Column(String(100), nullable=True)
    processing_notes = Column(Text, nullable=True)
    
    # Quality assessment
    image_quality = Column(String(20), nullable=True)  # "excellent", "good", "fair", "poor"
    focus_quality = Column(String(20), nullable=True)
    lighting_quality = Column(String(20), nullable=True)
    contrast_rating = Column(Integer, nullable=True)  # 1-10 scale
    
    # Annotations and landmarks
    has_landmarks = Column(Boolean, default=False)
    landmark_data = Column(JSON, nullable=True)  # Coordinate data for anatomical landmarks
    annotations = Column(JSON, nullable=True)  # Annotations and measurements from image
    
    notes = Column(Text, nullable=True)
    
    # Relationships
    specimen = relationship("OtolithSpecimen", back_populates="images")

class OtolithReference(BaseModel):
    """Reference collection and comparative data"""
    __tablename__ = "otolith_references"
    
    reference_id = Column(String(50), nullable=False, unique=True)
    species_id = Column(Integer, ForeignKey("taxonomic_units.id"), nullable=False)
    
    # Reference details
    reference_type = Column(String(50), nullable=False)  # "voucher", "type_specimen", "literature"
    source_collection = Column(String(200), nullable=True)
    catalog_number = Column(String(100), nullable=True)
    
    # Publication information
    publication_title = Column(Text, nullable=True)
    authors = Column(Text, nullable=True)
    journal = Column(String(200), nullable=True)
    year = Column(Integer, nullable=True)
    doi = Column(String(100), nullable=True)
    
    # Geographic range
    geographic_range = Column(Text, nullable=True)
    depth_range_min = Column(Float, nullable=True)
    depth_range_max = Column(Float, nullable=True)
    
    # Morphological characteristics
    distinguishing_features = Column(Text, nullable=True)
    morphological_description = Column(Text, nullable=True)
    
    # Validation and confidence
    confidence_level = Column(String(20), nullable=True)  # "high", "medium", "low"
    verified_by = Column(String(100), nullable=True)
    verification_date = Column(DateTime, nullable=True)
    
    notes = Column(Text, nullable=True)

class OtolithClassification(BaseModel):
    """AI/automated classification results"""
    __tablename__ = "otolith_classifications"
    
    specimen_id = Column(Integer, ForeignKey("otolith_specimens.id"), nullable=False, index=True)
    predicted_species_id = Column(Integer, ForeignKey("taxonomic_units.id"), nullable=True)
    
    # Classification details
    classification_method = Column(String(100), nullable=False)  # "CNN", "SVM", "Random_Forest", "manual"
    model_version = Column(String(50), nullable=True)
    classification_date = Column(DateTime, nullable=False)
    
    # Confidence metrics
    confidence_score = Column(Float, nullable=False)  # 0-1
    top_predictions = Column(JSON, nullable=True)  # Array of top N predictions with scores
    
    # Feature importance
    key_features = Column(JSON, nullable=True)  # Features that contributed most to classification
    feature_weights = Column(JSON, nullable=True)  # Relative importance of features
    
    # Validation
    is_validated = Column(Boolean, default=False)
    validated_by = Column(String(100), nullable=True)
    validation_date = Column(DateTime, nullable=True)
    validation_notes = Column(Text, nullable=True)
    
    # Performance metrics (when validation is available)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    
    notes = Column(Text, nullable=True)
    
    # Relationships
    specimen = relationship("OtolithSpecimen")

class OtolithStudy(BaseModel):
    """Research studies involving otolith analysis"""
    __tablename__ = "otolith_studies"
    
    study_name = Column(String(200), nullable=False)
    study_code = Column(String(50), nullable=True, unique=True)
    description = Column(Text, nullable=True)
    
    # Study details
    objective = Column(Text, nullable=True)
    methodology = Column(Text, nullable=True)
    study_area = Column(String(200), nullable=True)
    target_species = Column(Text, nullable=True)
    
    # Temporal scope
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    
    # Personnel
    principal_investigator = Column(String(100), nullable=True)
    organization = Column(String(200), nullable=True)
    collaborators = Column(Text, nullable=True)
    
    # Sample information
    total_specimens = Column(Integer, nullable=True)
    species_count = Column(Integer, nullable=True)
    
    # Publication information
    publication_status = Column(String(50), nullable=True)
    doi = Column(String(100), nullable=True)
    citation = Column(Text, nullable=True)
    
    # Data availability
    data_sharing = Column(String(50), nullable=True)  # "public", "restricted", "private"
    license = Column(String(100), nullable=True)
    
    notes = Column(Text, nullable=True)