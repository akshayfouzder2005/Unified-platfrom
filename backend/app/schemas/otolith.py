"""
Pydantic schemas for otolith models.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, field_validator


# Otolith Specimen Schemas
class OtolithSpecimenBase(BaseModel):
    """Base otolith specimen schema."""
    specimen_id: str
    specimen_name: Optional[str] = None
    fish_species_id: Optional[int] = None
    fish_length: Optional[float] = None
    fish_weight: Optional[float] = None
    fish_age: Optional[int] = None
    fish_sex: Optional[str] = None
    collection_date: Optional[datetime] = None
    collected_by: Optional[str] = None
    collection_method: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    depth: Optional[float] = None
    location_description: Optional[str] = None
    otolith_type: str
    otolith_side: Optional[str] = None
    preservation_method: Optional[str] = None
    storage_location: Optional[str] = None
    condition: Optional[str] = None
    completeness: Optional[float] = None
    notes: Optional[str] = None
    project_name: Optional[str] = None


class OtolithSpecimenCreate(OtolithSpecimenBase):
    """Schema for creating otolith specimen."""
    
    @field_validator('otolith_type')
    @classmethod
    def validate_otolith_type(cls, v):
        valid_types = ['sagitta', 'lapillus', 'asteriscus']
        if v not in valid_types:
            raise ValueError(f'Otolith type must be one of: {valid_types}')
        return v
    
    @field_validator('fish_sex')
    @classmethod
    def validate_fish_sex(cls, v):
        if v is not None:
            valid_sexes = ['male', 'female', 'unknown']
            if v not in valid_sexes:
                raise ValueError(f'Fish sex must be one of: {valid_sexes}')
        return v
    
    @field_validator('condition')
    @classmethod
    def validate_condition(cls, v):
        if v is not None:
            valid_conditions = ['excellent', 'good', 'fair', 'poor', 'damaged']
            if v not in valid_conditions:
                raise ValueError(f'Condition must be one of: {valid_conditions}')
        return v


class OtolithSpecimenUpdate(BaseModel):
    """Schema for updating otolith specimen."""
    specimen_name: Optional[str] = None
    fish_species_id: Optional[int] = None
    fish_length: Optional[float] = None
    fish_weight: Optional[float] = None
    fish_age: Optional[int] = None
    fish_sex: Optional[str] = None
    collection_date: Optional[datetime] = None
    collected_by: Optional[str] = None
    collection_method: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    depth: Optional[float] = None
    location_description: Optional[str] = None
    otolith_side: Optional[str] = None
    preservation_method: Optional[str] = None
    storage_location: Optional[str] = None
    condition: Optional[str] = None
    completeness: Optional[float] = None
    notes: Optional[str] = None
    project_name: Optional[str] = None


class OtolithSpecimenResponse(OtolithSpecimenBase):
    """Schema for otolith specimen response."""
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# Otolith Measurement Schemas
class OtolithMeasurementBase(BaseModel):
    """Base otolith measurement schema."""
    measurement_type: str
    length: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    area: Optional[float] = None
    perimeter: Optional[float] = None
    roundness: Optional[float] = None
    rectangularity: Optional[float] = None
    ellipticity: Optional[float] = None
    form_factor: Optional[float] = None
    aspect_ratio: Optional[float] = None
    fourier_descriptors: Optional[Dict[str, Any]] = None
    edge_roughness: Optional[float] = None
    surface_texture: Optional[float] = None
    opacity_index: Optional[float] = None
    measurement_date: datetime
    measured_by: Optional[str] = None
    measurement_method: Optional[str] = None
    measurement_precision: Optional[float] = None
    equipment: Optional[str] = None
    software: Optional[str] = None
    calibration_factor: Optional[float] = None
    notes: Optional[str] = None


class OtolithMeasurementCreate(OtolithMeasurementBase):
    """Schema for creating otolith measurement."""
    specimen_id: int
    
    @field_validator('measurement_method')
    @classmethod
    def validate_measurement_method(cls, v):
        if v is not None:
            valid_methods = ['manual', 'image_analysis', '3D_scan']
            if v not in valid_methods:
                raise ValueError(f'Measurement method must be one of: {valid_methods}')
        return v


class OtolithMeasurementUpdate(BaseModel):
    """Schema for updating otolith measurement."""
    measurement_type: Optional[str] = None
    length: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    area: Optional[float] = None
    perimeter: Optional[float] = None
    roundness: Optional[float] = None
    rectangularity: Optional[float] = None
    ellipticity: Optional[float] = None
    form_factor: Optional[float] = None
    aspect_ratio: Optional[float] = None
    fourier_descriptors: Optional[Dict[str, Any]] = None
    edge_roughness: Optional[float] = None
    surface_texture: Optional[float] = None
    opacity_index: Optional[float] = None
    measured_by: Optional[str] = None
    measurement_method: Optional[str] = None
    measurement_precision: Optional[float] = None
    equipment: Optional[str] = None
    software: Optional[str] = None
    calibration_factor: Optional[float] = None
    notes: Optional[str] = None


class OtolithMeasurementResponse(OtolithMeasurementBase):
    """Schema for otolith measurement response."""
    id: int
    specimen_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# Otolith Image Schemas
class OtolithImageBase(BaseModel):
    """Base otolith image schema."""
    image_filename: str
    image_path: str
    image_type: str
    capture_date: Optional[datetime] = None
    captured_by: Optional[str] = None
    resolution_x: Optional[int] = None
    resolution_y: Optional[int] = None
    pixel_size: Optional[float] = None
    magnification: Optional[float] = None
    file_format: Optional[str] = None
    file_size: Optional[int] = None
    camera_model: Optional[str] = None
    lens_model: Optional[str] = None
    lighting_conditions: Optional[str] = None
    background_type: Optional[str] = None
    is_processed: bool = False
    processing_software: Optional[str] = None
    processing_notes: Optional[str] = None
    image_quality: Optional[str] = None
    focus_quality: Optional[str] = None
    lighting_quality: Optional[str] = None
    contrast_rating: Optional[int] = None
    has_landmarks: bool = False
    landmark_data: Optional[Dict[str, Any]] = None
    annotations: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class OtolithImageCreate(OtolithImageBase):
    """Schema for creating otolith image."""
    specimen_id: int
    
    @field_validator('image_type')
    @classmethod
    def validate_image_type(cls, v):
        valid_types = ['dorsal', 'ventral', 'lateral', '3D', 'microscopy']
        if v not in valid_types:
            raise ValueError(f'Image type must be one of: {valid_types}')
        return v


class OtolithImageUpdate(BaseModel):
    """Schema for updating otolith image."""
    image_filename: Optional[str] = None
    image_path: Optional[str] = None
    image_type: Optional[str] = None
    capture_date: Optional[datetime] = None
    captured_by: Optional[str] = None
    resolution_x: Optional[int] = None
    resolution_y: Optional[int] = None
    pixel_size: Optional[float] = None
    magnification: Optional[float] = None
    file_format: Optional[str] = None
    file_size: Optional[int] = None
    camera_model: Optional[str] = None
    lens_model: Optional[str] = None
    lighting_conditions: Optional[str] = None
    background_type: Optional[str] = None
    is_processed: Optional[bool] = None
    processing_software: Optional[str] = None
    processing_notes: Optional[str] = None
    image_quality: Optional[str] = None
    focus_quality: Optional[str] = None
    lighting_quality: Optional[str] = None
    contrast_rating: Optional[int] = None
    has_landmarks: Optional[bool] = None
    landmark_data: Optional[Dict[str, Any]] = None
    annotations: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class OtolithImageResponse(OtolithImageBase):
    """Schema for otolith image response."""
    id: int
    specimen_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# Otolith Classification Schemas
class OtolithClassificationBase(BaseModel):
    """Base otolith classification schema."""
    predicted_species_id: Optional[int] = None
    classification_method: str
    model_version: Optional[str] = None
    classification_date: datetime
    confidence_score: float
    top_predictions: Optional[Dict[str, Any]] = None
    key_features: Optional[Dict[str, Any]] = None
    feature_weights: Optional[Dict[str, Any]] = None
    is_validated: bool = False
    validated_by: Optional[str] = None
    validation_date: Optional[datetime] = None
    validation_notes: Optional[str] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    notes: Optional[str] = None


class OtolithClassificationCreate(OtolithClassificationBase):
    """Schema for creating otolith classification."""
    specimen_id: int
    
    @field_validator('confidence_score')
    @classmethod
    def validate_confidence_score(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence score must be between 0 and 1')
        return v


class OtolithClassificationUpdate(BaseModel):
    """Schema for updating otolith classification."""
    predicted_species_id: Optional[int] = None
    confidence_score: Optional[float] = None
    top_predictions: Optional[Dict[str, Any]] = None
    is_validated: Optional[bool] = None
    validated_by: Optional[str] = None
    validation_date: Optional[datetime] = None
    validation_notes: Optional[str] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    notes: Optional[str] = None


class OtolithClassificationResponse(OtolithClassificationBase):
    """Schema for otolith classification response."""
    id: int
    specimen_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# Otolith Study Schemas
class OtolithStudyBase(BaseModel):
    """Base otolith study schema."""
    study_name: str
    study_code: Optional[str] = None
    description: Optional[str] = None
    objective: Optional[str] = None
    methodology: Optional[str] = None
    study_area: Optional[str] = None
    target_species: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    principal_investigator: Optional[str] = None
    organization: Optional[str] = None
    collaborators: Optional[str] = None
    total_specimens: Optional[int] = None
    species_count: Optional[int] = None
    publication_status: Optional[str] = None
    doi: Optional[str] = None
    citation: Optional[str] = None
    data_sharing: Optional[str] = None
    license: Optional[str] = None
    notes: Optional[str] = None


class OtolithStudyCreate(OtolithStudyBase):
    """Schema for creating otolith study."""
    pass


class OtolithStudyUpdate(BaseModel):
    """Schema for updating otolith study."""
    study_name: Optional[str] = None
    study_code: Optional[str] = None
    description: Optional[str] = None
    objective: Optional[str] = None
    methodology: Optional[str] = None
    study_area: Optional[str] = None
    target_species: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    principal_investigator: Optional[str] = None
    organization: Optional[str] = None
    collaborators: Optional[str] = None
    total_specimens: Optional[int] = None
    species_count: Optional[int] = None
    publication_status: Optional[str] = None
    doi: Optional[str] = None
    citation: Optional[str] = None
    data_sharing: Optional[str] = None
    license: Optional[str] = None
    notes: Optional[str] = None


class OtolithStudyResponse(OtolithStudyBase):
    """Schema for otolith study response."""
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# Comprehensive Search and Analysis Schemas
class OtolithSearchResponse(BaseModel):
    """Response schema for otolith search results."""
    items: List[OtolithSpecimenResponse]
    total: int
    limit: int
    offset: int
    has_more: bool


class OtolithStats(BaseModel):
    """Otolith statistics response."""
    total_specimens: int
    total_measurements: int
    total_images: int
    total_classifications: int
    species_count: int
    measurement_types: List[str]
    image_types: List[str]
    classification_accuracy: Optional[float]
    top_species: List[dict]


class MorphometricAnalysis(BaseModel):
    """Morphometric analysis results."""
    specimen_count: int
    measurements_summary: Dict[str, Dict[str, float]]  # measurement_type -> {mean, std, min, max}
    shape_indices: Dict[str, float]
    species_discrimination: Optional[Dict[str, Any]]
    pca_results: Optional[Dict[str, Any]]
    cluster_analysis: Optional[Dict[str, Any]]