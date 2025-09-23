from sqlalchemy import Column, String, Float, DateTime, Integer, Text, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from .base import BaseModel

class OceanographicStation(BaseModel):
    """Oceanographic monitoring stations"""
    __tablename__ = "oceanographic_stations"
    
    station_id = Column(String(50), nullable=False, unique=True, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    
    # Location
    latitude = Column(Float, nullable=False)  # WGS84 coordinates
    longitude = Column(Float, nullable=False)
    depth = Column(Float, nullable=True)  # Station depth in meters
    
    # Station metadata
    station_type = Column(String(50), nullable=True)  # "buoy", "shore", "ship", "satellite"
    operator = Column(String(100), nullable=True)
    country = Column(String(100), nullable=True)
    region = Column(String(100), nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    
    # Relationships
    measurements = relationship("OceanographicMeasurement", back_populates="station")

class OceanographicParameter(BaseModel):
    """Parameter definitions for oceanographic measurements"""
    __tablename__ = "oceanographic_parameters"
    
    parameter_code = Column(String(20), nullable=False, unique=True)
    parameter_name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    unit = Column(String(20), nullable=False)
    unit_description = Column(String(100), nullable=True)
    
    # Parameter classification
    category = Column(String(50), nullable=True)  # "physical", "chemical", "biological"
    subcategory = Column(String(50), nullable=True)
    
    # Quality control thresholds
    min_value = Column(Float, nullable=True)
    max_value = Column(Float, nullable=True)
    typical_range_min = Column(Float, nullable=True)
    typical_range_max = Column(Float, nullable=True)
    
    # Relationships
    measurements = relationship("OceanographicMeasurement", back_populates="parameter")

class OceanographicMeasurement(BaseModel):
    """Individual oceanographic measurements"""
    __tablename__ = "oceanographic_measurements"
    
    station_id = Column(Integer, ForeignKey("oceanographic_stations.id"), nullable=False, index=True)
    parameter_id = Column(Integer, ForeignKey("oceanographic_parameters.id"), nullable=False, index=True)
    
    # Measurement details
    measurement_time = Column(DateTime, nullable=False, index=True)
    value = Column(Float, nullable=False)
    measurement_depth = Column(Float, nullable=True)  # Depth at which measurement was taken
    
    # Quality control
    quality_flag = Column(String(10), nullable=True)  # "good", "suspect", "bad", "missing"
    quality_score = Column(Float, nullable=True)  # 0-1 quality score
    processing_level = Column(String(20), nullable=True)  # "raw", "processed", "quality_controlled"
    
    # Measurement metadata
    instrument = Column(String(100), nullable=True)
    method = Column(String(100), nullable=True)
    sampling_frequency = Column(String(50), nullable=True)
    data_source = Column(String(100), nullable=True)
    
    # Additional context
    weather_conditions = Column(Text, nullable=True)
    sea_state = Column(String(50), nullable=True)
    notes = Column(Text, nullable=True)
    
    # Relationships
    station = relationship("OceanographicStation", back_populates="measurements")
    parameter = relationship("OceanographicParameter", back_populates="measurements")

class OceanographicDataset(BaseModel):
    """Dataset metadata for bulk oceanographic data"""
    __tablename__ = "oceanographic_datasets"
    
    dataset_name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    source_organization = Column(String(200), nullable=True)
    
    # Temporal coverage
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    
    # Spatial coverage
    bbox_north = Column(Float, nullable=True)
    bbox_south = Column(Float, nullable=True)
    bbox_east = Column(Float, nullable=True)
    bbox_west = Column(Float, nullable=True)
    
    # Dataset metadata
    version = Column(String(20), nullable=True)
    license = Column(String(100), nullable=True)
    citation = Column(Text, nullable=True)
    doi = Column(String(100), nullable=True)
    url = Column(Text, nullable=True)
    
    # Processing information
    processing_date = Column(DateTime, nullable=True)
    processing_version = Column(String(50), nullable=True)
    total_records = Column(Integer, nullable=True)
    
class OceanographicAlert(BaseModel):
    """Environmental alerts and thresholds"""
    __tablename__ = "oceanographic_alerts"
    
    station_id = Column(Integer, ForeignKey("oceanographic_stations.id"), nullable=False)
    parameter_id = Column(Integer, ForeignKey("oceanographic_parameters.id"), nullable=False)
    
    alert_type = Column(String(50), nullable=False)  # "threshold", "anomaly", "trend"
    severity = Column(String(20), nullable=False)  # "low", "medium", "high", "critical"
    
    threshold_value = Column(Float, nullable=True)
    measured_value = Column(Float, nullable=False)
    alert_time = Column(DateTime, nullable=False)
    
    message = Column(Text, nullable=True)
    is_acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(100), nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    
    # Relationships
    station = relationship("OceanographicStation")
    parameter = relationship("OceanographicParameter")