"""
Pydantic schemas for fisheries models.
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, validator
from app.models.fisheries import VesselType, FishingMethod, CatchStatus


# Fishing Vessel Schemas
class FishingVesselBase(BaseModel):
    """Base fishing vessel schema."""
    vessel_name: str
    registration_number: str
    vessel_type: VesselType
    length: Optional[float] = None
    gross_tonnage: Optional[float] = None
    engine_power: Optional[float] = None
    home_port: Optional[str] = None
    owner_name: Optional[str] = None
    captain_name: Optional[str] = None
    crew_size: Optional[int] = None
    registration_date: Optional[datetime] = None
    last_inspection: Optional[datetime] = None
    notes: Optional[str] = None


class FishingVesselCreate(FishingVesselBase):
    """Schema for creating fishing vessel."""
    
    @validator('length')
    def validate_length(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Vessel length must be positive')
        return v
    
    @validator('crew_size')
    def validate_crew_size(cls, v):
        if v is not None and v <= 0:
            raise ValueError('Crew size must be positive')
        return v


class FishingVesselUpdate(BaseModel):
    """Schema for updating fishing vessel."""
    vessel_name: Optional[str] = None
    vessel_type: Optional[VesselType] = None
    length: Optional[float] = None
    gross_tonnage: Optional[float] = None
    engine_power: Optional[float] = None
    home_port: Optional[str] = None
    owner_name: Optional[str] = None
    captain_name: Optional[str] = None
    crew_size: Optional[int] = None
    is_active: Optional[bool] = None
    registration_date: Optional[datetime] = None
    last_inspection: Optional[datetime] = None
    notes: Optional[str] = None


class FishingVesselResponse(FishingVesselBase):
    """Schema for fishing vessel response."""
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


# Fishing Trip Schemas
class FishingTripBase(BaseModel):
    """Base fishing trip schema."""
    trip_number: str
    departure_date: datetime
    return_date: Optional[datetime] = None
    departure_port: Optional[str] = None
    landing_port: Optional[str] = None
    fishing_areas: Optional[str] = None
    target_species: Optional[str] = None
    fishing_method: Optional[FishingMethod] = None
    crew_count: Optional[int] = None
    fuel_consumed: Optional[float] = None
    total_catch_weight: Optional[float] = None
    trip_duration: Optional[float] = None
    weather_conditions: Optional[str] = None
    coordinates_start: Optional[str] = None
    coordinates_end: Optional[str] = None
    notes: Optional[str] = None


class FishingTripCreate(FishingTripBase):
    """Schema for creating fishing trip."""
    vessel_id: int
    
    @validator('fuel_consumed')
    def validate_fuel_consumed(cls, v):
        if v is not None and v < 0:
            raise ValueError('Fuel consumed cannot be negative')
        return v
    
    @validator('total_catch_weight')
    def validate_total_catch_weight(cls, v):
        if v is not None and v < 0:
            raise ValueError('Total catch weight cannot be negative')
        return v


class FishingTripUpdate(BaseModel):
    """Schema for updating fishing trip."""
    return_date: Optional[datetime] = None
    landing_port: Optional[str] = None
    fishing_areas: Optional[str] = None
    target_species: Optional[str] = None
    fishing_method: Optional[FishingMethod] = None
    crew_count: Optional[int] = None
    fuel_consumed: Optional[float] = None
    total_catch_weight: Optional[float] = None
    trip_duration: Optional[float] = None
    weather_conditions: Optional[str] = None
    coordinates_end: Optional[str] = None
    is_completed: Optional[bool] = None
    notes: Optional[str] = None


class FishingTripResponse(FishingTripBase):
    """Schema for fishing trip response."""
    id: int
    vessel_id: int
    is_completed: bool
    created_at: datetime
    vessel: Optional[FishingVesselResponse] = None
    
    class Config:
        from_attributes = True


# Catch Record Schemas
class CatchRecordBase(BaseModel):
    """Base catch record schema."""
    catch_date: datetime
    fishing_area: Optional[str] = None
    fishing_method: Optional[FishingMethod] = None
    catch_weight: float
    catch_quantity: Optional[int] = None
    average_size: Optional[float] = None
    market_grade: Optional[str] = None
    landing_port: Optional[str] = None
    buyer_name: Optional[str] = None
    price_per_kg: Optional[float] = None
    total_value: Optional[float] = None
    coordinates: Optional[str] = None
    depth: Optional[float] = None
    water_temperature: Optional[float] = None
    processing_method: Optional[str] = None
    storage_method: Optional[str] = None
    quality_notes: Optional[str] = None


class CatchRecordCreate(CatchRecordBase):
    """Schema for creating catch record."""
    vessel_id: int
    species_id: int
    trip_id: Optional[int] = None
    
    @validator('catch_weight')
    def validate_catch_weight(cls, v):
        if v <= 0:
            raise ValueError('Catch weight must be positive')
        return v
    
    @validator('price_per_kg')
    def validate_price_per_kg(cls, v):
        if v is not None and v < 0:
            raise ValueError('Price per kg cannot be negative')
        return v


class CatchRecordUpdate(BaseModel):
    """Schema for updating catch record."""
    fishing_area: Optional[str] = None
    fishing_method: Optional[FishingMethod] = None
    catch_weight: Optional[float] = None
    catch_quantity: Optional[int] = None
    average_size: Optional[float] = None
    market_grade: Optional[str] = None
    landing_port: Optional[str] = None
    buyer_name: Optional[str] = None
    price_per_kg: Optional[float] = None
    total_value: Optional[float] = None
    coordinates: Optional[str] = None
    depth: Optional[float] = None
    water_temperature: Optional[float] = None
    catch_status: Optional[CatchStatus] = None
    processing_method: Optional[str] = None
    storage_method: Optional[str] = None
    quality_notes: Optional[str] = None


class CatchRecordResponse(CatchRecordBase):
    """Schema for catch record response."""
    id: int
    vessel_id: int
    species_id: int
    trip_id: Optional[int] = None
    catch_status: CatchStatus
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# Fishing Quota Schemas
class FishingQuotaBase(BaseModel):
    """Base fishing quota schema."""
    quota_year: int
    allocated_quota: float
    quota_type: str = "annual"
    start_date: datetime
    end_date: datetime


class FishingQuotaCreate(FishingQuotaBase):
    """Schema for creating fishing quota."""
    vessel_id: int
    species_id: int
    
    @validator('allocated_quota')
    def validate_allocated_quota(cls, v):
        if v <= 0:
            raise ValueError('Allocated quota must be positive')
        return v
    
    @validator('end_date')
    def validate_end_date(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('End date must be after start date')
        return v


class FishingQuotaUpdate(BaseModel):
    """Schema for updating fishing quota."""
    allocated_quota: Optional[float] = None
    used_quota: Optional[float] = None
    remaining_quota: Optional[float] = None
    quota_type: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    is_active: Optional[bool] = None


class FishingQuotaResponse(FishingQuotaBase):
    """Schema for fishing quota response."""
    id: int
    vessel_id: int
    species_id: int
    used_quota: float
    remaining_quota: float
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# Market Price Schemas
class MarketPriceBase(BaseModel):
    """Base market price schema."""
    market_location: str
    price_date: datetime
    price_per_kg: float
    market_grade: Optional[str] = None
    supply_volume: Optional[float] = None
    demand_level: Optional[str] = None
    price_trend: Optional[str] = None
    seasonal_factor: Optional[float] = 1.0
    notes: Optional[str] = None


class MarketPriceCreate(MarketPriceBase):
    """Schema for creating market price."""
    species_id: int
    
    @validator('price_per_kg')
    def validate_price_per_kg(cls, v):
        if v < 0:
            raise ValueError('Price per kg cannot be negative')
        return v


class MarketPriceUpdate(BaseModel):
    """Schema for updating market price."""
    market_location: Optional[str] = None
    price_date: Optional[datetime] = None
    price_per_kg: Optional[float] = None
    market_grade: Optional[str] = None
    supply_volume: Optional[float] = None
    demand_level: Optional[str] = None
    price_trend: Optional[str] = None
    seasonal_factor: Optional[float] = None
    notes: Optional[str] = None


class MarketPriceResponse(MarketPriceBase):
    """Schema for market price response."""
    id: int
    species_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


# Comprehensive Search Response
class FisheriesSearchResponse(BaseModel):
    """Response schema for fisheries search results."""
    items: List[CatchRecordResponse]
    total: int
    limit: int
    offset: int
    has_more: bool


# Statistics Schemas
class FisheriesStats(BaseModel):
    """Fisheries statistics response."""
    total_vessels: int
    active_vessels: int
    total_trips: int
    completed_trips: int
    total_catch_records: int
    total_catch_weight: float
    average_catch_per_trip: float
    top_species: List[dict]
    top_fishing_areas: List[dict]