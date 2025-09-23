"""
Fisheries models for catch data, vessel management, and fishing operations.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from enum import Enum as PyEnum
from app.models.base import Base


class VesselType(PyEnum):
    """Types of fishing vessels."""
    TRAWLER = "trawler"
    LONGLINER = "longliner"
    PURSE_SEINER = "purse_seiner"
    GILLNETTER = "gillnetter"
    ARTISANAL = "artisanal"
    RESEARCH = "research"


class FishingMethod(PyEnum):
    """Fishing methods and gear types."""
    BOTTOM_TRAWL = "bottom_trawl"
    MIDWATER_TRAWL = "midwater_trawl"
    LONGLINE = "longline"
    PURSE_SEINE = "purse_seine"
    GILLNET = "gillnet"
    HOOK_AND_LINE = "hook_and_line"
    TRAP = "trap"
    SEINE_NET = "seine_net"


class CatchStatus(PyEnum):
    """Status of catch record."""
    DRAFT = "draft"
    VALIDATED = "validated"
    PROCESSED = "processed"
    EXPORTED = "exported"


class FishingVessel(Base):
    """
    Fishing vessel information and registration.
    
    Attributes:
        id: Primary key
        vessel_name: Name of the vessel
        registration_number: Official registration/license number
        vessel_type: Type of fishing vessel
        length: Vessel length in meters
        gross_tonnage: Gross tonnage of the vessel
        engine_power: Engine power in kW
        home_port: Home port of the vessel
        owner_name: Name of vessel owner
        captain_name: Name of vessel captain
        crew_size: Number of crew members
        is_active: Whether vessel is currently active
        registration_date: Date of vessel registration
        last_inspection: Date of last safety inspection
        created_at: Record creation timestamp
        notes: Additional notes about the vessel
    """
    __tablename__ = "fishing_vessels"

    id = Column(Integer, primary_key=True, index=True)
    vessel_name = Column(String(255), nullable=False, index=True)
    registration_number = Column(String(100), unique=True, nullable=False, index=True)
    vessel_type = Column(Enum(VesselType), nullable=False)
    length = Column(Float, nullable=True)
    gross_tonnage = Column(Float, nullable=True)
    engine_power = Column(Float, nullable=True)
    home_port = Column(String(255), nullable=True)
    owner_name = Column(String(255), nullable=True)
    captain_name = Column(String(255), nullable=True)
    crew_size = Column(Integer, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    registration_date = Column(DateTime, nullable=True)
    last_inspection = Column(DateTime, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    notes = Column(Text, nullable=True)

    # Relationships
    fishing_trips = relationship("FishingTrip", back_populates="vessel")
    catch_records = relationship("CatchRecord", back_populates="vessel")


class FishingTrip(Base):
    """
    Individual fishing trip information.
    
    Attributes:
        id: Primary key
        vessel_id: Foreign key to fishing vessel
        trip_number: Trip identification number
        departure_date: Trip start date and time
        return_date: Trip end date and time
        departure_port: Port of departure
        landing_port: Port where catch was landed
        fishing_areas: Fishing areas visited (JSON or comma-separated)
        target_species: Primary target species
        fishing_method: Primary fishing method used
        crew_count: Number of crew on this trip
        fuel_consumed: Fuel consumption in liters
        total_catch_weight: Total weight of catch in kg
        trip_duration: Trip duration in hours
        weather_conditions: Weather conditions during trip
        coordinates_start: Starting coordinates (lat,lon)
        coordinates_end: Ending coordinates (lat,lon)
        is_completed: Whether trip is completed
        created_at: Record creation timestamp
        notes: Additional trip notes
    """
    __tablename__ = "fishing_trips"

    id = Column(Integer, primary_key=True, index=True)
    vessel_id = Column(Integer, ForeignKey("fishing_vessels.id"), nullable=False)
    trip_number = Column(String(100), nullable=False, index=True)
    departure_date = Column(DateTime, nullable=False)
    return_date = Column(DateTime, nullable=True)
    departure_port = Column(String(255), nullable=True)
    landing_port = Column(String(255), nullable=True)
    fishing_areas = Column(Text, nullable=True)  # JSON string of areas
    target_species = Column(String(255), nullable=True)
    fishing_method = Column(Enum(FishingMethod), nullable=True)
    crew_count = Column(Integer, nullable=True)
    fuel_consumed = Column(Float, nullable=True)
    total_catch_weight = Column(Float, nullable=True)
    trip_duration = Column(Float, nullable=True)  # hours
    weather_conditions = Column(Text, nullable=True)
    coordinates_start = Column(String(100), nullable=True)  # "lat,lon"
    coordinates_end = Column(String(100), nullable=True)    # "lat,lon"
    is_completed = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    notes = Column(Text, nullable=True)

    # Relationships
    vessel = relationship("FishingVessel", back_populates="fishing_trips")
    catch_records = relationship("CatchRecord", back_populates="fishing_trip")


class CatchRecord(Base):
    """
    Individual catch records by species.
    
    Attributes:
        id: Primary key
        vessel_id: Foreign key to fishing vessel
        trip_id: Foreign key to fishing trip
        species_id: Foreign key to taxonomic unit (species)
        catch_date: Date and time of catch
        fishing_area: Specific fishing area
        fishing_method: Method used for this catch
        catch_weight: Weight of catch in kg
        catch_quantity: Number of individuals caught
        average_size: Average size/length in cm
        market_grade: Market grade or quality
        landing_port: Port where this catch was landed
        buyer_name: Name of buyer/processor
        price_per_kg: Price per kilogram
        total_value: Total value of this catch
        coordinates: GPS coordinates of catch
        depth: Fishing depth in meters
        water_temperature: Water temperature in Celsius
        catch_status: Status of this catch record
        processing_method: How catch was processed
        storage_method: Storage method used
        quality_notes: Notes about catch quality
        created_at: Record creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = "catch_records"

    id = Column(Integer, primary_key=True, index=True)
    vessel_id = Column(Integer, ForeignKey("fishing_vessels.id"), nullable=False)
    trip_id = Column(Integer, ForeignKey("fishing_trips.id"), nullable=True)
    species_id = Column(Integer, ForeignKey("taxonomic_units.id"), nullable=False)
    catch_date = Column(DateTime, nullable=False, index=True)
    fishing_area = Column(String(255), nullable=True)
    fishing_method = Column(Enum(FishingMethod), nullable=True)
    catch_weight = Column(Float, nullable=False)
    catch_quantity = Column(Integer, nullable=True)
    average_size = Column(Float, nullable=True)
    market_grade = Column(String(50), nullable=True)
    landing_port = Column(String(255), nullable=True)
    buyer_name = Column(String(255), nullable=True)
    price_per_kg = Column(Float, nullable=True)
    total_value = Column(Float, nullable=True)
    coordinates = Column(String(100), nullable=True)  # "lat,lon"
    depth = Column(Float, nullable=True)
    water_temperature = Column(Float, nullable=True)
    catch_status = Column(Enum(CatchStatus), default=CatchStatus.DRAFT, nullable=False)
    processing_method = Column(String(255), nullable=True)
    storage_method = Column(String(255), nullable=True)
    quality_notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    vessel = relationship("FishingVessel", back_populates="catch_records")
    fishing_trip = relationship("FishingTrip", back_populates="catch_records")
    species = relationship("TaxonomicUnit")


class FishingQuota(Base):
    """
    Fishing quota management by species and vessel.
    
    Attributes:
        id: Primary key
        vessel_id: Foreign key to fishing vessel
        species_id: Foreign key to taxonomic unit (species)
        quota_year: Year for this quota
        allocated_quota: Allocated quota in kg
        used_quota: Used quota in kg
        remaining_quota: Remaining quota in kg
        quota_type: Type of quota (annual, seasonal, etc.)
        start_date: Quota period start date
        end_date: Quota period end date
        is_active: Whether quota is currently active
        created_at: Record creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = "fishing_quotas"

    id = Column(Integer, primary_key=True, index=True)
    vessel_id = Column(Integer, ForeignKey("fishing_vessels.id"), nullable=False)
    species_id = Column(Integer, ForeignKey("taxonomic_units.id"), nullable=False)
    quota_year = Column(Integer, nullable=False, index=True)
    allocated_quota = Column(Float, nullable=False)
    used_quota = Column(Float, default=0.0, nullable=False)
    remaining_quota = Column(Float, nullable=False)
    quota_type = Column(String(50), default="annual", nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    vessel = relationship("FishingVessel")
    species = relationship("TaxonomicUnit")


class MarketPrice(Base):
    """
    Market price tracking for fish species.
    
    Attributes:
        id: Primary key
        species_id: Foreign key to taxonomic unit (species)
        market_location: Market or port location
        price_date: Date of price record
        price_per_kg: Price per kilogram
        market_grade: Market grade or quality
        supply_volume: Volume of supply in kg
        demand_level: Demand level (high, medium, low)
        price_trend: Price trend (increasing, stable, decreasing)
        seasonal_factor: Seasonal price factor
        created_at: Record creation timestamp
        notes: Additional market notes
    """
    __tablename__ = "market_prices"

    id = Column(Integer, primary_key=True, index=True)
    species_id = Column(Integer, ForeignKey("taxonomic_units.id"), nullable=False)
    market_location = Column(String(255), nullable=False)
    price_date = Column(DateTime, nullable=False, index=True)
    price_per_kg = Column(Float, nullable=False)
    market_grade = Column(String(50), nullable=True)
    supply_volume = Column(Float, nullable=True)
    demand_level = Column(String(20), nullable=True)
    price_trend = Column(String(20), nullable=True)
    seasonal_factor = Column(Float, default=1.0, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    notes = Column(Text, nullable=True)

    # Relationships
    species = relationship("TaxonomicUnit")