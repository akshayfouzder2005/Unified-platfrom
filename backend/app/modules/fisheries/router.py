"""
Comprehensive fisheries management API endpoints.
"""
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.dependencies import require_write_access, require_read_access, get_optional_user
from app.models.user import User
from app.models.fisheries import VesselType, FishingMethod, CatchStatus
from app.crud.fisheries import fisheries_crud
from app.schemas.fisheries import (
    FishingVesselCreate, FishingVesselUpdate, FishingVesselResponse,
    FishingTripCreate, FishingTripUpdate, FishingTripResponse,
    CatchRecordCreate, CatchRecordUpdate, CatchRecordResponse,
    FishingQuotaCreate, FishingQuotaUpdate, FishingQuotaResponse,
    MarketPriceCreate, MarketPriceUpdate, MarketPriceResponse,
    FisheriesSearchResponse, FisheriesStats
)
from app.core.exceptions import ValidationError, DatabaseError


router = APIRouter(prefix="/fisheries", tags=["Fisheries"])


# Fishing Vessel Endpoints
@router.post("/vessels", response_model=FishingVesselResponse, status_code=status.HTTP_201_CREATED)
async def create_vessel(
    vessel_data: FishingVesselCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_write_access)
):
    """Create a new fishing vessel."""
    try:
        # Check if registration number already exists
        existing_vessel = fisheries_crud.get_vessel_by_registration(db, vessel_data.registration_number)
        if existing_vessel:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Vessel with this registration number already exists"
            )
        
        vessel = fisheries_crud.create_vessel(db, vessel_data)
        return vessel
        
    except HTTPException:
        raise
    except Exception as e:
        raise DatabaseError(f"Failed to create vessel: {str(e)}")


@router.get("/vessels", response_model=List[FishingVesselResponse])
async def list_vessels(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, le=1000),
    vessel_type: Optional[VesselType] = Query(None),
    home_port: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_read_access)
):
    """Get list of fishing vessels with filtering."""
    try:
        vessels = fisheries_crud.get_vessels(
            db, skip=skip, limit=limit, vessel_type=vessel_type, 
            home_port=home_port, is_active=is_active
        )
        return vessels
        
    except Exception as e:
        raise DatabaseError(f"Failed to retrieve vessels: {str(e)}")


@router.get("/vessels/{vessel_id}", response_model=FishingVesselResponse)
async def get_vessel(
    vessel_id: int = Path(..., gt=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_read_access)
):
    """Get fishing vessel by ID."""
    vessel = fisheries_crud.get_vessel(db, vessel_id)
    if not vessel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vessel not found"
        )
    return vessel


@router.put("/vessels/{vessel_id}", response_model=FishingVesselResponse)
async def update_vessel(
    vessel_id: int = Path(..., gt=0),
    vessel_update: FishingVesselUpdate = ...,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_write_access)
):
    """Update fishing vessel."""
    try:
        vessel = fisheries_crud.update_vessel(db, vessel_id, vessel_update)
        if not vessel:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Vessel not found"
            )
        return vessel
        
    except HTTPException:
        raise
    except Exception as e:
        raise DatabaseError(f"Failed to update vessel: {str(e)}")


@router.delete("/vessels/{vessel_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_vessel(
    vessel_id: int = Path(..., gt=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_write_access)
):
    """Delete (deactivate) fishing vessel."""
    try:
        success = fisheries_crud.delete_vessel(db, vessel_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Vessel not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise DatabaseError(f"Failed to delete vessel: {str(e)}")


# Fishing Trip Endpoints
@router.post("/trips", response_model=FishingTripResponse, status_code=status.HTTP_201_CREATED)
async def create_trip(
    trip_data: FishingTripCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_write_access)
):
    """Create a new fishing trip."""
    try:
        # Verify vessel exists
        vessel = fisheries_crud.get_vessel(db, trip_data.vessel_id)
        if not vessel:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid vessel ID"
            )
        
        trip = fisheries_crud.create_trip(db, trip_data)
        return trip
        
    except HTTPException:
        raise
    except Exception as e:
        raise DatabaseError(f"Failed to create trip: {str(e)}")


@router.get("/trips", response_model=List[FishingTripResponse])
async def list_trips(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, le=1000),
    vessel_id: Optional[int] = Query(None),
    is_completed: Optional[bool] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_read_access)
):
    """Get list of fishing trips with filtering."""
    try:
        trips = fisheries_crud.get_trips(
            db, skip=skip, limit=limit, vessel_id=vessel_id,
            is_completed=is_completed, start_date=start_date, end_date=end_date
        )
        return trips
        
    except Exception as e:
        raise DatabaseError(f"Failed to retrieve trips: {str(e)}")


@router.get("/trips/{trip_id}", response_model=FishingTripResponse)
async def get_trip(
    trip_id: int = Path(..., gt=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_read_access)
):
    """Get fishing trip by ID."""
    trip = fisheries_crud.get_trip(db, trip_id)
    if not trip:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trip not found"
        )
    return trip


@router.put("/trips/{trip_id}", response_model=FishingTripResponse)
async def update_trip(
    trip_id: int = Path(..., gt=0),
    trip_update: FishingTripUpdate = ...,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_write_access)
):
    """Update fishing trip."""
    try:
        trip = fisheries_crud.update_trip(db, trip_id, trip_update)
        if not trip:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Trip not found"
            )
        return trip
        
    except HTTPException:
        raise
    except Exception as e:
        raise DatabaseError(f"Failed to update trip: {str(e)}")


@router.post("/trips/{trip_id}/complete", response_model=FishingTripResponse)
async def complete_trip(
    trip_id: int = Path(..., gt=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_write_access)
):
    """Mark fishing trip as completed."""
    try:
        trip = fisheries_crud.complete_trip(db, trip_id)
        if not trip:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Trip not found"
            )
        return trip
        
    except Exception as e:
        raise DatabaseError(f"Failed to complete trip: {str(e)}")


# Catch Record Endpoints
@router.post("/catches", response_model=CatchRecordResponse, status_code=status.HTTP_201_CREATED)
async def create_catch_record(
    catch_data: CatchRecordCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_write_access)
):
    """Create a new catch record."""
    try:
        # Verify vessel and species exist
        vessel = fisheries_crud.get_vessel(db, catch_data.vessel_id)
        if not vessel:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid vessel ID"
            )
        
        # Verify trip exists if provided
        if catch_data.trip_id:
            trip = fisheries_crud.get_trip(db, catch_data.trip_id)
            if not trip:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid trip ID"
                )
        
        catch_record = fisheries_crud.create_catch_record(db, catch_data)
        
        # Update quota usage
        fisheries_crud.update_quota_usage(
            db, catch_data.vessel_id, catch_data.species_id, catch_data.catch_weight
        )
        
        return catch_record
        
    except HTTPException:
        raise
    except Exception as e:
        raise DatabaseError(f"Failed to create catch record: {str(e)}")


@router.get("/catches", response_model=FisheriesSearchResponse)
async def search_catch_records(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, le=1000),
    vessel_id: Optional[int] = Query(None),
    species_id: Optional[int] = Query(None),
    trip_id: Optional[int] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    fishing_area: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_read_access)
):
    """Search catch records with filtering and pagination."""
    try:
        catches, total = fisheries_crud.get_catch_records(
            db, skip=skip, limit=limit, vessel_id=vessel_id,
            species_id=species_id, trip_id=trip_id,
            start_date=start_date, end_date=end_date, fishing_area=fishing_area
        )
        
        return {
            "items": catches,
            "total": total,
            "limit": limit,
            "offset": skip,
            "has_more": skip + limit < total
        }
        
    except Exception as e:
        raise DatabaseError(f"Failed to search catch records: {str(e)}")


@router.get("/catches/{catch_id}", response_model=CatchRecordResponse)
async def get_catch_record(
    catch_id: int = Path(..., gt=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_read_access)
):
    """Get catch record by ID."""
    catch_record = fisheries_crud.get_catch_record(db, catch_id)
    if not catch_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Catch record not found"
        )
    return catch_record


@router.put("/catches/{catch_id}", response_model=CatchRecordResponse)
async def update_catch_record(
    catch_id: int = Path(..., gt=0),
    catch_update: CatchRecordUpdate = ...,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_write_access)
):
    """Update catch record."""
    try:
        catch_record = fisheries_crud.update_catch_record(db, catch_id, catch_update)
        if not catch_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Catch record not found"
            )
        return catch_record
        
    except HTTPException:
        raise
    except Exception as e:
        raise DatabaseError(f"Failed to update catch record: {str(e)}")


@router.delete("/catches/{catch_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_catch_record(
    catch_id: int = Path(..., gt=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_write_access)
):
    """Delete catch record."""
    try:
        success = fisheries_crud.delete_catch_record(db, catch_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Catch record not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise DatabaseError(f"Failed to delete catch record: {str(e)}")


# Quota Management Endpoints
@router.post("/quotas", response_model=FishingQuotaResponse, status_code=status.HTTP_201_CREATED)
async def create_quota(
    quota_data: FishingQuotaCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_write_access)
):
    """Create a new fishing quota."""
    try:
        quota = fisheries_crud.create_quota(db, quota_data)
        return quota
        
    except Exception as e:
        raise DatabaseError(f"Failed to create quota: {str(e)}")


@router.get("/quotas", response_model=List[FishingQuotaResponse])
async def list_quotas(
    vessel_id: Optional[int] = Query(None),
    species_id: Optional[int] = Query(None),
    quota_year: Optional[int] = Query(None),
    is_active: Optional[bool] = Query(True),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_read_access)
):
    """Get fishing quotas with filtering."""
    try:
        quotas = fisheries_crud.get_quotas(
            db, vessel_id=vessel_id, species_id=species_id,
            quota_year=quota_year, is_active=is_active
        )
        return quotas
        
    except Exception as e:
        raise DatabaseError(f"Failed to retrieve quotas: {str(e)}")


# Market Price Endpoints
@router.post("/market-prices", response_model=MarketPriceResponse, status_code=status.HTTP_201_CREATED)
async def create_market_price(
    price_data: MarketPriceCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_write_access)
):
    """Create a new market price record."""
    try:
        price = fisheries_crud.create_market_price(db, price_data)
        return price
        
    except Exception as e:
        raise DatabaseError(f"Failed to create market price: {str(e)}")


@router.get("/market-prices", response_model=List[MarketPriceResponse])
async def list_market_prices(
    species_id: Optional[int] = Query(None),
    market_location: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_read_access)
):
    """Get market prices with filtering."""
    try:
        prices = fisheries_crud.get_market_prices(
            db, species_id=species_id, market_location=market_location,
            start_date=start_date, end_date=end_date
        )
        return prices
        
    except Exception as e:
        raise DatabaseError(f"Failed to retrieve market prices: {str(e)}")


# Statistics and Analytics Endpoints
@router.get("/stats", response_model=FisheriesStats)
async def get_fisheries_statistics(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_read_access)
):
    """Get comprehensive fisheries statistics."""
    try:
        stats = fisheries_crud.get_fisheries_stats(db)
        return FisheriesStats(**stats)
        
    except Exception as e:
        raise DatabaseError(f"Failed to get fisheries statistics: {str(e)}")


@router.get("/analytics/catch-trends")
async def get_catch_trends(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_read_access)
):
    """Get catch trends over specified number of days."""
    try:
        trends = fisheries_crud.get_catch_trends(db, days)
        return {
            "period_days": days,
            "trends": trends
        }
        
    except Exception as e:
        raise DatabaseError(f"Failed to get catch trends: {str(e)}")


# Utility Endpoints
@router.get("/vessel-types")
async def get_vessel_types():
    """Get available vessel types."""
    return [{"value": vt.value, "label": vt.value.replace("_", " ").title()} for vt in VesselType]


@router.get("/fishing-methods")
async def get_fishing_methods():
    """Get available fishing methods."""
    return [{"value": fm.value, "label": fm.value.replace("_", " ").title()} for fm in FishingMethod]


@router.get("/catch-statuses")
async def get_catch_statuses():
    """Get available catch statuses."""
    return [{"value": cs.value, "label": cs.value.replace("_", " ").title()} for cs in CatchStatus]