"""
CRUD operations for fisheries models.
"""
from typing import Optional, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, desc
from datetime import datetime, timedelta

from app.models.fisheries import (
    FishingVessel, FishingTrip, CatchRecord, FishingQuota, MarketPrice,
    VesselType, FishingMethod, CatchStatus
)
from app.schemas.fisheries import (
    FishingVesselCreate, FishingVesselUpdate,
    FishingTripCreate, FishingTripUpdate,
    CatchRecordCreate, CatchRecordUpdate,
    FishingQuotaCreate, FishingQuotaUpdate,
    MarketPriceCreate, MarketPriceUpdate
)


class FisheriesCRUD:
    """CRUD operations for fisheries models."""

    # Fishing Vessel operations
    def create_vessel(self, db: Session, vessel_data: FishingVesselCreate) -> FishingVessel:
        """Create a new fishing vessel."""
        db_vessel = FishingVessel(**vessel_data.dict())
        db.add(db_vessel)
        db.commit()
        db.refresh(db_vessel)
        return db_vessel

    def get_vessel(self, db: Session, vessel_id: int) -> Optional[FishingVessel]:
        """Get vessel by ID."""
        return db.query(FishingVessel).filter(FishingVessel.id == vessel_id).first()

    def get_vessel_by_registration(self, db: Session, registration_number: str) -> Optional[FishingVessel]:
        """Get vessel by registration number."""
        return db.query(FishingVessel).filter(
            FishingVessel.registration_number == registration_number
        ).first()

    def get_vessels(self, db: Session, skip: int = 0, limit: int = 100, 
                   vessel_type: Optional[VesselType] = None,
                   home_port: Optional[str] = None,
                   is_active: Optional[bool] = None) -> List[FishingVessel]:
        """Get vessels with optional filtering."""
        query = db.query(FishingVessel)
        
        if vessel_type:
            query = query.filter(FishingVessel.vessel_type == vessel_type)
        if home_port:
            query = query.filter(FishingVessel.home_port.ilike(f"%{home_port}%"))
        if is_active is not None:
            query = query.filter(FishingVessel.is_active == is_active)
            
        return query.offset(skip).limit(limit).all()

    def update_vessel(self, db: Session, vessel_id: int, 
                     vessel_update: FishingVesselUpdate) -> Optional[FishingVessel]:
        """Update vessel information."""
        db_vessel = self.get_vessel(db, vessel_id)
        if not db_vessel:
            return None
            
        update_data = vessel_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_vessel, field, value)
            
        db.commit()
        db.refresh(db_vessel)
        return db_vessel

    def delete_vessel(self, db: Session, vessel_id: int) -> bool:
        """Soft delete vessel by setting is_active to False."""
        db_vessel = self.get_vessel(db, vessel_id)
        if not db_vessel:
            return False
            
        db_vessel.is_active = False
        db.commit()
        return True

    # Fishing Trip operations
    def create_trip(self, db: Session, trip_data: FishingTripCreate) -> FishingTrip:
        """Create a new fishing trip."""
        db_trip = FishingTrip(**trip_data.dict())
        db.add(db_trip)
        db.commit()
        db.refresh(db_trip)
        return db_trip

    def get_trip(self, db: Session, trip_id: int) -> Optional[FishingTrip]:
        """Get trip by ID."""
        return db.query(FishingTrip).filter(FishingTrip.id == trip_id).first()

    def get_trips(self, db: Session, skip: int = 0, limit: int = 100,
                 vessel_id: Optional[int] = None,
                 is_completed: Optional[bool] = None,
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None) -> List[FishingTrip]:
        """Get trips with optional filtering."""
        query = db.query(FishingTrip)
        
        if vessel_id:
            query = query.filter(FishingTrip.vessel_id == vessel_id)
        if is_completed is not None:
            query = query.filter(FishingTrip.is_completed == is_completed)
        if start_date:
            query = query.filter(FishingTrip.departure_date >= start_date)
        if end_date:
            query = query.filter(FishingTrip.departure_date <= end_date)
            
        return query.order_by(desc(FishingTrip.departure_date)).offset(skip).limit(limit).all()

    def update_trip(self, db: Session, trip_id: int, 
                   trip_update: FishingTripUpdate) -> Optional[FishingTrip]:
        """Update trip information."""
        db_trip = self.get_trip(db, trip_id)
        if not db_trip:
            return None
            
        update_data = trip_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_trip, field, value)
            
        db.commit()
        db.refresh(db_trip)
        return db_trip

    def complete_trip(self, db: Session, trip_id: int) -> Optional[FishingTrip]:
        """Mark trip as completed."""
        db_trip = self.get_trip(db, trip_id)
        if not db_trip:
            return None
            
        db_trip.is_completed = True
        if not db_trip.return_date:
            db_trip.return_date = datetime.utcnow()
            
        db.commit()
        db.refresh(db_trip)
        return db_trip

    # Catch Record operations
    def create_catch_record(self, db: Session, catch_data: CatchRecordCreate) -> CatchRecord:
        """Create a new catch record."""
        db_catch = CatchRecord(**catch_data.dict())
        db.add(db_catch)
        db.commit()
        db.refresh(db_catch)
        
        # Update trip total catch weight if trip is specified
        if db_catch.trip_id:
            self._update_trip_catch_weight(db, db_catch.trip_id)
            
        return db_catch

    def get_catch_record(self, db: Session, catch_id: int) -> Optional[CatchRecord]:
        """Get catch record by ID."""
        return db.query(CatchRecord).filter(CatchRecord.id == catch_id).first()

    def get_catch_records(self, db: Session, skip: int = 0, limit: int = 100,
                         vessel_id: Optional[int] = None,
                         species_id: Optional[int] = None,
                         trip_id: Optional[int] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         fishing_area: Optional[str] = None) -> Tuple[List[CatchRecord], int]:
        """Get catch records with optional filtering and pagination."""
        query = db.query(CatchRecord)
        
        if vessel_id:
            query = query.filter(CatchRecord.vessel_id == vessel_id)
        if species_id:
            query = query.filter(CatchRecord.species_id == species_id)
        if trip_id:
            query = query.filter(CatchRecord.trip_id == trip_id)
        if start_date:
            query = query.filter(CatchRecord.catch_date >= start_date)
        if end_date:
            query = query.filter(CatchRecord.catch_date <= end_date)
        if fishing_area:
            query = query.filter(CatchRecord.fishing_area.ilike(f"%{fishing_area}%"))
            
        total = query.count()
        catches = query.order_by(desc(CatchRecord.catch_date)).offset(skip).limit(limit).all()
        
        return catches, total

    def update_catch_record(self, db: Session, catch_id: int, 
                          catch_update: CatchRecordUpdate) -> Optional[CatchRecord]:
        """Update catch record."""
        db_catch = self.get_catch_record(db, catch_id)
        if not db_catch:
            return None
            
        old_trip_id = db_catch.trip_id
        update_data = catch_update.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            setattr(db_catch, field, value)
            
        db.commit()
        db.refresh(db_catch)
        
        # Update trip catch weights if trip changed
        if old_trip_id != db_catch.trip_id:
            if old_trip_id:
                self._update_trip_catch_weight(db, old_trip_id)
            if db_catch.trip_id:
                self._update_trip_catch_weight(db, db_catch.trip_id)
        elif db_catch.trip_id:
            self._update_trip_catch_weight(db, db_catch.trip_id)
            
        return db_catch

    def delete_catch_record(self, db: Session, catch_id: int) -> bool:
        """Delete catch record."""
        db_catch = self.get_catch_record(db, catch_id)
        if not db_catch:
            return False
            
        trip_id = db_catch.trip_id
        db.delete(db_catch)
        db.commit()
        
        # Update trip catch weight
        if trip_id:
            self._update_trip_catch_weight(db, trip_id)
            
        return True

    # Quota operations
    def create_quota(self, db: Session, quota_data: FishingQuotaCreate) -> FishingQuota:
        """Create a new fishing quota."""
        quota_dict = quota_data.dict()
        quota_dict['remaining_quota'] = quota_dict['allocated_quota']
        
        db_quota = FishingQuota(**quota_dict)
        db.add(db_quota)
        db.commit()
        db.refresh(db_quota)
        return db_quota

    def get_quota(self, db: Session, quota_id: int) -> Optional[FishingQuota]:
        """Get quota by ID."""
        return db.query(FishingQuota).filter(FishingQuota.id == quota_id).first()

    def get_quotas(self, db: Session, vessel_id: Optional[int] = None, 
                  species_id: Optional[int] = None, quota_year: Optional[int] = None,
                  is_active: Optional[bool] = True) -> List[FishingQuota]:
        """Get quotas with filtering."""
        query = db.query(FishingQuota)
        
        if vessel_id:
            query = query.filter(FishingQuota.vessel_id == vessel_id)
        if species_id:
            query = query.filter(FishingQuota.species_id == species_id)
        if quota_year:
            query = query.filter(FishingQuota.quota_year == quota_year)
        if is_active is not None:
            query = query.filter(FishingQuota.is_active == is_active)
            
        return query.all()

    def update_quota_usage(self, db: Session, vessel_id: int, species_id: int, 
                          catch_weight: float) -> bool:
        """Update quota usage when catch is recorded."""
        current_year = datetime.utcnow().year
        quota = db.query(FishingQuota).filter(
            and_(
                FishingQuota.vessel_id == vessel_id,
                FishingQuota.species_id == species_id,
                FishingQuota.quota_year == current_year,
                FishingQuota.is_active == True
            )
        ).first()
        
        if quota:
            quota.used_quota += catch_weight
            quota.remaining_quota = quota.allocated_quota - quota.used_quota
            db.commit()
            return True
        return False

    # Market Price operations
    def create_market_price(self, db: Session, price_data: MarketPriceCreate) -> MarketPrice:
        """Create a new market price record."""
        db_price = MarketPrice(**price_data.dict())
        db.add(db_price)
        db.commit()
        db.refresh(db_price)
        return db_price

    def get_market_prices(self, db: Session, species_id: Optional[int] = None,
                         market_location: Optional[str] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> List[MarketPrice]:
        """Get market prices with filtering."""
        query = db.query(MarketPrice)
        
        if species_id:
            query = query.filter(MarketPrice.species_id == species_id)
        if market_location:
            query = query.filter(MarketPrice.market_location.ilike(f"%{market_location}%"))
        if start_date:
            query = query.filter(MarketPrice.price_date >= start_date)
        if end_date:
            query = query.filter(MarketPrice.price_date <= end_date)
            
        return query.order_by(desc(MarketPrice.price_date)).all()

    # Statistics and Analytics
    def get_fisheries_stats(self, db: Session) -> dict:
        """Get comprehensive fisheries statistics."""
        # Basic counts
        total_vessels = db.query(func.count(FishingVessel.id)).scalar()
        active_vessels = db.query(func.count(FishingVessel.id)).filter(
            FishingVessel.is_active == True
        ).scalar()
        
        total_trips = db.query(func.count(FishingTrip.id)).scalar()
        completed_trips = db.query(func.count(FishingTrip.id)).filter(
            FishingTrip.is_completed == True
        ).scalar()
        
        total_catch_records = db.query(func.count(CatchRecord.id)).scalar()
        total_catch_weight = db.query(func.sum(CatchRecord.catch_weight)).scalar() or 0
        
        # Average catch per trip
        avg_catch_per_trip = 0
        if completed_trips > 0:
            avg_catch_per_trip = total_catch_weight / completed_trips

        # Top species by catch weight
        top_species = db.query(
            CatchRecord.species_id,
            func.sum(CatchRecord.catch_weight).label('total_weight'),
            func.count(CatchRecord.id).label('record_count')
        ).group_by(
            CatchRecord.species_id
        ).order_by(
            desc('total_weight')
        ).limit(10).all()

        # Top fishing areas
        top_areas = db.query(
            CatchRecord.fishing_area,
            func.sum(CatchRecord.catch_weight).label('total_weight'),
            func.count(CatchRecord.id).label('record_count')
        ).filter(
            CatchRecord.fishing_area.isnot(None)
        ).group_by(
            CatchRecord.fishing_area
        ).order_by(
            desc('total_weight')
        ).limit(10).all()

        return {
            'total_vessels': total_vessels,
            'active_vessels': active_vessels,
            'total_trips': total_trips,
            'completed_trips': completed_trips,
            'total_catch_records': total_catch_records,
            'total_catch_weight': float(total_catch_weight),
            'average_catch_per_trip': float(avg_catch_per_trip),
            'top_species': [
                {
                    'species_id': item.species_id,
                    'total_weight': float(item.total_weight),
                    'record_count': item.record_count
                }
                for item in top_species
            ],
            'top_fishing_areas': [
                {
                    'fishing_area': item.fishing_area,
                    'total_weight': float(item.total_weight),
                    'record_count': item.record_count
                }
                for item in top_areas
            ]
        }

    def get_catch_trends(self, db: Session, days: int = 30) -> List[dict]:
        """Get catch trends over specified days."""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        trends = db.query(
            func.date(CatchRecord.catch_date).label('date'),
            func.sum(CatchRecord.catch_weight).label('total_weight'),
            func.count(CatchRecord.id).label('record_count')
        ).filter(
            CatchRecord.catch_date >= start_date
        ).group_by(
            func.date(CatchRecord.catch_date)
        ).order_by('date').all()
        
        return [
            {
                'date': trend.date.isoformat(),
                'total_weight': float(trend.total_weight),
                'record_count': trend.record_count
            }
            for trend in trends
        ]

    # Helper methods
    def _update_trip_catch_weight(self, db: Session, trip_id: int):
        """Update total catch weight for a trip."""
        total_weight = db.query(func.sum(CatchRecord.catch_weight)).filter(
            CatchRecord.trip_id == trip_id
        ).scalar() or 0
        
        trip = db.query(FishingTrip).filter(FishingTrip.id == trip_id).first()
        if trip:
            trip.total_catch_weight = float(total_weight)
            db.commit()


# Global instance
fisheries_crud = FisheriesCRUD()