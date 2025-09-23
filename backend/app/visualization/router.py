from typing import Optional, List
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from app.core.database import get_db
from app.core.dependencies import require_read_access
from app.models.user import User
from app.models.taxonomy import TaxonomicUnit
from app.models.fisheries import CatchRecord, FishingVessel, FishingTrip
from app.models.oceanographic import OceanographicMeasurement
from app.models.edna import EdnaSample, EdnaDetection

router = APIRouter()


@router.get("/dashboard-stats")
async def get_dashboard_statistics(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_read_access)
):
    """Get comprehensive dashboard statistics."""
    # Basic counts
    total_species = db.query(func.count(TaxonomicUnit.id)).scalar() or 0
    total_vessels = db.query(func.count(FishingVessel.id)).scalar() or 0
    total_catches = db.query(func.count(CatchRecord.id)).scalar() or 0
    total_trips = db.query(func.count(FishingTrip.id)).scalar() or 0
    
    # Recent activity (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    recent_catches = db.query(func.count(CatchRecord.id)).filter(
        CatchRecord.catch_date >= thirty_days_ago
    ).scalar() or 0
    
    return {
        "overview": {
            "total_species": total_species,
            "total_vessels": total_vessels,
            "total_catches": total_catches,
            "total_trips": total_trips,
            "recent_catches": recent_catches
        }
    }


@router.get("/species-distribution")
async def get_species_distribution(
    limit: int = Query(10, le=50),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_read_access)
):
    """Get species distribution data for charts."""
    # Top species by catch weight
    top_species = db.query(
        TaxonomicUnit.scientific_name,
        func.sum(CatchRecord.catch_weight).label('total_weight'),
        func.count(CatchRecord.id).label('catch_count')
    ).join(
        CatchRecord, CatchRecord.species_id == TaxonomicUnit.id
    ).group_by(
        TaxonomicUnit.id, TaxonomicUnit.scientific_name
    ).order_by(
        desc('total_weight')
    ).limit(limit).all()
    
    return {
        "chart_type": "pie",
        "title": "Species Distribution by Catch Weight",
        "data": [
            {
                "species": species.scientific_name,
                "weight": float(species.total_weight),
                "count": species.catch_count
            }
            for species in top_species
        ]
    }


@router.get("/catch-trends")
async def get_catch_trends(
    days: int = Query(30, ge=7, le=365),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_read_access)
):
    """Get catch trends over time."""
    start_date = datetime.utcnow() - timedelta(days=days)
    
    daily_catches = db.query(
        func.date(CatchRecord.catch_date).label('date'),
        func.sum(CatchRecord.catch_weight).label('total_weight'),
        func.count(CatchRecord.id).label('catch_count')
    ).filter(
        CatchRecord.catch_date >= start_date
    ).group_by(
        func.date(CatchRecord.catch_date)
    ).order_by('date').all()
    
    return {
        "chart_type": "line",
        "title": f"Catch Trends - Last {days} Days",
        "data": [
            {
                "date": catch.date.isoformat(),
                "weight": float(catch.total_weight or 0),
                "count": catch.catch_count
            }
            for catch in daily_catches
        ]
    }


@router.get("/fishing-areas")
async def get_fishing_areas_data(
    limit: int = Query(15, le=50),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_read_access)
):
    """Get fishing areas distribution."""
    areas_data = db.query(
        CatchRecord.fishing_area,
        func.sum(CatchRecord.catch_weight).label('total_weight'),
        func.count(CatchRecord.id).label('catch_count')
    ).filter(
        CatchRecord.fishing_area.isnot(None)
    ).group_by(
        CatchRecord.fishing_area
    ).order_by(
        desc('total_weight')
    ).limit(limit).all()
    
    return {
        "chart_type": "bar",
        "title": "Top Fishing Areas by Catch Weight",
        "data": [
            {
                "area": area.fishing_area,
                "weight": float(area.total_weight),
                "count": area.catch_count
            }
            for area in areas_data
        ]
    }


@router.get("/vessel-performance")
async def get_vessel_performance(
    limit: int = Query(10, le=30),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_read_access)
):
    """Get vessel performance metrics."""
    vessel_performance = db.query(
        FishingVessel.vessel_name,
        FishingVessel.registration_number,
        func.sum(CatchRecord.catch_weight).label('total_catch'),
        func.count(FishingTrip.id).label('trip_count'),
        func.avg(CatchRecord.catch_weight).label('avg_catch')
    ).join(
        CatchRecord, CatchRecord.vessel_id == FishingVessel.id
    ).join(
        FishingTrip, FishingTrip.vessel_id == FishingVessel.id
    ).group_by(
        FishingVessel.id, FishingVessel.vessel_name, FishingVessel.registration_number
    ).order_by(
        desc('total_catch')
    ).limit(limit).all()
    
    return {
        "chart_type": "bar",
        "title": "Top Performing Vessels",
        "data": [
            {
                "vessel": f"{vessel.vessel_name} ({vessel.registration_number})",
                "total_catch": float(vessel.total_catch or 0),
                "trip_count": vessel.trip_count,
                "avg_catch": float(vessel.avg_catch or 0)
            }
            for vessel in vessel_performance
        ]
    }


@router.get("/monthly-summary")
async def get_monthly_summary(
    months: int = Query(12, ge=3, le=24),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_read_access)
):
    """Get monthly catch summary."""
    start_date = datetime.utcnow() - timedelta(days=months*30)
    
    monthly_data = db.query(
        func.extract('year', CatchRecord.catch_date).label('year'),
        func.extract('month', CatchRecord.catch_date).label('month'),
        func.sum(CatchRecord.catch_weight).label('total_weight'),
        func.count(CatchRecord.id).label('catch_count'),
        func.count(func.distinct(CatchRecord.species_id)).label('species_count')
    ).filter(
        CatchRecord.catch_date >= start_date
    ).group_by(
        func.extract('year', CatchRecord.catch_date),
        func.extract('month', CatchRecord.catch_date)
    ).order_by('year', 'month').all()
    
    return {
        "chart_type": "line",
        "title": f"Monthly Catch Summary - Last {months} Months",
        "data": [
            {
                "period": f"{int(record.year)}-{int(record.month):02d}",
                "weight": float(record.total_weight or 0),
                "count": record.catch_count,
                "species_count": record.species_count
            }
            for record in monthly_data
        ]
    }


@router.get("/biodiversity-metrics")
async def get_biodiversity_metrics(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_read_access)
):
    """Get biodiversity and ecosystem health metrics."""
    # Species diversity by taxonomic level
    kingdom_count = db.query(func.count(func.distinct(TaxonomicUnit.kingdom))).scalar() or 0
    phylum_count = db.query(func.count(func.distinct(TaxonomicUnit.phylum))).scalar() or 0
    class_count = db.query(func.count(func.distinct(TaxonomicUnit.class_name))).scalar() or 0
    order_count = db.query(func.count(func.distinct(TaxonomicUnit.order_name))).scalar() or 0
    family_count = db.query(func.count(func.distinct(TaxonomicUnit.family))).scalar() or 0
    genus_count = db.query(func.count(func.distinct(TaxonomicUnit.genus))).scalar() or 0
    species_count = db.query(func.count(TaxonomicUnit.id)).scalar() or 0
    
    return {
        "diversity_hierarchy": {
            "kingdoms": kingdom_count,
            "phyla": phylum_count,
            "classes": class_count,
            "orders": order_count,
            "families": family_count,
            "genera": genus_count,
            "species": species_count
        },
        "chart_data": [
            {"level": "Kingdom", "count": kingdom_count},
            {"level": "Phylum", "count": phylum_count},
            {"level": "Class", "count": class_count},
            {"level": "Order", "count": order_count},
            {"level": "Family", "count": family_count},
            {"level": "Genus", "count": genus_count},
            {"level": "Species", "count": species_count}
        ]
    }


@router.get("/heatmap-data")
async def get_heatmap_data(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_read_access)
):
    """Get geographic heatmap data for catch locations."""
    # Get catch records with coordinates
    location_data = db.query(
        CatchRecord.coordinates,
        func.sum(CatchRecord.catch_weight).label('total_weight'),
        func.count(CatchRecord.id).label('catch_count')
    ).filter(
        CatchRecord.coordinates.isnot(None)
    ).group_by(
        CatchRecord.coordinates
    ).all()
    
    heatmap_points = []
    for record in location_data:
        try:
            # Parse coordinates (assuming "lat,lon" format)
            lat, lon = map(float, record.coordinates.split(','))
            heatmap_points.append({
                "lat": lat,
                "lon": lon,
                "weight": float(record.total_weight),
                "count": record.catch_count,
                "intensity": min(float(record.total_weight) / 1000, 1.0)  # Normalize for visualization
            })
        except (ValueError, AttributeError):
            continue
    
    return {
        "chart_type": "heatmap",
        "title": "Fishing Activity Heatmap",
        "data": heatmap_points
    }


# Legacy endpoint for backward compatibility
@router.get("/trends/ocean")
def ocean_trends():
    """Return sample oceanographic trend data."""
    return {
        "variable": "Sea Surface Temperature",
        "units": "°C",
        "series": [
            {"date": "2025-01-01", "value": 20.1},
            {"date": "2025-02-01", "value": 20.4},
            {"date": "2025-03-01", "value": 20.2},
            {"date": "2025-04-01", "value": 21.1},
            {"date": "2025-05-01", "value": 22.3},
            {"date": "2025-06-01", "value": 23.8}
        ],
    }
