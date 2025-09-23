# Import all models to ensure they are registered with SQLAlchemy
from .base import BaseModel
from .taxonomy import (
    TaxonomicRank,
    TaxonomicUnit, 
    TaxonomicSynonym,
    TaxonomicReference
)
from .oceanographic import (
    OceanographicStation,
    OceanographicParameter,
    OceanographicMeasurement,
    OceanographicDataset,
    OceanographicAlert
)
from .edna import (
    EdnaSample,
    EdnaExtraction,
    PcrReaction,
    DnaSequence,
    TaxonomicAssignment,
    EdnaDetection,
    EdnaStudy
)
from .otolith import (
    OtolithSpecimen,
    OtolithMeasurement,
    OtolithImage,
    OtolithReference,
    OtolithClassification,
    OtolithStudy
)
from .user import (
    User,
    UserRole
)
from .fisheries import (
    FishingVessel,
    FishingTrip,
    CatchRecord,
    FishingQuota,
    MarketPrice,
    VesselType,
    FishingMethod,
    CatchStatus
)

# Export all models for easy importing
__all__ = [
    "BaseModel",
    # Taxonomy
    "TaxonomicRank",
    "TaxonomicUnit", 
    "TaxonomicSynonym",
    "TaxonomicReference",
    # Oceanographic
    "OceanographicStation",
    "OceanographicParameter",
    "OceanographicMeasurement",
    "OceanographicDataset",
    "OceanographicAlert",
    # eDNA
    "EdnaSample",
    "EdnaExtraction",
    "PcrReaction",
    "DnaSequence",
    "TaxonomicAssignment",
    "EdnaDetection",
    "EdnaStudy",
    # Otolith
    "OtolithSpecimen",
    "OtolithMeasurement",
    "OtolithImage",
    "OtolithReference",
    "OtolithClassification",
    "OtolithStudy",
    # Authentication
    "User",
    "UserRole",
    # Fisheries
    "FishingVessel",
    "FishingTrip",
    "CatchRecord",
    "FishingQuota",
    "MarketPrice",
    "VesselType",
    "FishingMethod",
    "CatchStatus"
]
