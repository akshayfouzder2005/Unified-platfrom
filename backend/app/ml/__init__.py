"""
Machine Learning module for Ocean-Bio Platform
Provides AI/ML capabilities for species identification and analysis
"""
from .species_identifier import SpeciesIdentifier
from .model_manager import ModelManager
from .image_preprocessor import ImagePreprocessor

__all__ = ["SpeciesIdentifier", "ModelManager", "ImagePreprocessor"]