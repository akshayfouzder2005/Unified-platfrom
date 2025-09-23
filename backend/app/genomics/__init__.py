"""
🧬 Genomics Analysis Package

Phase 2 - Spatial & Predictive Analytics
Ocean-Bio Marine Data Platform

This package provides comprehensive genomic analysis capabilities:
- DNA sequence processing and quality control
- Phylogenetic analysis and tree construction
- Biodiversity metrics and diversity calculations
- Advanced taxonomic classification
- Comparative genomic analysis
- Population genetics and structure analysis

Author: Ocean-Bio Development Team
Version: 2.0.0
"""

from .sequence_processor import SequenceProcessor
from .phylogenetic_analysis import PhylogeneticAnalyzer
from .diversity_calculator import DiversityCalculator
from .taxonomic_classifier import TaxonomicClassifier
from .comparative_analysis import ComparativeAnalyzer

__all__ = [
    'SequenceProcessor',
    'PhylogeneticAnalyzer', 
    'DiversityCalculator',
    'TaxonomicClassifier',
    'ComparativeAnalyzer'
]

__version__ = "2.0.0"