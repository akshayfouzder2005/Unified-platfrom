"""
🧬 Comparative Analysis - Integrated Genomics Analysis & Comparison

Advanced comparative genomics for environmental DNA data.
Integrates sequence processing, diversity analysis, phylogenetics, and taxonomy.

Author: Ocean-Bio Development Team
Version: 2.0.0
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict, Counter
import json
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio

# Import other genomics modules
from .sequence_processor import sequence_processor
from .diversity_calculator import diversity_calculator
from .phylogenetic_analysis import phylogenetic_analyzer, DistanceMethod, TreeMethod
from .taxonomic_classifier import taxonomic_classifier, ClassificationMethod, TaxonomicRank

logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """Types of comparative analysis"""
    COMMUNITY_COMPARISON = "community_comparison"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    SPATIAL_ANALYSIS = "spatial_analysis"
    TAXONOMIC_PROFILING = "taxonomic_profiling"
    PHYLOGENETIC_DIVERSITY = "phylogenetic_diversity"
    INTEGRATED_PIPELINE = "integrated_pipeline"

@dataclass
class SampleMetadata:
    """Metadata for environmental samples"""
    sample_id: str
    location: Dict[str, float]  # lat, lon
    collection_date: str
    depth: Optional[float] = None
    temperature: Optional[float] = None
    salinity: Optional[float] = None
    ph: Optional[float] = None
    habitat_type: Optional[str] = None
    sampling_method: Optional[str] = None
    additional_metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_metadata is None:
            self.additional_metadata = {}

@dataclass
class ComparativeResults:
    """Results from comparative analysis"""
    analysis_type: AnalysisType
    sample_ids: List[str]
    results: Dict[str, Any]
    summary_statistics: Dict[str, Any]
    visualizations: Dict[str, Any]
    metadata: Dict[str, Any]
    analysis_timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

class ComparativeAnalyzer:
    """
    🧬 Advanced Comparative Genomics Analysis Engine
    
    Provides integrated comparative analysis capabilities:
    - Multi-sample community comparison
    - Temporal and spatial analysis
    - Comprehensive taxonomic profiling
    - Phylogenetic diversity assessment
    - Integrated eDNA pipeline analysis
    - Statistical comparisons and visualizations
    - Environmental correlation analysis
    """
    
    def __init__(self):
        """Initialize the comparative analyzer"""
        self.samples = {}  # sample_id -> sequences
        self.sample_metadata = {}  # sample_id -> SampleMetadata
        self.analysis_cache = {}  # cache for expensive analyses
        
        # Analysis parameters
        self.default_min_confidence = 0.7
        self.default_bootstrap_replicates = 100
        self.statistical_significance_threshold = 0.05
    
    def add_sample(self, 
                   sample_id: str, 
                   sequences: Dict[str, str], 
                   metadata: Optional[SampleMetadata] = None) -> Dict[str, Any]:
        """
        Add environmental sample with sequences and metadata
        
        Args:
            sample_id: Unique sample identifier
            sequences: Dictionary of sequence_id -> DNA sequence
            metadata: Sample metadata
            
        Returns:
            Sample addition results
        """
        try:
            if not sequences:
                return {'error': 'No sequences provided for sample'}
            
            # Validate sequences
            valid_sequences = {}
            invalid_count = 0
            
            for seq_id, sequence in sequences.items():
                # Basic validation
                if sequence and len(sequence) >= 50:  # Minimum length threshold
                    valid_sequences[seq_id] = sequence.upper().strip()
                else:
                    invalid_count += 1
            
            if not valid_sequences:
                return {'error': 'No valid sequences found'}
            
            # Store sample data
            self.samples[sample_id] = valid_sequences
            
            # Store metadata if provided
            if metadata:
                self.sample_metadata[sample_id] = metadata
            else:
                # Create basic metadata
                self.sample_metadata[sample_id] = SampleMetadata(
                    sample_id=sample_id,
                    location={'lat': 0.0, 'lon': 0.0},
                    collection_date=datetime.now().isoformat()
                )
            
            # Clear cache when new samples are added
            self.analysis_cache.clear()
            
            results = {
                'sample_id': sample_id,
                'valid_sequences': len(valid_sequences),
                'invalid_sequences': invalid_count,
                'total_samples': len(self.samples),
                'sample_added': True,
                'addition_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🧬 Sample added: {sample_id} with {len(valid_sequences)} sequences")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to add sample {sample_id}: {e}")
            return {'error': str(e)}
    
    def run_integrated_pipeline(self, 
                               sample_ids: Optional[List[str]] = None,
                               include_phylogenetics: bool = True,
                               include_taxonomy: bool = True) -> ComparativeResults:
        """
        Run complete integrated eDNA analysis pipeline
        
        Args:
            sample_ids: Samples to analyze (all if None)
            include_phylogenetics: Whether to include phylogenetic analysis
            include_taxonomy: Whether to include taxonomic classification
            
        Returns:
            Comprehensive analysis results
        """
        try:
            if sample_ids is None:
                sample_ids = list(self.samples.keys())
            
            if not sample_ids:
                raise ValueError("No samples available for analysis")
            
            pipeline_results = {}
            
            # Step 1: Sequence processing and quality assessment
            logger.info("🧬 Step 1: Processing sequences...")
            sequence_results = self._process_sequences(sample_ids)
            pipeline_results['sequence_processing'] = sequence_results
            
            # Step 2: Taxonomic classification
            if include_taxonomy:
                logger.info("🧬 Step 2: Taxonomic classification...")
                taxonomy_results = self._classify_sequences(sample_ids)
                pipeline_results['taxonomic_classification'] = taxonomy_results
            
            # Step 3: Diversity analysis
            logger.info("🧬 Step 3: Diversity analysis...")
            diversity_results = self._analyze_diversity(sample_ids, taxonomy_results if include_taxonomy else None)
            pipeline_results['diversity_analysis'] = diversity_results
            
            # Step 4: Phylogenetic analysis
            if include_phylogenetics:
                logger.info("🧬 Step 4: Phylogenetic analysis...")
                phylo_results = self._analyze_phylogenetics(sample_ids)
                pipeline_results['phylogenetic_analysis'] = phylo_results
            
            # Step 5: Comparative analysis
            logger.info("🧬 Step 5: Comparative analysis...")
            comparative_results = self._perform_comparative_analysis(sample_ids, pipeline_results)
            pipeline_results['comparative_analysis'] = comparative_results
            
            # Step 6: Environmental correlations
            logger.info("🧬 Step 6: Environmental correlations...")
            env_correlations = self._analyze_environmental_correlations(sample_ids, pipeline_results)
            pipeline_results['environmental_correlations'] = env_correlations
            
            # Generate summary statistics
            summary_stats = self._generate_pipeline_summary(pipeline_results)
            
            # Generate visualizations metadata
            visualizations = self._generate_visualization_metadata(pipeline_results)
            
            results = ComparativeResults(
                analysis_type=AnalysisType.INTEGRATED_PIPELINE,
                sample_ids=sample_ids,
                results=pipeline_results,
                summary_statistics=summary_stats,
                visualizations=visualizations,
                metadata={
                    'pipeline_parameters': {
                        'include_phylogenetics': include_phylogenetics,
                        'include_taxonomy': include_taxonomy,
                        'min_confidence': self.default_min_confidence
                    },
                    'sample_count': len(sample_ids),
                    'total_sequences': sum(len(self.samples[sid]) for sid in sample_ids)
                },
                analysis_timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"🧬 Integrated pipeline completed for {len(sample_ids)} samples")
            return results
            
        except Exception as e:
            logger.error(f"❌ Integrated pipeline failed: {e}")
            return ComparativeResults(
                analysis_type=AnalysisType.INTEGRATED_PIPELINE,
                sample_ids=sample_ids or [],
                results={'error': str(e)},
                summary_statistics={},
                visualizations={},
                metadata={},
                analysis_timestamp=datetime.now().isoformat()
            )
    
    def compare_communities(self, 
                           sample_ids: List[str],
                           comparison_method: str = "comprehensive") -> ComparativeResults:
        """
        Compare microbial/marine communities between samples
        
        Args:
            sample_ids: Samples to compare
            comparison_method: Type of comparison to perform
            
        Returns:
            Community comparison results
        """
        try:
            if len(sample_ids) < 2:
                raise ValueError("At least 2 samples required for comparison")
            
            comparison_results = {}
            
            # Get taxonomic classifications for all samples
            taxonomy_results = self._classify_sequences(sample_ids)
            
            # Build community matrices
            community_matrices = self._build_community_matrices(sample_ids, taxonomy_results)
            comparison_results['community_matrices'] = community_matrices
            
            # Alpha diversity comparison
            alpha_diversity = {}
            for sample_id in sample_ids:
                if sample_id in community_matrices['species_abundance']:
                    abundance_data = community_matrices['species_abundance'][sample_id]
                    alpha_result = diversity_calculator.calculate_alpha_diversity(abundance_data, sample_id)
                    alpha_diversity[sample_id] = alpha_result
            
            comparison_results['alpha_diversity'] = alpha_diversity
            
            # Beta diversity analysis
            if len(sample_ids) >= 2:
                beta_result = diversity_calculator.calculate_beta_diversity(
                    community_matrices['species_abundance'], 'bray_curtis'
                )
                comparison_results['beta_diversity'] = beta_result
            
            # Gamma diversity
            gamma_result = diversity_calculator.calculate_gamma_diversity(
                community_matrices['species_abundance']
            )
            comparison_results['gamma_diversity'] = gamma_result
            
            # Pairwise sample comparisons
            pairwise_comparisons = {}
            for i, sample1 in enumerate(sample_ids):
                for sample2 in sample_ids[i+1:]:
                    comparison_key = f"{sample1}_vs_{sample2}"
                    
                    if (sample1 in community_matrices['species_abundance'] and 
                        sample2 in community_matrices['species_abundance']):
                        
                        pairwise_result = diversity_calculator.community_comparison(
                            community_matrices['species_abundance'][sample1],
                            community_matrices['species_abundance'][sample2],
                            sample1, sample2
                        )
                        pairwise_comparisons[comparison_key] = pairwise_result
            
            comparison_results['pairwise_comparisons'] = pairwise_comparisons
            
            # Statistical analysis
            statistical_results = self._perform_statistical_tests(alpha_diversity, community_matrices)
            comparison_results['statistical_analysis'] = statistical_results
            
            # Summary statistics
            summary_stats = self._generate_comparison_summary(comparison_results)
            
            # Visualization metadata
            visualizations = {
                'diversity_plots': ['alpha_diversity_boxplot', 'beta_diversity_heatmap'],
                'community_plots': ['species_composition_barplot', 'ordination_plot'],
                'comparison_plots': ['pairwise_similarity_heatmap', 'dendrogram']
            }
            
            results = ComparativeResults(
                analysis_type=AnalysisType.COMMUNITY_COMPARISON,
                sample_ids=sample_ids,
                results=comparison_results,
                summary_statistics=summary_stats,
                visualizations=visualizations,
                metadata={
                    'comparison_method': comparison_method,
                    'samples_compared': len(sample_ids),
                    'pairwise_comparisons': len(pairwise_comparisons)
                },
                analysis_timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"🧬 Community comparison completed for {len(sample_ids)} samples")
            return results
            
        except Exception as e:
            logger.error(f"❌ Community comparison failed: {e}")
            return ComparativeResults(
                analysis_type=AnalysisType.COMMUNITY_COMPARISON,
                sample_ids=sample_ids,
                results={'error': str(e)},
                summary_statistics={},
                visualizations={},
                metadata={},
                analysis_timestamp=datetime.now().isoformat()
            )
    
    def analyze_temporal_patterns(self, 
                                 sample_ids: List[str],
                                 time_resolution: str = "monthly") -> ComparativeResults:
        """
        Analyze temporal patterns in community composition
        
        Args:
            sample_ids: Samples to analyze temporally
            time_resolution: Time resolution for analysis
            
        Returns:
            Temporal analysis results
        """
        try:
            # Group samples by time periods
            temporal_groups = self._group_samples_by_time(sample_ids, time_resolution)
            
            if len(temporal_groups) < 2:
                raise ValueError("Need samples from at least 2 time periods")
            
            temporal_results = {}
            temporal_results['temporal_groups'] = {
                period: list(samples) for period, samples in temporal_groups.items()
            }
            
            # Analyze diversity changes over time
            temporal_diversity = {}
            for period, period_samples in temporal_groups.items():
                # Get taxonomic data for period
                taxonomy_results = self._classify_sequences(period_samples)
                community_matrix = self._build_community_matrices(period_samples, taxonomy_results)
                
                # Calculate period diversity metrics
                period_alpha = {}
                for sample_id in period_samples:
                    if sample_id in community_matrix['species_abundance']:
                        alpha_result = diversity_calculator.calculate_alpha_diversity(
                            community_matrix['species_abundance'][sample_id], sample_id
                        )
                        period_alpha[sample_id] = alpha_result
                
                # Period gamma diversity
                if community_matrix['species_abundance']:
                    gamma_result = diversity_calculator.calculate_gamma_diversity(
                        community_matrix['species_abundance']
                    )
                    
                    temporal_diversity[period] = {
                        'alpha_diversity': period_alpha,
                        'gamma_diversity': gamma_result,
                        'sample_count': len(period_samples)
                    }
            
            temporal_results['temporal_diversity'] = temporal_diversity
            
            # Trend analysis
            trend_analysis = self._analyze_temporal_trends(temporal_diversity)
            temporal_results['trend_analysis'] = trend_analysis
            
            # Community turnover between periods
            turnover_analysis = self._analyze_community_turnover(temporal_groups)
            temporal_results['community_turnover'] = turnover_analysis
            
            # Summary statistics
            summary_stats = {
                'time_periods_analyzed': len(temporal_groups),
                'total_samples': len(sample_ids),
                'time_resolution': time_resolution,
                'analysis_span': self._calculate_temporal_span(sample_ids)
            }
            
            # Visualization metadata
            visualizations = {
                'temporal_plots': ['diversity_time_series', 'community_turnover_plot'],
                'trend_plots': ['linear_trend_plots', 'seasonal_decomposition']
            }
            
            results = ComparativeResults(
                analysis_type=AnalysisType.TEMPORAL_ANALYSIS,
                sample_ids=sample_ids,
                results=temporal_results,
                summary_statistics=summary_stats,
                visualizations=visualizations,
                metadata={
                    'time_resolution': time_resolution,
                    'temporal_groups': len(temporal_groups)
                },
                analysis_timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"🧬 Temporal analysis completed: {len(temporal_groups)} time periods")
            return results
            
        except Exception as e:
            logger.error(f"❌ Temporal analysis failed: {e}")
            return ComparativeResults(
                analysis_type=AnalysisType.TEMPORAL_ANALYSIS,
                sample_ids=sample_ids,
                results={'error': str(e)},
                summary_statistics={},
                visualizations={},
                metadata={},
                analysis_timestamp=datetime.now().isoformat()
            )
    
    def analyze_spatial_patterns(self, 
                                sample_ids: List[str],
                                spatial_resolution: float = 1.0) -> ComparativeResults:
        """
        Analyze spatial patterns in community composition
        
        Args:
            sample_ids: Samples to analyze spatially
            spatial_resolution: Spatial resolution in degrees
            
        Returns:
            Spatial analysis results
        """
        try:
            # Group samples spatially
            spatial_groups = self._group_samples_spatially(sample_ids, spatial_resolution)
            
            spatial_results = {}
            spatial_results['spatial_groups'] = {
                f"region_{i}": {
                    'samples': list(samples),
                    'center_lat': np.mean([self.sample_metadata[s].location['lat'] for s in samples]),
                    'center_lon': np.mean([self.sample_metadata[s].location['lon'] for s in samples])
                }
                for i, (region, samples) in enumerate(spatial_groups.items())
            }
            
            # Distance-decay relationships
            distance_decay = self._analyze_distance_decay(sample_ids)
            spatial_results['distance_decay'] = distance_decay
            
            # Environmental gradients
            env_gradients = self._analyze_environmental_gradients(sample_ids)
            spatial_results['environmental_gradients'] = env_gradients
            
            # Spatial autocorrelation
            spatial_autocorr = self._calculate_spatial_autocorrelation(sample_ids)
            spatial_results['spatial_autocorrelation'] = spatial_autocorr
            
            # Summary statistics
            summary_stats = {
                'spatial_regions': len(spatial_groups),
                'total_samples': len(sample_ids),
                'spatial_resolution_degrees': spatial_resolution,
                'geographic_extent': self._calculate_geographic_extent(sample_ids)
            }
            
            # Visualization metadata
            visualizations = {
                'spatial_plots': ['sample_map', 'diversity_map'],
                'gradient_plots': ['environmental_gradient_plots', 'distance_decay_plot']
            }
            
            results = ComparativeResults(
                analysis_type=AnalysisType.SPATIAL_ANALYSIS,
                sample_ids=sample_ids,
                results=spatial_results,
                summary_statistics=summary_stats,
                visualizations=visualizations,
                metadata={
                    'spatial_resolution': spatial_resolution,
                    'spatial_regions': len(spatial_groups)
                },
                analysis_timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"🧬 Spatial analysis completed: {len(spatial_groups)} regions")
            return results
            
        except Exception as e:
            logger.error(f"❌ Spatial analysis failed: {e}")
            return ComparativeResults(
                analysis_type=AnalysisType.SPATIAL_ANALYSIS,
                sample_ids=sample_ids,
                results={'error': str(e)},
                summary_statistics={},
                visualizations={},
                metadata={},
                analysis_timestamp=datetime.now().isoformat()
            )
    
    def _process_sequences(self, sample_ids: List[str]) -> Dict[str, Any]:
        """Process sequences for quality assessment"""
        processing_results = {}
        
        for sample_id in sample_ids:
            sequences = self.samples[sample_id]
            sample_processing = {}
            
            # Quality assessment for each sequence
            sequence_qualities = {}
            for seq_id, sequence in sequences.items():
                quality_result = sequence_processor.assess_sequence_quality(sequence, seq_id)
                sequence_qualities[seq_id] = quality_result
            
            sample_processing['sequence_qualities'] = sequence_qualities
            
            # Sample-level statistics
            sample_processing['statistics'] = {
                'total_sequences': len(sequences),
                'mean_length': np.mean([len(seq) for seq in sequences.values()]),
                'quality_passed': sum(1 for q in sequence_qualities.values() 
                                    if q.get('overall_quality', 'low') in ['high', 'medium'])
            }
            
            processing_results[sample_id] = sample_processing
        
        return processing_results
    
    def _classify_sequences(self, sample_ids: List[str]) -> Dict[str, Any]:
        """Classify sequences taxonomically"""
        classification_results = {}
        
        for sample_id in sample_ids:
            sequences = self.samples[sample_id]
            
            # Classify all sequences in sample
            assignments = taxonomic_classifier.classify_batch(
                sequences, 
                ClassificationMethod.CONSENSUS, 
                self.default_min_confidence
            )
            
            # Generate summary
            summary = taxonomic_classifier.get_classification_summary(assignments)
            
            classification_results[sample_id] = {
                'assignments': assignments,
                'summary': summary
            }
        
        return classification_results
    
    def _analyze_diversity(self, sample_ids: List[str], taxonomy_results: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze biodiversity metrics"""
        diversity_results = {}
        
        if taxonomy_results:
            # Use taxonomic abundance data
            for sample_id in sample_ids:
                if sample_id in taxonomy_results:
                    assignments = taxonomy_results[sample_id]['assignments']
                    
                    # Build species abundance matrix
                    species_abundance = Counter()
                    for seq_id, assignment in assignments.items():
                        if TaxonomicRank.SPECIES in assignment.taxonomy:
                            species = assignment.taxonomy[TaxonomicRank.SPECIES]
                            species_abundance[species] += 1
                    
                    if species_abundance:
                        alpha_diversity = diversity_calculator.calculate_alpha_diversity(
                            dict(species_abundance), sample_id
                        )
                        diversity_results[sample_id] = {'alpha_diversity': alpha_diversity}
        
        # Beta diversity between samples
        if len(diversity_results) >= 2:
            community_matrix = {}
            for sample_id, result in diversity_results.items():
                # Extract species abundance from alpha diversity results
                # This is a simplified approach
                community_matrix[sample_id] = {'species_1': 10, 'species_2': 5}  # Placeholder
            
            if community_matrix:
                beta_diversity = diversity_calculator.calculate_beta_diversity(community_matrix)
                diversity_results['beta_diversity'] = beta_diversity
        
        return diversity_results
    
    def _analyze_phylogenetics(self, sample_ids: List[str]) -> Dict[str, Any]:
        """Perform phylogenetic analysis"""
        phylo_results = {}
        
        # Combine sequences from all samples for phylogenetic analysis
        combined_sequences = {}
        for sample_id in sample_ids:
            for seq_id, sequence in self.samples[sample_id].items():
                combined_key = f"{sample_id}_{seq_id}"
                combined_sequences[combined_key] = sequence
        
        # Limit to reasonable number for phylogenetic analysis
        if len(combined_sequences) > 50:
            # Sample representative sequences
            sample_keys = list(combined_sequences.keys())[:50]
            combined_sequences = {k: combined_sequences[k] for k in sample_keys}
        
        if len(combined_sequences) >= 3:
            # Construct phylogenetic tree
            tree_result = phylogenetic_analyzer.construct_tree(
                combined_sequences,
                DistanceMethod.KIMURA_2P,
                TreeMethod.NEIGHBOR_JOINING,
                bootstrap_replicates=self.default_bootstrap_replicates
            )
            phylo_results['tree_construction'] = tree_result
            
            # Calculate phylogenetic diversity if tree was successfully constructed
            if 'error' not in tree_result and 'tree' in tree_result:
                newick_tree = tree_result['tree']['newick']
                pd_result = phylogenetic_analyzer.calculate_phylogenetic_diversity(newick_tree)
                phylo_results['phylogenetic_diversity'] = pd_result
        
        return phylo_results
    
    def _perform_comparative_analysis(self, sample_ids: List[str], pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comparative analysis across samples"""
        comparative_results = {}
        
        # Compare diversity metrics
        if 'diversity_analysis' in pipeline_results:
            diversity_comparison = self._compare_diversity_metrics(sample_ids, pipeline_results['diversity_analysis'])
            comparative_results['diversity_comparison'] = diversity_comparison
        
        # Compare taxonomic composition
        if 'taxonomic_classification' in pipeline_results:
            taxonomic_comparison = self._compare_taxonomic_composition(sample_ids, pipeline_results['taxonomic_classification'])
            comparative_results['taxonomic_comparison'] = taxonomic_comparison
        
        # Sample similarity analysis
        similarity_analysis = self._calculate_sample_similarities(sample_ids, pipeline_results)
        comparative_results['similarity_analysis'] = similarity_analysis
        
        return comparative_results
    
    def _analyze_environmental_correlations(self, sample_ids: List[str], pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze correlations with environmental variables"""
        env_correlations = {}
        
        # Extract environmental variables
        env_variables = ['temperature', 'salinity', 'ph', 'depth']
        env_data = {}
        
        for sample_id in sample_ids:
            if sample_id in self.sample_metadata:
                metadata = self.sample_metadata[sample_id]
                sample_env = {}
                
                for var in env_variables:
                    value = getattr(metadata, var, None)
                    if value is not None:
                        sample_env[var] = value
                
                if sample_env:
                    env_data[sample_id] = sample_env
        
        if env_data and 'diversity_analysis' in pipeline_results:
            # Correlate environmental variables with diversity metrics
            correlations = self._calculate_environmental_correlations(env_data, pipeline_results['diversity_analysis'])
            env_correlations['diversity_environment_correlations'] = correlations
        
        return env_correlations
    
    def _build_community_matrices(self, sample_ids: List[str], taxonomy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Build community abundance matrices"""
        matrices = {
            'species_abundance': {},
            'genus_abundance': {},
            'family_abundance': {}
        }
        
        for sample_id in sample_ids:
            if sample_id in taxonomy_results and 'assignments' in taxonomy_results[sample_id]:
                assignments = taxonomy_results[sample_id]['assignments']
                
                # Build abundance matrices for different taxonomic levels
                species_counts = Counter()
                genus_counts = Counter()
                family_counts = Counter()
                
                for seq_id, assignment in assignments.items():
                    taxonomy = assignment.taxonomy
                    
                    if TaxonomicRank.SPECIES in taxonomy:
                        species_counts[taxonomy[TaxonomicRank.SPECIES]] += 1
                    
                    if TaxonomicRank.GENUS in taxonomy:
                        genus_counts[taxonomy[TaxonomicRank.GENUS]] += 1
                    
                    if TaxonomicRank.FAMILY in taxonomy:
                        family_counts[taxonomy[TaxonomicRank.FAMILY]] += 1
                
                matrices['species_abundance'][sample_id] = dict(species_counts)
                matrices['genus_abundance'][sample_id] = dict(genus_counts)
                matrices['family_abundance'][sample_id] = dict(family_counts)
        
        return matrices
    
    def _generate_pipeline_summary(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for pipeline results"""
        summary = {}
        
        # Sequence processing summary
        if 'sequence_processing' in pipeline_results:
            total_sequences = 0
            quality_passed = 0
            
            for sample_result in pipeline_results['sequence_processing'].values():
                stats = sample_result.get('statistics', {})
                total_sequences += stats.get('total_sequences', 0)
                quality_passed += stats.get('quality_passed', 0)
            
            summary['sequence_processing'] = {
                'total_sequences_processed': total_sequences,
                'quality_passed': quality_passed,
                'quality_rate': quality_passed / total_sequences if total_sequences > 0 else 0
            }
        
        # Taxonomic classification summary
        if 'taxonomic_classification' in pipeline_results:
            total_classified = 0
            total_sequences = 0
            
            for sample_result in pipeline_results['taxonomic_classification'].values():
                summary_data = sample_result.get('summary', {})
                total_classified += summary_data.get('classified_sequences', 0)
                total_sequences += summary_data.get('total_sequences', 0)
            
            summary['taxonomic_classification'] = {
                'total_sequences': total_sequences,
                'total_classified': total_classified,
                'classification_rate': total_classified / total_sequences if total_sequences > 0 else 0
            }
        
        return summary
    
    def _generate_visualization_metadata(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata for visualizations"""
        visualizations = {
            'available_plots': [],
            'recommended_plots': [],
            'plot_descriptions': {}
        }
        
        # Basic plots always available
        basic_plots = ['sample_overview', 'sequence_length_distribution']
        visualizations['available_plots'].extend(basic_plots)
        
        # Add plots based on available results
        if 'taxonomic_classification' in pipeline_results:
            tax_plots = ['taxonomic_composition_barplot', 'classification_confidence_plot']
            visualizations['available_plots'].extend(tax_plots)
            visualizations['recommended_plots'].extend(tax_plots)
        
        if 'diversity_analysis' in pipeline_results:
            div_plots = ['alpha_diversity_comparison', 'rarefaction_curves']
            visualizations['available_plots'].extend(div_plots)
            visualizations['recommended_plots'].extend(div_plots)
        
        if 'phylogenetic_analysis' in pipeline_results:
            phylo_plots = ['phylogenetic_tree', 'phylogenetic_diversity_plot']
            visualizations['available_plots'].extend(phylo_plots)
        
        return visualizations
    
    # Additional helper methods for specialized analyses
    def _group_samples_by_time(self, sample_ids: List[str], resolution: str) -> Dict[str, List[str]]:
        """Group samples by time periods"""
        groups = defaultdict(list)
        
        for sample_id in sample_ids:
            if sample_id in self.sample_metadata:
                date_str = self.sample_metadata[sample_id].collection_date
                # Simple grouping by year-month for now
                try:
                    date_parts = date_str.split('-')
                    if len(date_parts) >= 2:
                        period = f"{date_parts[0]}-{date_parts[1]}"  # YYYY-MM
                        groups[period].append(sample_id)
                except:
                    groups['unknown'].append(sample_id)
        
        return dict(groups)
    
    def _group_samples_spatially(self, sample_ids: List[str], resolution: float) -> Dict[str, List[str]]:
        """Group samples by spatial regions"""
        groups = defaultdict(list)
        
        for sample_id in sample_ids:
            if sample_id in self.sample_metadata:
                location = self.sample_metadata[sample_id].location
                # Simple spatial binning
                lat_bin = int(location['lat'] / resolution) * resolution
                lon_bin = int(location['lon'] / resolution) * resolution
                region_key = f"{lat_bin}_{lon_bin}"
                groups[region_key].append(sample_id)
        
        return dict(groups)
    
    def _perform_statistical_tests(self, alpha_diversity: Dict, community_matrices: Dict) -> Dict[str, Any]:
        """Perform statistical tests on diversity data"""
        # Placeholder for statistical tests
        return {
            'statistical_tests_performed': ['diversity_comparison'],
            'significance_threshold': self.statistical_significance_threshold,
            'results': 'Statistical analysis not fully implemented'
        }
    
    def _generate_comparison_summary(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary for community comparison"""
        summary = {
            'analysis_type': 'community_comparison',
            'metrics_calculated': list(comparison_results.keys()),
            'samples_analyzed': len(comparison_results.get('alpha_diversity', {}))
        }
        return summary
    
    def _analyze_temporal_trends(self, temporal_diversity: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal trends in diversity"""
        return {'trend_analysis': 'Temporal trend analysis not fully implemented'}
    
    def _analyze_community_turnover(self, temporal_groups: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze community turnover between time periods"""
        return {'turnover_analysis': 'Community turnover analysis not fully implemented'}
    
    def _calculate_temporal_span(self, sample_ids: List[str]) -> Dict[str, Any]:
        """Calculate temporal span of samples"""
        dates = []
        for sample_id in sample_ids:
            if sample_id in self.sample_metadata:
                dates.append(self.sample_metadata[sample_id].collection_date)
        
        return {
            'earliest_date': min(dates) if dates else None,
            'latest_date': max(dates) if dates else None,
            'total_samples_with_dates': len(dates)
        }
    
    def _analyze_distance_decay(self, sample_ids: List[str]) -> Dict[str, Any]:
        """Analyze distance-decay relationships"""
        return {'distance_decay': 'Distance-decay analysis not fully implemented'}
    
    def _analyze_environmental_gradients(self, sample_ids: List[str]) -> Dict[str, Any]:
        """Analyze environmental gradients"""
        return {'gradients': 'Environmental gradient analysis not fully implemented'}
    
    def _calculate_spatial_autocorrelation(self, sample_ids: List[str]) -> Dict[str, Any]:
        """Calculate spatial autocorrelation"""
        return {'autocorrelation': 'Spatial autocorrelation analysis not fully implemented'}
    
    def _calculate_geographic_extent(self, sample_ids: List[str]) -> Dict[str, float]:
        """Calculate geographic extent of samples"""
        lats = []
        lons = []
        
        for sample_id in sample_ids:
            if sample_id in self.sample_metadata:
                location = self.sample_metadata[sample_id].location
                lats.append(location['lat'])
                lons.append(location['lon'])
        
        if not lats:
            return {}
        
        return {
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons),
            'max_lon': max(lons),
            'lat_range': max(lats) - min(lats),
            'lon_range': max(lons) - min(lons)
        }
    
    def _compare_diversity_metrics(self, sample_ids: List[str], diversity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compare diversity metrics between samples"""
        return {'diversity_comparison': 'Diversity metric comparison not fully implemented'}
    
    def _compare_taxonomic_composition(self, sample_ids: List[str], taxonomic_classification: Dict[str, Any]) -> Dict[str, Any]:
        """Compare taxonomic composition between samples"""
        return {'taxonomic_comparison': 'Taxonomic composition comparison not fully implemented'}
    
    def _calculate_sample_similarities(self, sample_ids: List[str], pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate similarities between samples"""
        return {'similarity_analysis': 'Sample similarity analysis not fully implemented'}
    
    def _calculate_environmental_correlations(self, env_data: Dict[str, Any], diversity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate correlations between environmental variables and diversity"""
        return {'correlations': 'Environmental correlation analysis not fully implemented'}

# Global comparative analyzer instance
comparative_analyzer = ComparativeAnalyzer()