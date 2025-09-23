"""
🧬 Comprehensive Test Suite for Genomics and eDNA Analysis Components

Unit and integration tests for sequence processing, taxonomic classification, and genomics analysis.
Tests eDNA processing, phylogenetics, diversity metrics, and comparative genomics functionality.

Author: Ocean-Bio Development Team
Version: 2.0.0
"""

import pytest
import numpy as np
import pandas as pd
import json
import io
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, mock_open
from typing import Dict, List, Any, Optional
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

# Import the modules under test
from backend.app.genomics.sequence_processing import SequenceProcessor
from backend.app.genomics.taxonomic_classification import TaxonomicClassifier
from backend.app.genomics.diversity_analysis import DiversityAnalyzer
from backend.app.genomics.phylogenetic_analysis import PhylogeneticAnalyzer
from backend.app.genomics.comparative_genomics import ComparativeGenomics

class TestSequenceProcessor:
    """Test suite for sequence processing functionality"""
    
    @pytest.fixture
    def sequence_processor(self):
        """Create sequence processor instance for testing"""
        return SequenceProcessor()
    
    @pytest.fixture
    def sample_sequences(self):
        """Sample DNA sequences for testing"""
        return {
            'valid_sequences': [
                'ATCGATCGATCGATCG',
                'GCTAGCTAGCTAGCTA',
                'TTTTTTTTTTTTTTTT',
                'AAAAAAAAAAAAAAA'
            ],
            'sequences_with_ambiguous': [
                'ATCGATNGATCGATCG',
                'GCTAGCTAGCWAGCTA',
                'TTTYYTTTTTTTTTTT'
            ],
            'low_quality_sequences': [
                'ATCGNNNNNGATCG',
                'NNNNNNNNNNNNNN',
                'ATCGATCG'  # Too short
            ]
        }
    
    @pytest.fixture
    def sample_fasta_content(self):
        """Sample FASTA file content"""
        return """>seq1_marine_sample_1
ATCGATCGATCGATCGATCGATCGATCG
>seq2_marine_sample_2
GCTAGCTAGCTAGCTAGCTAGCTAGCTA
>seq3_marine_sample_3
TTTTTTTTTTTTTTTTTTTTTTTTTTTT
>seq4_marine_sample_4
AAAAAAAAAAAAAAAAAAAAAAAAAAAA
"""
    
    def test_validate_sequence(self, sequence_processor, sample_sequences):
        """Test DNA sequence validation"""
        # Test valid sequences
        for seq in sample_sequences['valid_sequences']:
            result = sequence_processor.validate_sequence(seq)
            assert result['is_valid'] is True
            assert result['length'] == len(seq)
            assert result['gc_content'] >= 0
            assert result['n_count'] == 0
        
        # Test sequences with ambiguous bases
        for seq in sample_sequences['sequences_with_ambiguous']:
            result = sequence_processor.validate_sequence(seq)
            assert result['is_valid'] is True
            assert result['n_count'] > 0
        
        # Test low quality sequences
        for seq in sample_sequences['low_quality_sequences']:
            result = sequence_processor.validate_sequence(seq, min_length=15)
            if len(seq) < 15 or seq.count('N') > len(seq) * 0.5:
                assert result['is_valid'] is False
    
    def test_calculate_gc_content(self, sequence_processor):
        """Test GC content calculation"""
        # Test known GC content
        sequence_100_gc = 'GGGGCCCCGGGGCCCC'  # 100% GC
        sequence_0_gc = 'AAAATTTTAAAATTTT'    # 0% GC
        sequence_50_gc = 'ATCGATCGATCGATCG'    # 50% GC
        
        assert sequence_processor._calculate_gc_content(sequence_100_gc) == 100.0
        assert sequence_processor._calculate_gc_content(sequence_0_gc) == 0.0
        assert sequence_processor._calculate_gc_content(sequence_50_gc) == 50.0
    
    @pytest.mark.asyncio
    async def test_process_fasta_file(self, sequence_processor, sample_fasta_content):
        """Test FASTA file processing"""
        # Create mock file handle
        mock_file = io.StringIO(sample_fasta_content)
        
        with patch('builtins.open', return_value=mock_file):
            result = await sequence_processor.process_fasta_file(
                'test.fasta',
                min_length=20,
                max_ambiguous=0.1
            )
        
        assert 'total_sequences' in result
        assert 'processed_sequences' in result
        assert 'quality_metrics' in result
        assert result['total_sequences'] == 4
        
        # Check quality metrics
        metrics = result['quality_metrics']
        assert 'average_length' in metrics
        assert 'gc_content_distribution' in metrics
        assert 'sequence_count_by_length' in metrics
    
    @pytest.mark.asyncio
    async def test_quality_filtering(self, sequence_processor, sample_sequences):
        """Test sequence quality filtering"""
        all_sequences = (
            sample_sequences['valid_sequences'] + 
            sample_sequences['sequences_with_ambiguous'] + 
            sample_sequences['low_quality_sequences']
        )
        
        # Create sequence records
        seq_records = [
            {'id': f'seq_{i}', 'sequence': seq, 'quality': None}
            for i, seq in enumerate(all_sequences)
        ]
        
        filtered_result = await sequence_processor._apply_quality_filters(
            sequences=seq_records,
            min_length=15,
            max_ambiguous=0.2,
            min_complexity=0.5
        )
        
        assert 'passed_sequences' in filtered_result
        assert 'failed_sequences' in filtered_result
        assert 'filter_statistics' in filtered_result
        
        # Should filter out short sequences and high N-content sequences
        assert len(filtered_result['failed_sequences']) > 0
    
    def test_sequence_complexity_calculation(self, sequence_processor):
        """Test sequence complexity calculation"""
        # High complexity sequence
        complex_seq = 'ATCGATCGATCGATCG'
        # Low complexity sequence
        simple_seq = 'AAAAAAAAAAAAAAAA'
        
        complex_result = sequence_processor._calculate_complexity(complex_seq)
        simple_result = sequence_processor._calculate_complexity(simple_seq)
        
        assert complex_result > simple_result
        assert 0 <= complex_result <= 1
        assert 0 <= simple_result <= 1
    
    @pytest.mark.asyncio
    async def test_reverse_complement(self, sequence_processor):
        """Test reverse complement calculation"""
        test_sequence = 'ATCGATCG'
        expected_rc = 'CGATCGAT'
        
        result = sequence_processor.reverse_complement(test_sequence)
        assert result == expected_rc
        
        # Test that reverse complement of reverse complement equals original
        double_rc = sequence_processor.reverse_complement(result)
        assert double_rc == test_sequence


class TestTaxonomicClassifier:
    """Test suite for taxonomic classification functionality"""
    
    @pytest.fixture
    def taxonomic_classifier(self):
        """Create taxonomic classifier instance for testing"""
        return TaxonomicClassifier()
    
    @pytest.fixture
    def sample_query_sequences(self):
        """Sample sequences for taxonomic classification"""
        return [
            {
                'id': 'marine_seq_1',
                'sequence': 'ATCGATCGATCGATCGATCGATCGATCGATCG',
                'description': 'Marine eDNA sample 1'
            },
            {
                'id': 'marine_seq_2',
                'sequence': 'GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA',
                'description': 'Marine eDNA sample 2'
            }
        ]
    
    @pytest.fixture
    def mock_blast_results(self):
        """Mock BLAST search results"""
        return [
            {
                'query_id': 'marine_seq_1',
                'subject_id': 'ref_seq_001',
                'identity': 98.5,
                'alignment_length': 300,
                'e_value': 1e-150,
                'bit_score': 558,
                'taxonomy': {
                    'kingdom': 'Eukaryota',
                    'phylum': 'Chordata',
                    'class': 'Actinopterygii',
                    'order': 'Perciformes',
                    'family': 'Scombridae',
                    'genus': 'Thunnus',
                    'species': 'Thunnus albacares'
                }
            },
            {
                'query_id': 'marine_seq_2',
                'subject_id': 'ref_seq_002',
                'identity': 95.2,
                'alignment_length': 280,
                'e_value': 1e-120,
                'bit_score': 445,
                'taxonomy': {
                    'kingdom': 'Eukaryota',
                    'phylum': 'Chordata',
                    'class': 'Actinopterygii',
                    'order': 'Clupeiformes',
                    'family': 'Clupeidae',
                    'genus': 'Sardinella',
                    'species': 'Sardinella longiceps'
                }
            }
        ]
    
    def test_get_available_databases(self, taxonomic_classifier):
        """Test getting available reference databases"""
        databases = taxonomic_classifier.get_available_databases()
        
        assert isinstance(databases, list)
        assert len(databases) > 0
        
        # Check database structure
        for db in databases:
            assert 'name' in db
            assert 'description' in db
            assert 'type' in db
            assert 'last_updated' in db
    
    def test_get_classification_methods(self, taxonomic_classifier):
        """Test getting available classification methods"""
        methods = taxonomic_classifier.get_classification_methods()
        
        assert isinstance(methods, list)
        assert 'blast' in methods
        assert 'kraken2' in methods
        assert 'diamond' in methods
    
    @pytest.mark.asyncio
    async def test_classify_sequences(self, taxonomic_classifier, sample_query_sequences, mock_blast_results):
        """Test taxonomic classification of sequences"""
        classification_params = {
            'method': 'blast',
            'database': 'ncbi_nt',
            'e_value_threshold': 1e-10,
            'identity_threshold': 95.0,
            'coverage_threshold': 80.0
        }
        
        with patch.object(taxonomic_classifier, '_run_blast_search') as mock_blast:
            mock_blast.return_value = mock_blast_results
            
            result = await taxonomic_classifier.classify_sequences(
                sequences=sample_query_sequences,
                **classification_params
            )
            
            assert 'classification_id' in result
            assert 'method' in result
            assert 'results' in result
            assert 'summary_statistics' in result
            
            # Check individual results
            results = result['results']
            assert len(results) == 2
            
            for res in results:
                assert 'query_id' in res
                assert 'best_match' in res
                assert 'taxonomy' in res
                assert 'confidence_score' in res
    
    @pytest.mark.asyncio
    async def test_batch_classification(self, taxonomic_classifier):
        """Test batch classification of multiple sequence sets"""
        batch_data = [
            {
                'sample_id': 'site_A',
                'sequences': [
                    {'id': 'siteA_seq1', 'sequence': 'ATCGATCGATCG'},
                    {'id': 'siteA_seq2', 'sequence': 'GCTAGCTAGCTA'}
                ]
            },
            {
                'sample_id': 'site_B',
                'sequences': [
                    {'id': 'siteB_seq1', 'sequence': 'TTTTTTTTTTTT'},
                    {'id': 'siteB_seq2', 'sequence': 'AAAAAAAAAAAA'}
                ]
            }
        ]
        
        with patch.object(taxonomic_classifier, '_run_classification_pipeline') as mock_pipeline:
            mock_pipeline.return_value = {
                'results': [{'query_id': 'test', 'taxonomy': {}}],
                'summary': {'total_classified': 1}
            }
            
            result = await taxonomic_classifier.batch_classify(
                batch_data=batch_data,
                method='blast',
                database='ncbi_nt'
            )
            
            assert 'batch_id' in result
            assert 'sample_results' in result
            assert len(result['sample_results']) == 2
    
    def test_confidence_score_calculation(self, taxonomic_classifier):
        """Test confidence score calculation for classifications"""
        # High confidence match
        high_conf_match = {
            'identity': 99.5,
            'coverage': 95.0,
            'e_value': 1e-180,
            'bit_score': 600
        }
        
        # Low confidence match
        low_conf_match = {
            'identity': 85.0,
            'coverage': 60.0,
            'e_value': 1e-5,
            'bit_score': 50
        }
        
        high_score = taxonomic_classifier._calculate_confidence_score(high_conf_match)
        low_score = taxonomic_classifier._calculate_confidence_score(low_conf_match)
        
        assert high_score > low_score
        assert 0 <= high_score <= 1
        assert 0 <= low_score <= 1
        assert high_score > 0.8  # Should be high confidence
        assert low_score < 0.6   # Should be low confidence
    
    def test_taxonomy_validation(self, taxonomic_classifier):
        """Test taxonomy validation functionality"""
        # Valid taxonomy
        valid_taxonomy = {
            'kingdom': 'Eukaryota',
            'phylum': 'Chordata',
            'class': 'Actinopterygii',
            'order': 'Perciformes',
            'family': 'Scombridae',
            'genus': 'Thunnus',
            'species': 'Thunnus albacares'
        }
        
        # Invalid taxonomy (missing levels)
        invalid_taxonomy = {
            'kingdom': 'Eukaryota',
            'species': 'Thunnus albacares'  # Missing intermediate levels
        }
        
        assert taxonomic_classifier._validate_taxonomy(valid_taxonomy) is True
        assert taxonomic_classifier._validate_taxonomy(invalid_taxonomy) is False
    
    @pytest.mark.asyncio
    async def test_lca_analysis(self, taxonomic_classifier, mock_blast_results):
        """Test Lowest Common Ancestor (LCA) analysis"""
        # Multiple hits for same query with different taxonomies
        multiple_hits = [
            {
                'query_id': 'test_seq',
                'taxonomy': {
                    'kingdom': 'Eukaryota',
                    'phylum': 'Chordata',
                    'class': 'Actinopterygii',
                    'family': 'Scombridae',
                    'genus': 'Thunnus',
                    'species': 'Thunnus albacares'
                }
            },
            {
                'query_id': 'test_seq',
                'taxonomy': {
                    'kingdom': 'Eukaryota',
                    'phylum': 'Chordata',
                    'class': 'Actinopterygii',
                    'family': 'Scombridae',
                    'genus': 'Thunnus',
                    'species': 'Thunnus obesus'
                }
            }
        ]
        
        lca_result = await taxonomic_classifier._perform_lca_analysis(multiple_hits)
        
        # Should find common ancestor at genus level
        assert lca_result['lca_level'] == 'genus'
        assert lca_result['lca_taxon'] == 'Thunnus'


class TestDiversityAnalyzer:
    """Test suite for diversity analysis functionality"""
    
    @pytest.fixture
    def diversity_analyzer(self):
        """Create diversity analyzer instance for testing"""
        return DiversityAnalyzer()
    
    @pytest.fixture
    def sample_community_data(self):
        """Sample community composition data"""
        return pd.DataFrame({
            'sample_id': ['site_A'] * 5 + ['site_B'] * 5 + ['site_C'] * 5,
            'species': [
                'Thunnus albacares', 'Sardinella longiceps', 'Rastrelliger kanagurta',
                'Scomberomorus commerson', 'Lutjanus johnii'
            ] * 3,
            'abundance': [
                100, 80, 60, 40, 20,  # Site A
                120, 90, 50, 30, 10,  # Site B
                80, 100, 70, 50, 30   # Site C
            ],
            'reads_count': [
                1000, 800, 600, 400, 200,  # Site A
                1200, 900, 500, 300, 100,  # Site B
                800, 1000, 700, 500, 300   # Site C
            ]
        })
    
    @pytest.fixture
    def phylogenetic_tree_data(self):
        """Mock phylogenetic tree data for diversity analysis"""
        return {
            'tree_structure': '((Thunnus_albacares:0.1,Thunnus_obesus:0.1):0.05,(Sardinella_longiceps:0.15,Rastrelliger_kanagurta:0.15):0.05):0.1;',
            'species_list': ['Thunnus_albacares', 'Thunnus_obesus', 'Sardinella_longiceps', 'Rastrelliger_kanagurta'],
            'branch_lengths': True
        }
    
    def test_get_available_metrics(self, diversity_analyzer):
        """Test getting available diversity metrics"""
        metrics = diversity_analyzer.get_available_metrics()
        
        assert isinstance(metrics, dict)
        assert 'alpha_diversity' in metrics
        assert 'beta_diversity' in metrics
        assert 'phylogenetic_diversity' in metrics
        
        # Check alpha diversity metrics
        alpha_metrics = metrics['alpha_diversity']
        assert 'shannon' in alpha_metrics
        assert 'simpson' in alpha_metrics
        assert 'chao1' in alpha_metrics
    
    @pytest.mark.asyncio
    async def test_calculate_alpha_diversity(self, diversity_analyzer, sample_community_data):
        """Test alpha diversity calculation"""
        # Calculate for one site
        site_a_data = sample_community_data[sample_community_data['sample_id'] == 'site_A']
        
        result = await diversity_analyzer.calculate_alpha_diversity(
            abundance_data=site_a_data,
            abundance_column='abundance',
            metrics=['shannon', 'simpson', 'chao1']
        )
        
        assert 'sample_id' in result
        assert 'metrics' in result
        
        metrics = result['metrics']
        assert 'shannon' in metrics
        assert 'simpson' in metrics
        assert 'chao1' in metrics
        
        # Shannon index should be positive
        assert metrics['shannon'] > 0
        # Simpson index should be between 0 and 1
        assert 0 <= metrics['simpson'] <= 1
        # Chao1 should be >= observed species count
        assert metrics['chao1'] >= len(site_a_data)
    
    @pytest.mark.asyncio
    async def test_calculate_beta_diversity(self, diversity_analyzer, sample_community_data):
        """Test beta diversity calculation"""
        # Pivot data for beta diversity analysis
        community_matrix = sample_community_data.pivot(
            index='sample_id', columns='species', values='abundance'
        ).fillna(0)
        
        result = await diversity_analyzer.calculate_beta_diversity(
            community_matrix=community_matrix,
            method='bray_curtis'
        )
        
        assert 'method' in result
        assert result['method'] == 'bray_curtis'
        assert 'distance_matrix' in result
        assert 'ordination' in result
        
        # Check distance matrix properties
        distance_matrix = np.array(result['distance_matrix'])
        n_samples = len(community_matrix)
        
        assert distance_matrix.shape == (n_samples, n_samples)
        # Diagonal should be zero (distance from sample to itself)
        assert np.allclose(np.diag(distance_matrix), 0)
        # Matrix should be symmetric
        assert np.allclose(distance_matrix, distance_matrix.T)
    
    @pytest.mark.asyncio
    async def test_phylogenetic_diversity(self, diversity_analyzer, sample_community_data, phylogenetic_tree_data):
        """Test phylogenetic diversity calculation"""
        # Use subset of data that matches tree species
        tree_species = phylogenetic_tree_data['species_list']
        filtered_data = sample_community_data[
            sample_community_data['species'].isin(tree_species)
        ]
        
        with patch.object(diversity_analyzer, '_load_phylogenetic_tree') as mock_tree:
            mock_tree.return_value = phylogenetic_tree_data
            
            result = await diversity_analyzer.calculate_phylogenetic_diversity(
                community_data=filtered_data,
                tree_data=phylogenetic_tree_data,
                metrics=['pd', 'mpd', 'mntd']
            )
            
            assert 'metrics' in result
            assert 'pd' in result['metrics']  # Phylogenetic Diversity
            assert 'mpd' in result['metrics']  # Mean Pairwise Distance
            assert 'mntd' in result['metrics']  # Mean Nearest Taxon Distance
            
            # All metrics should be positive
            for metric_value in result['metrics'].values():
                assert metric_value >= 0
    
    @pytest.mark.asyncio
    async def test_diversity_comparison(self, diversity_analyzer, sample_community_data):
        """Test diversity comparison between samples"""
        result = await diversity_analyzer.compare_diversity(
            community_data=sample_community_data,
            group_column='sample_id',
            abundance_column='abundance',
            comparison_metrics=['shannon', 'simpson']
        )
        
        assert 'comparison_type' in result
        assert 'statistical_tests' in result
        assert 'diversity_values' in result
        
        # Should have diversity values for each sample
        diversity_values = result['diversity_values']
        sample_ids = sample_community_data['sample_id'].unique()
        
        for sample_id in sample_ids:
            assert sample_id in diversity_values
            assert 'shannon' in diversity_values[sample_id]
            assert 'simpson' in diversity_values[sample_id]
    
    def test_rarefaction_curve_calculation(self, diversity_analyzer):
        """Test rarefaction curve calculation"""
        # Create sample abundance data
        abundance_data = np.array([100, 80, 60, 40, 20, 15, 10, 8, 5, 2])
        
        result = diversity_analyzer._calculate_rarefaction_curve(
            abundance_data, max_samples=200
        )
        
        assert 'sample_sizes' in result
        assert 'species_counts' in result
        assert 'confidence_intervals' in result
        
        # Should have increasing species counts with sample size
        sample_sizes = result['sample_sizes']
        species_counts = result['species_counts']
        
        assert len(sample_sizes) == len(species_counts)
        assert species_counts[0] <= species_counts[-1]  # Non-decreasing
    
    def test_evenness_calculation(self, diversity_analyzer):
        """Test community evenness calculation"""
        # High evenness community (all species equal abundance)
        even_community = np.array([25, 25, 25, 25])
        # Low evenness community (one dominant species)
        uneven_community = np.array([90, 5, 3, 2])
        
        even_result = diversity_analyzer._calculate_evenness(even_community)
        uneven_result = diversity_analyzer._calculate_evenness(uneven_community)
        
        # Evenness should be higher for even community
        assert even_result > uneven_result
        assert 0 <= even_result <= 1
        assert 0 <= uneven_result <= 1


class TestPhylogeneticAnalyzer:
    """Test suite for phylogenetic analysis functionality"""
    
    @pytest.fixture
    def phylogenetic_analyzer(self):
        """Create phylogenetic analyzer instance for testing"""
        return PhylogeneticAnalyzer()
    
    @pytest.fixture
    def sample_sequence_data(self):
        """Sample aligned sequences for phylogenetic analysis"""
        return [
            {'id': 'seq1', 'sequence': 'ATCGATCGATCG'},
            {'id': 'seq2', 'sequence': 'ATCGATCGATCG'},
            {'id': 'seq3', 'sequence': 'ATCGATGGATCG'},
            {'id': 'seq4', 'sequence': 'ATCGATGGATCG'},
        ]
    
    def test_get_available_methods(self, phylogenetic_analyzer):
        """Test getting available phylogenetic methods"""
        methods = phylogenetic_analyzer.get_available_methods()
        
        assert isinstance(methods, dict)
        assert 'distance_based' in methods
        assert 'character_based' in methods
        
        # Check specific methods
        distance_methods = methods['distance_based']
        assert 'neighbor_joining' in distance_methods
        assert 'upgma' in distance_methods
        
        character_methods = methods['character_based']
        assert 'maximum_likelihood' in character_methods
        assert 'maximum_parsimony' in character_methods
    
    @pytest.mark.asyncio
    async def test_multiple_sequence_alignment(self, phylogenetic_analyzer):
        """Test multiple sequence alignment"""
        unaligned_sequences = [
            {'id': 'seq1', 'sequence': 'ATCGATCGATCG'},
            {'id': 'seq2', 'sequence': 'ATCGAGATCGATCG'},
            {'id': 'seq3', 'sequence': 'ATCGATGATCG'},
        ]
        
        with patch.object(phylogenetic_analyzer, '_run_muscle_alignment') as mock_muscle:
            mock_muscle.return_value = [
                {'id': 'seq1', 'sequence': 'ATCGATCGATCG--'},
                {'id': 'seq2', 'sequence': 'ATCGAGATCGATCG'},
                {'id': 'seq3', 'sequence': 'ATCGAT-GATCG--'},
            ]
            
            result = await phylogenetic_analyzer.multiple_sequence_alignment(
                sequences=unaligned_sequences,
                method='muscle'
            )
            
            assert 'alignment_id' in result
            assert 'method' in result
            assert 'aligned_sequences' in result
            assert 'alignment_stats' in result
            
            # All sequences should have same length after alignment
            aligned_seqs = result['aligned_sequences']
            lengths = [len(seq['sequence']) for seq in aligned_seqs]
            assert len(set(lengths)) == 1  # All same length
    
    @pytest.mark.asyncio
    async def test_distance_matrix_calculation(self, phylogenetic_analyzer, sample_sequence_data):
        """Test phylogenetic distance matrix calculation"""
        result = await phylogenetic_analyzer.calculate_distance_matrix(
            aligned_sequences=sample_sequence_data,
            model='jukes_cantor'
        )
        
        assert 'distance_matrix' in result
        assert 'model' in result
        assert result['model'] == 'jukes_cantor'
        assert 'sequence_names' in result
        
        # Check matrix properties
        distance_matrix = np.array(result['distance_matrix'])
        n_sequences = len(sample_sequence_data)
        
        assert distance_matrix.shape == (n_sequences, n_sequences)
        # Diagonal should be zero
        assert np.allclose(np.diag(distance_matrix), 0)
        # Matrix should be symmetric
        assert np.allclose(distance_matrix, distance_matrix.T)
    
    @pytest.mark.asyncio
    async def test_build_phylogenetic_tree(self, phylogenetic_analyzer, sample_sequence_data):
        """Test phylogenetic tree construction"""
        with patch.object(phylogenetic_analyzer, '_run_tree_building') as mock_tree:
            mock_tree.return_value = {
                'newick_format': '((seq1:0.1,seq2:0.1):0.05,(seq3:0.08,seq4:0.08):0.05);',
                'branch_lengths': True,
                'bootstrap_values': [95, 87, 92]
            }
            
            result = await phylogenetic_analyzer.build_tree(
                aligned_sequences=sample_sequence_data,
                method='neighbor_joining',
                bootstrap_replicates=100
            )
            
            assert 'tree_id' in result
            assert 'method' in result
            assert result['method'] == 'neighbor_joining'
            assert 'newick_tree' in result
            assert 'tree_statistics' in result
            
            # Check tree statistics
            stats = result['tree_statistics']
            assert 'total_length' in stats
            assert 'bootstrap_support' in stats
    
    @pytest.mark.asyncio
    async def test_tree_validation(self, phylogenetic_analyzer):
        """Test phylogenetic tree validation"""
        # Valid Newick format
        valid_tree = '((seq1:0.1,seq2:0.1):0.05,(seq3:0.08,seq4:0.08):0.05);'
        # Invalid Newick format (unbalanced parentheses)
        invalid_tree = '((seq1:0.1,seq2:0.1):0.05,(seq3:0.08,seq4:0.08:0.05);'
        
        valid_result = phylogenetic_analyzer._validate_tree_format(valid_tree)
        invalid_result = phylogenetic_analyzer._validate_tree_format(invalid_tree)
        
        assert valid_result['is_valid'] is True
        assert invalid_result['is_valid'] is False
        assert len(invalid_result['errors']) > 0
    
    def test_bootstrap_support_analysis(self, phylogenetic_analyzer):
        """Test bootstrap support analysis"""
        bootstrap_values = [95, 87, 92, 78, 65, 88]
        
        result = phylogenetic_analyzer._analyze_bootstrap_support(bootstrap_values)
        
        assert 'mean_support' in result
        assert 'median_support' in result
        assert 'support_categories' in result
        
        # Check support categories
        categories = result['support_categories']
        assert 'high_support' in categories  # >= 90
        assert 'moderate_support' in categories  # 70-89
        assert 'low_support' in categories  # < 70
    
    @pytest.mark.asyncio
    async def test_molecular_clock_analysis(self, phylogenetic_analyzer):
        """Test molecular clock analysis"""
        tree_with_dates = {
            'newick_tree': '((seq1:0.1,seq2:0.1):0.05,(seq3:0.08,seq4:0.08):0.05);',
            'tip_dates': {
                'seq1': '2020-01-01',
                'seq2': '2021-01-01',
                'seq3': '2022-01-01',
                'seq4': '2023-01-01'
            }
        }
        
        result = await phylogenetic_analyzer.molecular_clock_analysis(
            tree_data=tree_with_dates,
            clock_model='strict'
        )
        
        assert 'clock_rate' in result
        assert 'tmrca' in result  # Time to Most Recent Common Ancestor
        assert 'node_ages' in result
        assert 'clock_test' in result
        
        # Clock rate should be positive
        assert result['clock_rate'] > 0


class TestComparativeGenomics:
    """Test suite for comparative genomics functionality"""
    
    @pytest.fixture
    def comparative_genomics(self):
        """Create comparative genomics instance for testing"""
        return ComparativeGenomics()
    
    @pytest.fixture
    def sample_genome_data(self):
        """Sample genome/sequence data for comparison"""
        return {
            'genome1': {
                'id': 'species_A_genome',
                'sequences': [
                    {'id': 'chr1', 'sequence': 'ATCGATCGATCGATCG'},
                    {'id': 'chr2', 'sequence': 'GCTAGCTAGCTAGCTA'}
                ],
                'annotations': [
                    {'type': 'gene', 'start': 1, 'end': 8, 'strand': '+', 'id': 'gene1'},
                    {'type': 'gene', 'start': 9, 'end': 16, 'strand': '-', 'id': 'gene2'}
                ]
            },
            'genome2': {
                'id': 'species_B_genome',
                'sequences': [
                    {'id': 'chr1', 'sequence': 'ATCGATCGATGGATCG'},
                    {'id': 'chr2', 'sequence': 'GCTAGCTAGGTAGCTA'}
                ],
                'annotations': [
                    {'type': 'gene', 'start': 1, 'end': 8, 'strand': '+', 'id': 'gene1'},
                    {'type': 'gene', 'start': 10, 'end': 16, 'strand': '-', 'id': 'gene2'}
                ]
            }
        }
    
    def test_get_available_analyses(self, comparative_genomics):
        """Test getting available comparative analyses"""
        analyses = comparative_genomics.get_available_analyses()
        
        assert isinstance(analyses, list)
        assert 'synteny_analysis' in analyses
        assert 'ortholog_detection' in analyses
        assert 'sequence_similarity' in analyses
        assert 'structural_variation' in analyses
    
    @pytest.mark.asyncio
    async def test_pairwise_genome_comparison(self, comparative_genomics, sample_genome_data):
        """Test pairwise genome comparison"""
        genome1 = sample_genome_data['genome1']
        genome2 = sample_genome_data['genome2']
        
        result = await comparative_genomics.compare_genomes(
            genome1=genome1,
            genome2=genome2,
            analysis_types=['sequence_similarity', 'synteny_analysis']
        )
        
        assert 'comparison_id' in result
        assert 'genome1_id' in result
        assert 'genome2_id' in result
        assert 'analyses' in result
        
        # Check individual analyses
        analyses = result['analyses']
        assert 'sequence_similarity' in analyses
        assert 'synteny_analysis' in analyses
        
        # Check sequence similarity results
        sim_result = analyses['sequence_similarity']
        assert 'overall_similarity' in sim_result
        assert 'alignment_stats' in sim_result
        assert 0 <= sim_result['overall_similarity'] <= 100
    
    @pytest.mark.asyncio
    async def test_ortholog_detection(self, comparative_genomics):
        """Test orthologous gene detection"""
        # Mock gene sequences for two species
        species1_genes = [
            {'id': 'gene1_sp1', 'sequence': 'ATCGATCGATCG'},
            {'id': 'gene2_sp1', 'sequence': 'GCTAGCTAGCTA'},
            {'id': 'gene3_sp1', 'sequence': 'TTTTTTTTTTTT'}
        ]
        
        species2_genes = [
            {'id': 'gene1_sp2', 'sequence': 'ATCGATCGATCG'},  # Ortholog of gene1_sp1
            {'id': 'gene2_sp2', 'sequence': 'GCTAGCTAGGTA'},  # Similar to gene2_sp1
            {'id': 'gene4_sp2', 'sequence': 'AAAAAAAAAAAA'}   # Unique to species2
        ]
        
        with patch.object(comparative_genomics, '_perform_blast_search') as mock_blast:
            mock_blast.return_value = [
                {'query': 'gene1_sp1', 'subject': 'gene1_sp2', 'identity': 100.0, 'e_value': 0.0},
                {'query': 'gene2_sp1', 'subject': 'gene2_sp2', 'identity': 85.0, 'e_value': 1e-20}
            ]
            
            result = await comparative_genomics.detect_orthologs(
                species1_genes=species1_genes,
                species2_genes=species2_genes,
                identity_threshold=80.0,
                coverage_threshold=70.0
            )
            
            assert 'ortholog_pairs' in result
            assert 'paralog_groups' in result
            assert 'unique_genes' in result
            
            # Should find ortholog pairs
            ortholog_pairs = result['ortholog_pairs']
            assert len(ortholog_pairs) >= 1
            
            # Check ortholog pair structure
            for pair in ortholog_pairs:
                assert 'species1_gene' in pair
                assert 'species2_gene' in pair
                assert 'similarity_score' in pair
    
    @pytest.mark.asyncio
    async def test_synteny_analysis(self, comparative_genomics, sample_genome_data):
        """Test synteny analysis between genomes"""
        result = await comparative_genomics.analyze_synteny(
            genome1=sample_genome_data['genome1'],
            genome2=sample_genome_data['genome2'],
            window_size=10,
            min_syntenic_genes=2
        )
        
        assert 'syntenic_blocks' in result
        assert 'inversions' in result
        assert 'translocations' in result
        assert 'synteny_statistics' in result
        
        # Check statistics
        stats = result['synteny_statistics']
        assert 'total_syntenic_blocks' in stats
        assert 'average_block_size' in stats
        assert 'synteny_coverage' in stats
    
    def test_sequence_alignment_scoring(self, comparative_genomics):
        """Test sequence alignment scoring"""
        # Test identical sequences
        seq1 = 'ATCGATCGATCG'
        seq2 = 'ATCGATCGATCG'
        score_identical = comparative_genomics._calculate_alignment_score(seq1, seq2)
        
        # Test different sequences
        seq3 = 'ATCGATGGATCG'
        score_different = comparative_genomics._calculate_alignment_score(seq1, seq3)
        
        assert score_identical > score_different
        assert score_identical == 100.0  # Perfect match
        assert 0 <= score_different <= 100
    
    @pytest.mark.asyncio
    async def test_structural_variation_detection(self, comparative_genomics):
        """Test structural variation detection"""
        # Mock alignment data with structural variations
        alignment_data = [
            {'type': 'match', 'query_start': 1, 'query_end': 100, 'subject_start': 1, 'subject_end': 100},
            {'type': 'deletion', 'query_start': 101, 'query_end': 150, 'subject_start': 101, 'subject_end': 101},
            {'type': 'insertion', 'query_start': 151, 'query_end': 151, 'subject_start': 102, 'subject_end': 152},
            {'type': 'inversion', 'query_start': 200, 'query_end': 300, 'subject_start': 300, 'subject_end': 200}
        ]
        
        result = await comparative_genomics.detect_structural_variations(
            alignment_data=alignment_data,
            min_sv_size=10
        )
        
        assert 'deletions' in result
        assert 'insertions' in result
        assert 'inversions' in result
        assert 'translocations' in result
        
        # Check that variations are detected
        assert len(result['deletions']) > 0
        assert len(result['insertions']) > 0
        assert len(result['inversions']) > 0


class TestIntegration:
    """Integration tests for genomics components"""
    
    @pytest.fixture
    def services(self):
        """Create all service instances for integration testing"""
        return {
            'processor': SequenceProcessor(),
            'classifier': TaxonomicClassifier(),
            'diversity': DiversityAnalyzer(),
            'phylogenetics': PhylogeneticAnalyzer(),
            'comparative': ComparativeGenomics()
        }
    
    @pytest.fixture
    def integrated_edna_data(self):
        """Comprehensive eDNA dataset for integration testing"""
        return {
            'raw_sequences': [
                {'id': 'seq1', 'sequence': 'ATCGATCGATCGATCGATCGATCG'},
                {'id': 'seq2', 'sequence': 'GCTAGCTAGCTAGCTAGCTAGCTA'},
                {'id': 'seq3', 'sequence': 'TTTTTTTTTTTTTTTTTTTTTTTT'},
                {'id': 'seq4', 'sequence': 'AAAAAAAAAAAAAAAAAAAAAAA'},
                {'id': 'seq5', 'sequence': 'ATCGATCGATCGATCGATCGATCG'}
            ],
            'sample_metadata': {
                'site_A': ['seq1', 'seq2'],
                'site_B': ['seq3', 'seq4'],
                'site_C': ['seq5']
            }
        }
    
    @pytest.mark.asyncio
    async def test_complete_edna_workflow(self, services, integrated_edna_data):
        """Test complete eDNA analysis workflow"""
        # Step 1: Sequence Processing
        raw_sequences = integrated_edna_data['raw_sequences']
        
        # Mock FASTA content
        fasta_content = '\n'.join([
            f">{seq['id']}\n{seq['sequence']}"
            for seq in raw_sequences
        ])
        
        with patch('builtins.open', mock_open(read_data=fasta_content)):
            processing_result = await services['processor'].process_fasta_file(
                'test.fasta',
                min_length=20,
                max_ambiguous=0.1
            )
            
            assert 'processed_sequences' in processing_result
        
        # Step 2: Taxonomic Classification
        with patch.object(services['classifier'], '_run_blast_search') as mock_blast:
            mock_blast.return_value = [
                {
                    'query_id': 'seq1',
                    'subject_id': 'ref1',
                    'identity': 98.0,
                    'e_value': 1e-100,
                    'taxonomy': {
                        'kingdom': 'Eukaryota',
                        'phylum': 'Chordata',
                        'species': 'Thunnus albacares'
                    }
                }
            ]
            
            classification_result = await services['classifier'].classify_sequences(
                sequences=raw_sequences[:1],
                method='blast',
                database='ncbi_nt'
            )
            
            assert 'results' in classification_result
        
        # Step 3: Diversity Analysis
        # Create mock community data from classification results
        community_data = pd.DataFrame({
            'sample_id': ['site_A', 'site_B', 'site_C'],
            'species': ['Species_A', 'Species_B', 'Species_C'],
            'abundance': [100, 80, 60]
        })
        
        diversity_result = await services['diversity'].calculate_alpha_diversity(
            abundance_data=community_data,
            abundance_column='abundance'
        )
        
        assert 'metrics' in diversity_result
        
        # Step 4: Phylogenetic Analysis
        with patch.object(services['phylogenetics'], '_run_tree_building') as mock_tree:
            mock_tree.return_value = {
                'newick_format': '((seq1:0.1,seq2:0.1):0.05,seq3:0.08);',
                'bootstrap_values': [95, 87]
            }
            
            tree_result = await services['phylogenetics'].build_tree(
                aligned_sequences=raw_sequences[:3],
                method='neighbor_joining'
            )
            
            assert 'newick_tree' in tree_result
    
    @pytest.mark.asyncio
    async def test_cross_component_data_flow(self, services):
        """Test data flow between genomics components"""
        # Sequence processing results feed into classification
        processed_sequences = [
            {'id': 'seq1', 'sequence': 'ATCGATCGATCGATCG', 'quality_score': 0.95}
        ]
        
        # Classification results feed into diversity analysis
        classification_results = [
            {
                'query_id': 'seq1',
                'taxonomy': {
                    'species': 'Thunnus albacares'
                },
                'confidence_score': 0.98
            }
        ]
        
        # Convert to community data for diversity analysis
        community_data = pd.DataFrame({
            'sample_id': ['test_sample'],
            'species': ['Thunnus_albacares'],
            'abundance': [1]
        })
        
        # Test the flow
        diversity_result = await services['diversity'].calculate_alpha_diversity(
            abundance_data=community_data,
            abundance_column='abundance'
        )
        
        assert diversity_result['metrics']['shannon'] >= 0
    
    def test_performance_benchmarks(self, services):
        """Test performance benchmarks for genomics operations"""
        import time
        
        # Benchmark sequence validation
        test_sequence = 'ATCGATCGATCG' * 100  # 1200bp sequence
        
        start_time = time.time()
        for _ in range(100):
            services['processor'].validate_sequence(test_sequence)
        validation_time = time.time() - start_time
        
        # Should validate 100 sequences in less than 1 second
        assert validation_time < 1.0
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, services):
        """Test error handling across genomics services"""
        # Test invalid sequence data
        invalid_sequences = [
            {'id': 'bad_seq', 'sequence': 'XYZXYZ'}  # Invalid nucleotides
        ]
        
        validation_result = services['processor'].validate_sequence(
            invalid_sequences[0]['sequence']
        )
        assert validation_result['is_valid'] is False
        
        # Test empty community data for diversity analysis
        empty_community = pd.DataFrame()
        
        with pytest.raises((ValueError, IndexError)):
            await services['diversity'].calculate_alpha_diversity(
                abundance_data=empty_community,
                abundance_column='abundance'
            )


# Test fixtures and utilities
@pytest.fixture
def mock_blast_database():
    """Mock BLAST database for testing"""
    return {
        'database_name': 'test_marine_db',
        'sequences': [
            {
                'accession': 'TEST001',
                'sequence': 'ATCGATCGATCGATCG',
                'taxonomy': {
                    'species': 'Thunnus albacares'
                }
            }
        ]
    }

@pytest.fixture  
def sample_fasta_file():
    """Mock FASTA file for testing"""
    content = """>seq1 Marine sample 1
ATCGATCGATCGATCGATCGATCG
>seq2 Marine sample 2  
GCTAGCTAGCTAGCTAGCTAGCTA
>seq3 Marine sample 3
TTTTTTTTTTTTTTTTTTTTTTTT
"""
    return io.StringIO(content)

# Run tests with coverage
if __name__ == '__main__':
    pytest.main([
        __file__,
        '-v',
        '--cov=backend.app.genomics',
        '--cov-report=html',
        '--cov-report=term-missing'
    ])