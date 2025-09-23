"""
🧬 Taxonomic Classifier - Species Identification from eDNA Sequences

Advanced taxonomic classification for environmental DNA data.
Implements BLAST-like alignment, k-mer based classification, and phylogenetic placement.

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
from dataclasses import dataclass
from enum import Enum
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class ClassificationMethod(Enum):
    """Classification methods for taxonomic assignment"""
    BLAST_LIKE = "blast_like"
    KMER_BASED = "kmer_based"
    NAIVE_BAYES = "naive_bayes"
    PHYLOGENETIC = "phylogenetic"
    CONSENSUS = "consensus"

class TaxonomicRank(Enum):
    """Standard taxonomic ranks"""
    KINGDOM = "kingdom"
    PHYLUM = "phylum"
    CLASS = "class"
    ORDER = "order"
    FAMILY = "family"
    GENUS = "genus"
    SPECIES = "species"

@dataclass
class TaxonomicAssignment:
    """Represents a taxonomic assignment with confidence metrics"""
    query_id: str
    taxonomy: Dict[TaxonomicRank, str]
    confidence_scores: Dict[TaxonomicRank, float]
    method_used: ClassificationMethod
    best_match: Optional[str] = None
    alignment_score: Optional[float] = None
    e_value: Optional[float] = None
    percent_identity: Optional[float] = None
    query_coverage: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ReferenceSequence:
    """Reference sequence with taxonomic information"""
    sequence_id: str
    sequence: str
    taxonomy: Dict[TaxonomicRank, str]
    source_database: str = "custom"
    accession: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class TaxonomicClassifier:
    """
    🧬 Advanced Taxonomic Classification Engine
    
    Provides comprehensive taxonomic classification capabilities:
    - BLAST-like sequence alignment and scoring
    - K-mer based rapid classification
    - Naive Bayes probabilistic classification
    - Phylogenetic placement methods
    - Consensus classification from multiple methods
    - Confidence estimation and uncertainty quantification
    - Database management and caching
    """
    
    def __init__(self):
        """Initialize the taxonomic classifier"""
        self.reference_database = {}  # sequence_id -> ReferenceSequence
        self.kmer_database = {}  # k-mer -> list of sequence_ids
        self.taxonomy_tree = {}  # hierarchical taxonomy structure
        self.classification_cache = {}  # query_sequence -> classification results
        
        # Default parameters
        self.default_kmer_size = 8
        self.min_alignment_length = 50
        self.min_identity_threshold = 0.7
        self.e_value_threshold = 1e-5
        
        # Marine species reference data (sample entries)
        self._initialize_marine_reference_data()
    
    def add_reference_sequences(self, reference_sequences: List[ReferenceSequence]) -> Dict[str, Any]:
        """
        Add reference sequences to the classification database
        
        Args:
            reference_sequences: List of reference sequences with taxonomy
            
        Returns:
            Database update results
        """
        try:
            added_count = 0
            updated_count = 0
            errors = []
            
            for ref_seq in reference_sequences:
                try:
                    # Validate sequence
                    if not self._validate_dna_sequence(ref_seq.sequence):
                        errors.append(f"Invalid sequence format: {ref_seq.sequence_id}")
                        continue
                    
                    # Check if sequence already exists
                    if ref_seq.sequence_id in self.reference_database:
                        updated_count += 1
                    else:
                        added_count += 1
                    
                    # Add to main database
                    self.reference_database[ref_seq.sequence_id] = ref_seq
                    
                    # Update k-mer index
                    self._update_kmer_index(ref_seq)
                    
                    # Update taxonomy tree
                    self._update_taxonomy_tree(ref_seq.taxonomy)
                    
                except Exception as e:
                    errors.append(f"Error processing {ref_seq.sequence_id}: {str(e)}")
            
            # Clear cache when database is updated
            self.classification_cache.clear()
            
            results = {
                'sequences_added': added_count,
                'sequences_updated': updated_count,
                'total_references': len(self.reference_database),
                'errors': errors,
                'database_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🧬 Reference database updated: {added_count + updated_count} sequences processed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Reference database update failed: {e}")
            return {'error': str(e)}
    
    def classify_sequence(self, 
                         query_sequence: str,
                         query_id: str = None,
                         method: ClassificationMethod = ClassificationMethod.CONSENSUS,
                         min_confidence: float = 0.5) -> TaxonomicAssignment:
        """
        Classify a single DNA sequence taxonomically
        
        Args:
            query_sequence: DNA sequence to classify
            query_id: Identifier for the query sequence
            method: Classification method to use
            min_confidence: Minimum confidence threshold
            
        Returns:
            Taxonomic assignment with confidence scores
        """
        try:
            if query_id is None:
                query_id = f"query_{hash(query_sequence) % 100000}"
            
            # Validate query sequence
            if not self._validate_dna_sequence(query_sequence):
                return TaxonomicAssignment(
                    query_id=query_id,
                    taxonomy={},
                    confidence_scores={},
                    method_used=method,
                    metadata={'error': 'Invalid DNA sequence format'}
                )
            
            # Check cache first
            cache_key = f"{query_sequence}_{method.value}_{min_confidence}"
            if cache_key in self.classification_cache:
                cached_result = self.classification_cache[cache_key]
                cached_result.query_id = query_id  # Update query ID
                return cached_result
            
            # Perform classification based on method
            if method == ClassificationMethod.BLAST_LIKE:
                assignment = self._blast_like_classification(query_sequence, query_id)
            elif method == ClassificationMethod.KMER_BASED:
                assignment = self._kmer_classification(query_sequence, query_id)
            elif method == ClassificationMethod.NAIVE_BAYES:
                assignment = self._naive_bayes_classification(query_sequence, query_id)
            elif method == ClassificationMethod.PHYLOGENETIC:
                assignment = self._phylogenetic_classification(query_sequence, query_id)
            elif method == ClassificationMethod.CONSENSUS:
                assignment = self._consensus_classification(query_sequence, query_id, min_confidence)
            else:
                return TaxonomicAssignment(
                    query_id=query_id,
                    taxonomy={},
                    confidence_scores={},
                    method_used=method,
                    metadata={'error': f'Unsupported method: {method.value}'}
                )
            
            # Filter by confidence threshold
            filtered_taxonomy = {}
            filtered_confidence = {}
            
            for rank, taxon in assignment.taxonomy.items():
                confidence = assignment.confidence_scores.get(rank, 0.0)
                if confidence >= min_confidence:
                    filtered_taxonomy[rank] = taxon
                    filtered_confidence[rank] = confidence
            
            assignment.taxonomy = filtered_taxonomy
            assignment.confidence_scores = filtered_confidence
            
            # Cache result
            self.classification_cache[cache_key] = assignment
            
            logger.info(f"🧬 Sequence classified: {query_id} -> {assignment.taxonomy.get(TaxonomicRank.SPECIES, 'Unknown')}")
            return assignment
            
        except Exception as e:
            logger.error(f"❌ Classification failed for {query_id}: {e}")
            return TaxonomicAssignment(
                query_id=query_id or "unknown",
                taxonomy={},
                confidence_scores={},
                method_used=method,
                metadata={'error': str(e)}
            )
    
    def classify_batch(self, 
                      query_sequences: Dict[str, str],
                      method: ClassificationMethod = ClassificationMethod.CONSENSUS,
                      min_confidence: float = 0.5) -> Dict[str, TaxonomicAssignment]:
        """
        Classify multiple DNA sequences taxonomically
        
        Args:
            query_sequences: Dictionary of sequence_id -> DNA sequence
            method: Classification method to use
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary of sequence_id -> taxonomic assignment
        """
        try:
            results = {}
            total_sequences = len(query_sequences)
            
            for i, (seq_id, sequence) in enumerate(query_sequences.items()):
                if i % 10 == 0:  # Progress logging
                    logger.info(f"🧬 Processing batch: {i+1}/{total_sequences}")
                
                assignment = self.classify_sequence(sequence, seq_id, method, min_confidence)
                results[seq_id] = assignment
            
            logger.info(f"🧬 Batch classification completed: {len(results)} sequences processed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch classification failed: {e}")
            return {}
    
    def get_classification_summary(self, assignments: Dict[str, TaxonomicAssignment]) -> Dict[str, Any]:
        """
        Generate summary statistics from classification results
        
        Args:
            assignments: Dictionary of taxonomic assignments
            
        Returns:
            Classification summary with statistics
        """
        try:
            if not assignments:
                return {'error': 'No assignments provided'}
            
            # Count classifications by taxonomic level
            rank_counts = {rank: Counter() for rank in TaxonomicRank}
            method_counts = Counter()
            confidence_stats = {rank: [] for rank in TaxonomicRank}
            
            classified_count = 0
            unclassified_count = 0
            
            for seq_id, assignment in assignments.items():
                if assignment.taxonomy:
                    classified_count += 1
                else:
                    unclassified_count += 1
                
                # Count taxonomic assignments
                for rank in TaxonomicRank:
                    if rank in assignment.taxonomy:
                        taxon = assignment.taxonomy[rank]
                        rank_counts[rank][taxon] += 1
                        
                        # Collect confidence scores
                        confidence = assignment.confidence_scores.get(rank, 0.0)
                        confidence_stats[rank].append(confidence)
                
                # Count methods used
                method_counts[assignment.method_used.value] += 1
            
            # Calculate statistics
            rank_diversity = {}
            confidence_summary = {}
            
            for rank in TaxonomicRank:
                counts = rank_counts[rank]
                confidences = confidence_stats[rank]
                
                rank_diversity[rank.value] = {
                    'unique_taxa': len(counts),
                    'most_common': counts.most_common(5),
                    'total_assignments': sum(counts.values())
                }
                
                if confidences:
                    confidence_summary[rank.value] = {
                        'mean_confidence': round(np.mean(confidences), 3),
                        'std_confidence': round(np.std(confidences), 3),
                        'min_confidence': round(min(confidences), 3),
                        'max_confidence': round(max(confidences), 3)
                    }
            
            summary = {
                'total_sequences': len(assignments),
                'classified_sequences': classified_count,
                'unclassified_sequences': unclassified_count,
                'classification_rate': round(classified_count / len(assignments), 3),
                'methods_used': dict(method_counts),
                'taxonomic_diversity': rank_diversity,
                'confidence_statistics': confidence_summary,
                'database_size': len(self.reference_database),
                'summary_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🧬 Classification summary generated: {classified_count}/{len(assignments)} classified")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Summary generation failed: {e}")
            return {'error': str(e)}
    
    def _initialize_marine_reference_data(self):
        """Initialize with sample marine species reference data"""
        try:
            # Sample marine species data (in practice, this would come from databases like GenBank, BOLD, etc.)
            sample_references = [
                ReferenceSequence(
                    sequence_id="sample_fish_001",
                    sequence="ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC",
                    taxonomy={
                        TaxonomicRank.KINGDOM: "Animalia",
                        TaxonomicRank.PHYLUM: "Chordata",
                        TaxonomicRank.CLASS: "Actinopterygii",
                        TaxonomicRank.ORDER: "Perciformes",
                        TaxonomicRank.FAMILY: "Scombridae",
                        TaxonomicRank.GENUS: "Thunnus",
                        TaxonomicRank.SPECIES: "Thunnus albacares"
                    },
                    source_database="custom_marine",
                    description="Yellowfin tuna COI gene"
                ),
                ReferenceSequence(
                    sequence_id="sample_coral_001",
                    sequence="GCTATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGC",
                    taxonomy={
                        TaxonomicRank.KINGDOM: "Animalia",
                        TaxonomicRank.PHYLUM: "Cnidaria",
                        TaxonomicRank.CLASS: "Anthozoa",
                        TaxonomicRank.ORDER: "Scleractinia",
                        TaxonomicRank.FAMILY: "Acroporidae",
                        TaxonomicRank.GENUS: "Acropora",
                        TaxonomicRank.SPECIES: "Acropora cervicornis"
                    },
                    source_database="custom_marine",
                    description="Staghorn coral 18S rRNA"
                ),
                # Add more sample references...
            ]
            
            self.add_reference_sequences(sample_references)
            logger.info("🧬 Marine reference database initialized with sample data")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize reference data: {e}")
    
    def _validate_dna_sequence(self, sequence: str) -> bool:
        """Validate DNA sequence format"""
        if not sequence:
            return False
        
        # Check for valid DNA bases (including ambiguous bases)
        valid_bases = set('ATGCNRYWSMKHBVD-')
        return all(base.upper() in valid_bases for base in sequence)
    
    def _update_kmer_index(self, ref_seq: ReferenceSequence):
        """Update k-mer index with new reference sequence"""
        sequence = ref_seq.sequence.upper()
        
        for i in range(len(sequence) - self.default_kmer_size + 1):
            kmer = sequence[i:i + self.default_kmer_size]
            
            if 'N' not in kmer:  # Skip k-mers with ambiguous bases
                if kmer not in self.kmer_database:
                    self.kmer_database[kmer] = []
                
                if ref_seq.sequence_id not in self.kmer_database[kmer]:
                    self.kmer_database[kmer].append(ref_seq.sequence_id)
    
    def _update_taxonomy_tree(self, taxonomy: Dict[TaxonomicRank, str]):
        """Update hierarchical taxonomy tree"""
        current_level = self.taxonomy_tree
        
        # Build hierarchical structure
        rank_order = [TaxonomicRank.KINGDOM, TaxonomicRank.PHYLUM, TaxonomicRank.CLASS,
                     TaxonomicRank.ORDER, TaxonomicRank.FAMILY, TaxonomicRank.GENUS, TaxonomicRank.SPECIES]
        
        for rank in rank_order:
            if rank in taxonomy:
                taxon = taxonomy[rank]
                if taxon not in current_level:
                    current_level[taxon] = {}
                current_level = current_level[taxon]
    
    def _blast_like_classification(self, query_sequence: str, query_id: str) -> TaxonomicAssignment:
        """BLAST-like alignment-based classification"""
        try:
            best_matches = []
            query_seq_upper = query_sequence.upper()
            
            # Find best alignments
            for ref_id, ref_seq in self.reference_database.items():
                alignment_score = self._calculate_alignment_score(query_seq_upper, ref_seq.sequence.upper())
                
                if alignment_score['percent_identity'] >= self.min_identity_threshold:
                    best_matches.append({
                        'reference_id': ref_id,
                        'reference_seq': ref_seq,
                        'alignment_score': alignment_score
                    })
            
            if not best_matches:
                return TaxonomicAssignment(
                    query_id=query_id,
                    taxonomy={},
                    confidence_scores={},
                    method_used=ClassificationMethod.BLAST_LIKE,
                    metadata={'message': 'No significant matches found'}
                )
            
            # Sort by alignment score
            best_matches.sort(key=lambda x: x['alignment_score']['percent_identity'], reverse=True)
            best_match = best_matches[0]
            
            # Assign taxonomy from best match
            taxonomy = best_match['reference_seq'].taxonomy.copy()
            
            # Calculate confidence scores based on alignment quality and consensus
            confidence_scores = self._calculate_blast_confidence(best_matches, query_seq_upper)
            
            return TaxonomicAssignment(
                query_id=query_id,
                taxonomy=taxonomy,
                confidence_scores=confidence_scores,
                method_used=ClassificationMethod.BLAST_LIKE,
                best_match=best_match['reference_id'],
                alignment_score=best_match['alignment_score']['percent_identity'],
                percent_identity=best_match['alignment_score']['percent_identity'],
                query_coverage=best_match['alignment_score']['coverage']
            )
            
        except Exception as e:
            return TaxonomicAssignment(
                query_id=query_id,
                taxonomy={},
                confidence_scores={},
                method_used=ClassificationMethod.BLAST_LIKE,
                metadata={'error': str(e)}
            )
    
    def _kmer_classification(self, query_sequence: str, query_id: str) -> TaxonomicAssignment:
        """K-mer based rapid classification"""
        try:
            query_kmers = set()
            query_seq_upper = query_sequence.upper()
            
            # Extract k-mers from query
            for i in range(len(query_seq_upper) - self.default_kmer_size + 1):
                kmer = query_seq_upper[i:i + self.default_kmer_size]
                if 'N' not in kmer:
                    query_kmers.add(kmer)
            
            if not query_kmers:
                return TaxonomicAssignment(
                    query_id=query_id,
                    taxonomy={},
                    confidence_scores={},
                    method_used=ClassificationMethod.KMER_BASED,
                    metadata={'message': 'No valid k-mers found'}
                )
            
            # Find matching references
            reference_scores = defaultdict(int)
            
            for kmer in query_kmers:
                if kmer in self.kmer_database:
                    for ref_id in self.kmer_database[kmer]:
                        reference_scores[ref_id] += 1
            
            if not reference_scores:
                return TaxonomicAssignment(
                    query_id=query_id,
                    taxonomy={},
                    confidence_scores={},
                    method_used=ClassificationMethod.KMER_BASED,
                    metadata={'message': 'No k-mer matches found'}
                )
            
            # Get best match
            best_ref_id = max(reference_scores, key=reference_scores.get)
            best_ref = self.reference_database[best_ref_id]
            
            # Calculate k-mer similarity score
            kmer_score = reference_scores[best_ref_id] / len(query_kmers)
            
            # Assign taxonomy and confidence
            taxonomy = best_ref.taxonomy.copy()
            confidence_scores = {rank: min(kmer_score * 1.2, 1.0) for rank in taxonomy.keys()}
            
            return TaxonomicAssignment(
                query_id=query_id,
                taxonomy=taxonomy,
                confidence_scores=confidence_scores,
                method_used=ClassificationMethod.KMER_BASED,
                best_match=best_ref_id,
                alignment_score=kmer_score
            )
            
        except Exception as e:
            return TaxonomicAssignment(
                query_id=query_id,
                taxonomy={},
                confidence_scores={},
                method_used=ClassificationMethod.KMER_BASED,
                metadata={'error': str(e)}
            )
    
    def _naive_bayes_classification(self, query_sequence: str, query_id: str) -> TaxonomicAssignment:
        """Naive Bayes probabilistic classification"""
        try:
            # Simplified Naive Bayes based on k-mer frequencies
            query_kmers = Counter()
            query_seq_upper = query_sequence.upper()
            
            # Count k-mers in query
            for i in range(len(query_seq_upper) - self.default_kmer_size + 1):
                kmer = query_seq_upper[i:i + self.default_kmer_size]
                if 'N' not in kmer:
                    query_kmers[kmer] += 1
            
            if not query_kmers:
                return TaxonomicAssignment(
                    query_id=query_id,
                    taxonomy={},
                    confidence_scores={},
                    method_used=ClassificationMethod.NAIVE_BAYES,
                    metadata={'message': 'No valid k-mers for classification'}
                )
            
            # Calculate probabilities for each reference
            reference_probabilities = {}
            
            for ref_id, ref_seq in self.reference_database.items():
                # Calculate log probability
                log_prob = 0.0
                ref_kmers = Counter()
                
                # Count k-mers in reference
                ref_sequence = ref_seq.sequence.upper()
                for i in range(len(ref_sequence) - self.default_kmer_size + 1):
                    kmer = ref_sequence[i:i + self.default_kmer_size]
                    if 'N' not in kmer:
                        ref_kmers[kmer] += 1
                
                # Calculate probability based on k-mer overlap
                total_ref_kmers = sum(ref_kmers.values())
                
                for kmer, count in query_kmers.items():
                    kmer_prob = (ref_kmers.get(kmer, 0) + 1) / (total_ref_kmers + len(self.kmer_database))
                    log_prob += count * np.log(kmer_prob)
                
                reference_probabilities[ref_id] = log_prob
            
            if not reference_probabilities:
                return TaxonomicAssignment(
                    query_id=query_id,
                    taxonomy={},
                    confidence_scores={},
                    method_used=ClassificationMethod.NAIVE_BAYES,
                    metadata={'message': 'No probability calculations possible'}
                )
            
            # Get best match
            best_ref_id = max(reference_probabilities, key=reference_probabilities.get)
            best_ref = self.reference_database[best_ref_id]
            
            # Normalize probabilities for confidence
            max_prob = max(reference_probabilities.values())
            normalized_prob = np.exp(reference_probabilities[best_ref_id] - max_prob)
            
            taxonomy = best_ref.taxonomy.copy()
            confidence_scores = {rank: min(normalized_prob * 0.9, 1.0) for rank in taxonomy.keys()}
            
            return TaxonomicAssignment(
                query_id=query_id,
                taxonomy=taxonomy,
                confidence_scores=confidence_scores,
                method_used=ClassificationMethod.NAIVE_BAYES,
                best_match=best_ref_id,
                alignment_score=normalized_prob
            )
            
        except Exception as e:
            return TaxonomicAssignment(
                query_id=query_id,
                taxonomy={},
                confidence_scores={},
                method_used=ClassificationMethod.NAIVE_BAYES,
                metadata={'error': str(e)}
            )
    
    def _phylogenetic_classification(self, query_sequence: str, query_id: str) -> TaxonomicAssignment:
        """Phylogenetic placement-based classification"""
        try:
            # Simplified phylogenetic placement using distance-based approach
            
            # Find closest references by evolutionary distance
            distances = []
            
            for ref_id, ref_seq in self.reference_database.items():
                # Calculate evolutionary distance (simplified)
                distance = self._calculate_evolutionary_distance(query_sequence.upper(), ref_seq.sequence.upper())
                distances.append((distance, ref_id, ref_seq))
            
            if not distances:
                return TaxonomicAssignment(
                    query_id=query_id,
                    taxonomy={},
                    confidence_scores={},
                    method_used=ClassificationMethod.PHYLOGENETIC,
                    metadata={'message': 'No references for phylogenetic placement'}
                )
            
            # Sort by distance (closest first)
            distances.sort(key=lambda x: x[0])
            closest_distance, best_ref_id, best_ref = distances[0]
            
            # Use phylogenetic distance to determine confidence
            # Closer sequences get higher confidence
            max_confidence = 1.0 / (1.0 + closest_distance)
            
            taxonomy = best_ref.taxonomy.copy()
            confidence_scores = {rank: max_confidence * 0.9 for rank in taxonomy.keys()}
            
            return TaxonomicAssignment(
                query_id=query_id,
                taxonomy=taxonomy,
                confidence_scores=confidence_scores,
                method_used=ClassificationMethod.PHYLOGENETIC,
                best_match=best_ref_id,
                alignment_score=1.0 - closest_distance
            )
            
        except Exception as e:
            return TaxonomicAssignment(
                query_id=query_id,
                taxonomy={},
                confidence_scores={},
                method_used=ClassificationMethod.PHYLOGENETIC,
                metadata={'error': str(e)}
            )
    
    def _consensus_classification(self, query_sequence: str, query_id: str, min_confidence: float) -> TaxonomicAssignment:
        """Consensus classification using multiple methods"""
        try:
            # Run multiple classification methods
            methods = [
                ClassificationMethod.BLAST_LIKE,
                ClassificationMethod.KMER_BASED,
                ClassificationMethod.NAIVE_BAYES
            ]
            
            method_results = []
            
            for method in methods:
                try:
                    if method == ClassificationMethod.BLAST_LIKE:
                        result = self._blast_like_classification(query_sequence, query_id)
                    elif method == ClassificationMethod.KMER_BASED:
                        result = self._kmer_classification(query_sequence, query_id)
                    elif method == ClassificationMethod.NAIVE_BAYES:
                        result = self._naive_bayes_classification(query_sequence, query_id)
                    
                    if result.taxonomy:  # Only include results with classifications
                        method_results.append(result)
                        
                except Exception as e:
                    logger.warning(f"⚠️ Method {method.value} failed: {e}")
                    continue
            
            if not method_results:
                return TaxonomicAssignment(
                    query_id=query_id,
                    taxonomy={},
                    confidence_scores={},
                    method_used=ClassificationMethod.CONSENSUS,
                    metadata={'message': 'No valid results from any method'}
                )
            
            # Build consensus taxonomy
            consensus_taxonomy = {}
            consensus_confidence = {}
            
            for rank in TaxonomicRank:
                rank_assignments = []
                rank_confidences = []
                
                for result in method_results:
                    if rank in result.taxonomy:
                        rank_assignments.append(result.taxonomy[rank])
                        rank_confidences.append(result.confidence_scores.get(rank, 0.0))
                
                if rank_assignments:
                    # Find most common assignment
                    assignment_counts = Counter(rank_assignments)
                    most_common_assignment, count = assignment_counts.most_common(1)[0]
                    
                    # Calculate consensus confidence
                    agreement_ratio = count / len(rank_assignments)
                    mean_confidence = np.mean([conf for assign, conf in zip(rank_assignments, rank_confidences) 
                                             if assign == most_common_assignment])
                    
                    consensus_conf = agreement_ratio * mean_confidence
                    
                    if consensus_conf >= min_confidence:
                        consensus_taxonomy[rank] = most_common_assignment
                        consensus_confidence[rank] = round(consensus_conf, 3)
            
            # Find best individual result for metadata
            best_result = max(method_results, key=lambda x: max(x.confidence_scores.values()) if x.confidence_scores else 0)
            
            return TaxonomicAssignment(
                query_id=query_id,
                taxonomy=consensus_taxonomy,
                confidence_scores=consensus_confidence,
                method_used=ClassificationMethod.CONSENSUS,
                best_match=best_result.best_match,
                alignment_score=best_result.alignment_score,
                metadata={
                    'methods_used': [r.method_used.value for r in method_results],
                    'total_methods': len(method_results),
                    'consensus_agreement': len(consensus_taxonomy)
                }
            )
            
        except Exception as e:
            return TaxonomicAssignment(
                query_id=query_id,
                taxonomy={},
                confidence_scores={},
                method_used=ClassificationMethod.CONSENSUS,
                metadata={'error': str(e)}
            )
    
    def _calculate_alignment_score(self, seq1: str, seq2: str) -> Dict[str, float]:
        """Calculate alignment score between two sequences"""
        try:
            # Use SequenceMatcher for basic alignment scoring
            matcher = SequenceMatcher(None, seq1, seq2)
            
            # Get matching blocks
            matches = matcher.get_matching_blocks()
            total_matches = sum(match.size for match in matches)
            
            # Calculate metrics
            percent_identity = (total_matches / max(len(seq1), len(seq2))) * 100
            coverage = (total_matches / len(seq1)) * 100
            
            # Simple e-value approximation
            e_value = 10 ** (-(percent_identity / 10))
            
            return {
                'percent_identity': round(percent_identity, 2),
                'coverage': round(coverage, 2),
                'matches': total_matches,
                'alignment_length': max(len(seq1), len(seq2)),
                'e_value': e_value
            }
            
        except Exception as e:
            return {
                'percent_identity': 0.0,
                'coverage': 0.0,
                'matches': 0,
                'alignment_length': 0,
                'e_value': 1.0
            }
    
    def _calculate_blast_confidence(self, matches: List[Dict], query_sequence: str) -> Dict[TaxonomicRank, float]:
        """Calculate confidence scores for BLAST-like results"""
        confidence_scores = {}
        
        if not matches:
            return confidence_scores
        
        # Group matches by taxonomic rank
        for rank in TaxonomicRank:
            rank_assignments = []
            rank_scores = []
            
            for match in matches[:10]:  # Consider top 10 matches
                ref_seq = match['reference_seq']
                if rank in ref_seq.taxonomy:
                    rank_assignments.append(ref_seq.taxonomy[rank])
                    rank_scores.append(match['alignment_score']['percent_identity'])
            
            if rank_assignments:
                # Calculate confidence based on consensus and quality
                assignment_counts = Counter(rank_assignments)
                most_common_assignment, count = assignment_counts.most_common(1)[0]
                
                # Agreement ratio
                agreement = count / len(rank_assignments)
                
                # Average score for most common assignment
                avg_score = np.mean([score for assign, score in zip(rank_assignments, rank_scores)
                                   if assign == most_common_assignment])
                
                # Combined confidence
                confidence = (agreement * 0.6 + (avg_score / 100) * 0.4)
                confidence_scores[rank] = round(confidence, 3)
        
        return confidence_scores
    
    def _calculate_evolutionary_distance(self, seq1: str, seq2: str) -> float:
        """Calculate evolutionary distance between sequences (simplified)"""
        try:
            if len(seq1) != len(seq2):
                # Pad shorter sequence or truncate longer one
                min_len = min(len(seq1), len(seq2))
                seq1 = seq1[:min_len]
                seq2 = seq2[:min_len]
            
            if not seq1 or not seq2:
                return 1.0  # Maximum distance for empty sequences
            
            # Count differences
            differences = sum(c1 != c2 for c1, c2 in zip(seq1, seq2))
            
            # P-distance
            p_distance = differences / len(seq1)
            
            # Jukes-Cantor correction (simplified)
            if p_distance >= 0.75:
                return 3.0  # Saturated distance
            else:
                jc_distance = -0.75 * np.log(1 - (4/3) * p_distance)
                return max(0.0, jc_distance)
            
        except Exception as e:
            return 1.0  # Return maximum distance on error

# Global taxonomic classifier instance
taxonomic_classifier = TaxonomicClassifier()