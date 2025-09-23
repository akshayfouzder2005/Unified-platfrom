"""
🧬 Sequence Processor - DNA Sequence Analysis & Quality Control

Advanced sequence processing for environmental DNA analysis.
Implements quality assessment, filtering, trimming, and preprocessing.

Author: Ocean-Bio Development Team
Version: 2.0.0
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import re
from collections import Counter, defaultdict

# BioPython imports
try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio.SeqUtils import GC, molecular_weight
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

logger = logging.getLogger(__name__)

class SequenceProcessor:
    """
    🧬 Advanced DNA Sequence Processor
    
    Provides comprehensive sequence analysis capabilities:
    - Quality assessment and filtering
    - Sequence trimming and cleaning
    - Primer detection and removal
    - N-base handling and gap analysis
    - Sequence statistics and metrics
    - Format conversion and standardization
    """
    
    def __init__(self):
        """Initialize the sequence processor"""
        self.quality_thresholds = {
            'min_length': 100,
            'max_length': 2000,
            'min_gc_content': 0.2,
            'max_gc_content': 0.8,
            'max_n_content': 0.05,
            'min_complexity': 0.3
        }
        
        # Common primer sequences for eDNA
        self.common_primers = {
            'COI': {
                'forward': ['GGTCAACAAATCATAAAGATATTGG', 'TCAACAAATCATAAAGATATTGG'],
                'reverse': ['TAAACTTCAGGGTGACCAAAAA', 'AAACTTCAGGGTGACCAAAAA']
            },
            '16S': {
                'forward': ['AGAGTTTGATCMTGGCTCAG', 'GAGTTTGATCMTGGCTCAG'],
                'reverse': ['TACGGYTACCTTGTTACGACTT', 'CGGYTACCTTGTTACGACTT']
            },
            '18S': {
                'forward': ['AACCTGGTTGATCCTGCCAGT', 'CCTGGTTGATCCTGCCAGT'],
                'reverse': ['TGATCCTTCTGCAGGTTCACCTAC', 'TCCTTCTGCAGGTTCACCTAC']
            }
        }
    
    def quality_assessment(self, sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform comprehensive quality assessment of sequences
        
        Args:
            sequences: List of sequence dictionaries with 'id', 'sequence', and optional 'quality'
            
        Returns:
            Quality assessment results
        """
        try:
            if not BIOPYTHON_AVAILABLE:
                return {'error': 'BioPython is required for sequence analysis'}
            
            results = {
                'total_sequences': len(sequences),
                'sequence_statistics': [],
                'overall_statistics': {},
                'quality_flags': [],
                'assessment_timestamp': datetime.now().isoformat()
            }
            
            # Initialize counters
            length_stats = []
            gc_contents = []
            n_contents = []
            quality_scores = []
            
            for i, seq_data in enumerate(sequences):
                seq_id = seq_data.get('id', f'seq_{i}')
                sequence = seq_data.get('sequence', '').upper()
                quality = seq_data.get('quality', None)
                
                if not sequence:
                    continue
                
                # Basic sequence statistics
                seq_length = len(sequence)
                gc_content = self._calculate_gc_content(sequence)
                n_content = sequence.count('N') / seq_length if seq_length > 0 else 0
                
                # Calculate sequence complexity (entropy-based)
                complexity = self._calculate_complexity(sequence)
                
                # Quality score statistics (if available)
                avg_quality = None
                if quality and len(quality) == seq_length:
                    avg_quality = np.mean([ord(q) - 33 for q in quality])  # Phred+33 encoding
                    quality_scores.append(avg_quality)
                
                # Identify potential issues
                issues = []
                if seq_length < self.quality_thresholds['min_length']:
                    issues.append('Too short')
                if seq_length > self.quality_thresholds['max_length']:
                    issues.append('Too long')
                if gc_content < self.quality_thresholds['min_gc_content']:
                    issues.append('Low GC content')
                if gc_content > self.quality_thresholds['max_gc_content']:
                    issues.append('High GC content')
                if n_content > self.quality_thresholds['max_n_content']:
                    issues.append('High N content')
                if complexity < self.quality_thresholds['min_complexity']:
                    issues.append('Low complexity')
                
                # Store sequence statistics
                seq_stats = {
                    'sequence_id': seq_id,
                    'length': seq_length,
                    'gc_content': round(gc_content, 4),
                    'n_content': round(n_content, 4),
                    'complexity': round(complexity, 4),
                    'average_quality': round(avg_quality, 2) if avg_quality else None,
                    'issues': issues,
                    'is_valid': len(issues) == 0
                }
                
                results['sequence_statistics'].append(seq_stats)
                
                # Collect for overall statistics
                length_stats.append(seq_length)
                gc_contents.append(gc_content)
                n_contents.append(n_content)
            
            # Calculate overall statistics
            if length_stats:
                results['overall_statistics'] = {
                    'sequence_count': len(length_stats),
                    'length_statistics': {
                        'mean': round(np.mean(length_stats), 2),
                        'median': round(np.median(length_stats), 2),
                        'std': round(np.std(length_stats), 2),
                        'min': int(np.min(length_stats)),
                        'max': int(np.max(length_stats))
                    },
                    'gc_content_statistics': {
                        'mean': round(np.mean(gc_contents), 4),
                        'std': round(np.std(gc_contents), 4),
                        'min': round(np.min(gc_contents), 4),
                        'max': round(np.max(gc_contents), 4)
                    },
                    'n_content_statistics': {
                        'mean': round(np.mean(n_contents), 4),
                        'max': round(np.max(n_contents), 4)
                    },
                    'quality_statistics': {
                        'mean': round(np.mean(quality_scores), 2) if quality_scores else None,
                        'std': round(np.std(quality_scores), 2) if quality_scores else None
                    } if quality_scores else None
                }
                
                # Generate quality flags
                valid_sequences = sum(1 for s in results['sequence_statistics'] if s['is_valid'])
                results['quality_summary'] = {
                    'total_sequences': len(sequences),
                    'valid_sequences': valid_sequences,
                    'invalid_sequences': len(sequences) - valid_sequences,
                    'pass_rate': round(valid_sequences / len(sequences) * 100, 2) if sequences else 0
                }
            
            logger.info(f"🧬 Quality assessment completed: {len(sequences)} sequences analyzed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Quality assessment failed: {e}")
            return {'error': str(e)}
    
    def trim_sequences(self, 
                      sequences: List[Dict[str, Any]],
                      trim_start: int = 0,
                      trim_end: int = 0,
                      quality_threshold: int = 20) -> Dict[str, Any]:
        """
        Trim sequences based on position and quality
        
        Args:
            sequences: List of sequence dictionaries
            trim_start: Bases to trim from start
            trim_end: Bases to trim from end
            quality_threshold: Minimum quality score for trimming
            
        Returns:
            Trimmed sequences results
        """
        try:
            trimmed_sequences = []
            trimming_stats = {
                'original_count': len(sequences),
                'trimmed_count': 0,
                'removed_count': 0,
                'average_length_before': 0,
                'average_length_after': 0
            }
            
            original_lengths = []
            trimmed_lengths = []
            
            for seq_data in sequences:
                seq_id = seq_data.get('id', '')
                sequence = seq_data.get('sequence', '').upper()
                quality = seq_data.get('quality', None)
                
                original_length = len(sequence)
                original_lengths.append(original_length)
                
                # Position-based trimming
                start_pos = trim_start
                end_pos = len(sequence) - trim_end if trim_end > 0 else len(sequence)
                
                trimmed_seq = sequence[start_pos:end_pos]
                trimmed_qual = quality[start_pos:end_pos] if quality else None
                
                # Quality-based trimming (if quality scores available)
                if quality and len(quality) == original_length:
                    # Trim from 3' end based on quality
                    qual_scores = [ord(q) - 33 for q in trimmed_qual]
                    
                    # Find last position with acceptable quality
                    last_good_pos = len(qual_scores)
                    for i in range(len(qual_scores) - 1, -1, -1):
                        if qual_scores[i] >= quality_threshold:
                            last_good_pos = i + 1
                            break
                        
                    trimmed_seq = trimmed_seq[:last_good_pos]
                    trimmed_qual = trimmed_qual[:last_good_pos] if trimmed_qual else None
                
                # Only keep sequences with reasonable length
                if len(trimmed_seq) >= self.quality_thresholds['min_length']:
                    trimmed_data = {
                        'id': seq_id,
                        'sequence': trimmed_seq,
                        'quality': trimmed_qual,
                        'original_length': original_length,
                        'trimmed_length': len(trimmed_seq),
                        'bases_removed': original_length - len(trimmed_seq)
                    }
                    
                    trimmed_sequences.append(trimmed_data)
                    trimmed_lengths.append(len(trimmed_seq))
                    trimming_stats['trimmed_count'] += 1
                else:
                    trimming_stats['removed_count'] += 1
            
            # Calculate statistics
            if original_lengths:
                trimming_stats['average_length_before'] = round(np.mean(original_lengths), 2)
            if trimmed_lengths:
                trimming_stats['average_length_after'] = round(np.mean(trimmed_lengths), 2)
            
            results = {
                'trimmed_sequences': trimmed_sequences,
                'trimming_statistics': trimming_stats,
                'parameters': {
                    'trim_start': trim_start,
                    'trim_end': trim_end,
                    'quality_threshold': quality_threshold
                },
                'trimming_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🧬 Sequence trimming completed: {trimming_stats['trimmed_count']} sequences trimmed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Sequence trimming failed: {e}")
            return {'error': str(e)}
    
    def remove_primers(self, 
                      sequences: List[Dict[str, Any]],
                      primer_type: str = 'auto',
                      custom_primers: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Remove primer sequences from DNA sequences
        
        Args:
            sequences: List of sequence dictionaries
            primer_type: Type of primers ('COI', '16S', '18S', 'auto', or 'custom')
            custom_primers: Custom primer sequences if primer_type is 'custom'
            
        Returns:
            Sequences with primers removed
        """
        try:
            processed_sequences = []
            removal_stats = {
                'total_sequences': len(sequences),
                'primers_detected': 0,
                'primers_removed': 0,
                'primer_matches': defaultdict(int)
            }
            
            # Determine primer set
            if primer_type == 'custom' and custom_primers:
                primers = custom_primers
            elif primer_type in self.common_primers:
                primers = self.common_primers[primer_type]
            elif primer_type == 'auto':
                # Try all known primers
                primers = {}
                for p_type, p_seqs in self.common_primers.items():
                    primers.update(p_seqs)
            else:
                primers = {}
            
            for seq_data in sequences:
                seq_id = seq_data.get('id', '')
                sequence = seq_data.get('sequence', '').upper()
                original_seq = sequence
                
                primers_found = []
                
                # Check for forward primers at start
                if 'forward' in primers:
                    for primer in primers['forward']:
                        primer_upper = primer.upper()
                        # Allow for some mismatches
                        if self._fuzzy_match(sequence[:len(primer_upper) + 5], primer_upper):
                            # Find exact position
                            match_pos = self._find_primer_position(sequence, primer_upper)
                            if match_pos is not None:
                                sequence = sequence[match_pos + len(primer_upper):]
                                primers_found.append(f"forward_{primer}")
                                removal_stats['primer_matches'][f"forward_{primer}"] += 1
                                break
                
                # Check for reverse primers at end
                if 'reverse' in primers:
                    for primer in primers['reverse']:
                        primer_upper = primer.upper()
                        # Check reverse complement as well
                        primer_rc = self._reverse_complement(primer_upper)
                        
                        for p in [primer_upper, primer_rc]:
                            if self._fuzzy_match(sequence[-(len(p) + 5):], p):
                                match_pos = self._find_primer_position_reverse(sequence, p)
                                if match_pos is not None:
                                    sequence = sequence[:match_pos]
                                    primers_found.append(f"reverse_{primer}")
                                    removal_stats['primer_matches'][f"reverse_{primer}"] += 1
                                    break
                        if primers_found and primers_found[-1].startswith('reverse_'):
                            break
                
                # Update statistics
                if primers_found:
                    removal_stats['primers_detected'] += 1
                    if len(sequence) < len(original_seq):
                        removal_stats['primers_removed'] += 1
                
                processed_data = {
                    'id': seq_id,
                    'sequence': sequence,
                    'quality': seq_data.get('quality'),
                    'original_length': len(original_seq),
                    'processed_length': len(sequence),
                    'primers_found': primers_found,
                    'bases_removed': len(original_seq) - len(sequence)
                }
                
                processed_sequences.append(processed_data)
            
            results = {
                'processed_sequences': processed_sequences,
                'removal_statistics': removal_stats,
                'parameters': {
                    'primer_type': primer_type,
                    'primers_used': list(primers.keys()) if primers else []
                },
                'processing_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🧬 Primer removal completed: {removal_stats['primers_removed']} sequences processed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Primer removal failed: {e}")
            return {'error': str(e)}
    
    def filter_sequences(self, 
                        sequences: List[Dict[str, Any]],
                        min_length: Optional[int] = None,
                        max_length: Optional[int] = None,
                        min_gc: Optional[float] = None,
                        max_gc: Optional[float] = None,
                        max_n_content: Optional[float] = None) -> Dict[str, Any]:
        """
        Filter sequences based on various criteria
        
        Args:
            sequences: List of sequence dictionaries
            min_length: Minimum sequence length
            max_length: Maximum sequence length
            min_gc: Minimum GC content
            max_gc: Maximum GC content
            max_n_content: Maximum N content fraction
            
        Returns:
            Filtered sequences results
        """
        try:
            # Use default thresholds if not specified
            min_length = min_length or self.quality_thresholds['min_length']
            max_length = max_length or self.quality_thresholds['max_length']
            min_gc = min_gc or self.quality_thresholds['min_gc_content']
            max_gc = max_gc or self.quality_thresholds['max_gc_content']
            max_n_content = max_n_content or self.quality_thresholds['max_n_content']
            
            filtered_sequences = []
            filter_stats = {
                'input_sequences': len(sequences),
                'output_sequences': 0,
                'filtered_out': {
                    'too_short': 0,
                    'too_long': 0,
                    'low_gc': 0,
                    'high_gc': 0,
                    'high_n_content': 0
                }
            }
            
            for seq_data in sequences:
                sequence = seq_data.get('sequence', '').upper()
                
                # Apply filters
                seq_length = len(sequence)
                gc_content = self._calculate_gc_content(sequence)
                n_content = sequence.count('N') / seq_length if seq_length > 0 else 0
                
                # Check each filter
                passed = True
                
                if seq_length < min_length:
                    filter_stats['filtered_out']['too_short'] += 1
                    passed = False
                elif seq_length > max_length:
                    filter_stats['filtered_out']['too_long'] += 1
                    passed = False
                elif gc_content < min_gc:
                    filter_stats['filtered_out']['low_gc'] += 1
                    passed = False
                elif gc_content > max_gc:
                    filter_stats['filtered_out']['high_gc'] += 1
                    passed = False
                elif n_content > max_n_content:
                    filter_stats['filtered_out']['high_n_content'] += 1
                    passed = False
                
                if passed:
                    filtered_sequences.append(seq_data)
                    filter_stats['output_sequences'] += 1
            
            results = {
                'filtered_sequences': filtered_sequences,
                'filter_statistics': filter_stats,
                'filter_parameters': {
                    'min_length': min_length,
                    'max_length': max_length,
                    'min_gc_content': min_gc,
                    'max_gc_content': max_gc,
                    'max_n_content': max_n_content
                },
                'filtering_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🧬 Sequence filtering completed: {filter_stats['output_sequences']} sequences passed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Sequence filtering failed: {e}")
            return {'error': str(e)}
    
    def _calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content of a sequence"""
        if not sequence:
            return 0.0
        
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence)
    
    def _calculate_complexity(self, sequence: str) -> float:
        """Calculate sequence complexity using Shannon entropy"""
        if not sequence:
            return 0.0
        
        # Count nucleotides
        counts = Counter(sequence)
        total = len(sequence)
        
        # Calculate Shannon entropy
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        # Normalize by maximum possible entropy (log2(4) for DNA)
        max_entropy = 2.0  # log2(4)
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _fuzzy_match(self, sequence: str, primer: str, max_mismatches: int = 2) -> bool:
        """Check for fuzzy match allowing mismatches"""
        if len(sequence) < len(primer):
            return False
        
        # Check for exact match first
        if primer in sequence:
            return True
        
        # Check with mismatches
        for i in range(len(sequence) - len(primer) + 1):
            mismatches = 0
            for j in range(len(primer)):
                if sequence[i + j] != primer[j]:
                    mismatches += 1
                    if mismatches > max_mismatches:
                        break
            
            if mismatches <= max_mismatches:
                return True
        
        return False
    
    def _find_primer_position(self, sequence: str, primer: str) -> Optional[int]:
        """Find exact position of primer in sequence"""
        return sequence.find(primer) if primer in sequence else None
    
    def _find_primer_position_reverse(self, sequence: str, primer: str) -> Optional[int]:
        """Find position of primer from reverse end"""
        pos = sequence.rfind(primer)
        return pos if pos != -1 else None
    
    def _reverse_complement(self, sequence: str) -> str:
        """Generate reverse complement of DNA sequence"""
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        return ''.join(complement.get(base, base) for base in reversed(sequence))

# Global sequence processor instance
sequence_processor = SequenceProcessor()