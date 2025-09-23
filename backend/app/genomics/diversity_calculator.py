"""
🧬 Diversity Calculator - Biodiversity Metrics & Ecological Indices

Advanced biodiversity analysis for environmental DNA data.
Implements alpha, beta, and gamma diversity calculations.

Author: Ocean-Bio Development Team
Version: 2.0.0
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
import math
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import jaccard_score

logger = logging.getLogger(__name__)

class DiversityCalculator:
    """
    🧬 Advanced Biodiversity Metrics Calculator
    
    Provides comprehensive diversity analysis capabilities:
    - Alpha diversity indices (Shannon, Simpson, Chao1, ACE)
    - Beta diversity metrics (Jaccard, Bray-Curtis, UniFrac)
    - Gamma diversity calculations
    - Rarefaction and accumulation curves
    - Community structure analysis
    - Phylogenetic diversity metrics
    """
    
    def __init__(self):
        """Initialize the diversity calculator"""
        self.alpha_indices = [
            'shannon', 'simpson', 'chao1', 'ace', 'pielou_evenness',
            'fisher_alpha', 'berger_parker', 'brillouin', 'mcintosh'
        ]
        
        self.beta_indices = [
            'jaccard', 'bray_curtis', 'sorensen', 'horn_morisita',
            'unifrac_unweighted', 'unifrac_weighted'
        ]
    
    def calculate_alpha_diversity(self, 
                                 abundance_data: Dict[str, int],
                                 sample_id: str = None) -> Dict[str, Any]:
        """
        Calculate alpha diversity indices for a single sample
        
        Args:
            abundance_data: Dictionary of species/OTU -> abundance counts
            sample_id: Sample identifier
            
        Returns:
            Alpha diversity metrics
        """
        try:
            if not abundance_data:
                return {'error': 'No abundance data provided'}
            
            # Convert to arrays
            species_names = list(abundance_data.keys())
            abundances = np.array(list(abundance_data.values()), dtype=float)
            
            # Filter out zero abundances
            abundances = abundances[abundances > 0]
            total_individuals = np.sum(abundances)
            num_species = len(abundances)
            
            if total_individuals == 0:
                return {'error': 'No individuals found in sample'}
            
            # Calculate relative abundances
            proportions = abundances / total_individuals
            
            # Shannon diversity index
            shannon = -np.sum(proportions * np.log(proportions))
            
            # Simpson diversity index (1 - D)
            simpson = 1 - np.sum(proportions ** 2)
            
            # Pielou's evenness
            max_shannon = np.log(num_species)
            pielou_evenness = shannon / max_shannon if max_shannon > 0 else 0
            
            # Berger-Parker dominance
            berger_parker = np.max(proportions)
            
            # Fisher's alpha (approximation)
            if num_species > 1 and total_individuals > num_species:
                # Iterative solution for Fisher's alpha
                fisher_alpha = self._calculate_fisher_alpha(abundances)
            else:
                fisher_alpha = None
            
            # Brillouin diversity
            brillouin = self._calculate_brillouin(abundances)
            
            # McIntosh diversity
            mcintosh = self._calculate_mcintosh(abundances)
            
            # Chao1 estimator (for richness estimation)
            chao1 = self._calculate_chao1(abundances)
            
            # ACE estimator
            ace = self._calculate_ace(abundances)
            
            # Coverage estimator
            coverage = self._calculate_coverage(abundances)
            
            alpha_results = {
                'sample_id': sample_id,
                'basic_metrics': {
                    'observed_species': int(num_species),
                    'total_individuals': int(total_individuals),
                    'coverage': round(coverage, 4)
                },
                'diversity_indices': {
                    'shannon': round(shannon, 4),
                    'shannon_exp': round(np.exp(shannon), 4),  # Effective number of species
                    'simpson': round(simpson, 4),
                    'simpson_reciprocal': round(1 / (1 - simpson), 4) if simpson < 1 else float('inf'),
                    'brillouin': round(brillouin, 4),
                    'mcintosh': round(mcintosh, 4)
                },
                'evenness_indices': {
                    'pielou_evenness': round(pielou_evenness, 4),
                    'berger_parker_dominance': round(berger_parker, 4),
                    'simpson_evenness': round(simpson / (1 - 1/num_species), 4) if num_species > 1 else 1.0
                },
                'richness_estimators': {
                    'chao1': round(chao1, 2) if chao1 is not None else None,
                    'ace': round(ace, 2) if ace is not None else None,
                    'fisher_alpha': round(fisher_alpha, 4) if fisher_alpha is not None else None
                },
                'abundance_distribution': {
                    'singletons': int(np.sum(abundances == 1)),
                    'doubletons': int(np.sum(abundances == 2)),
                    'most_abundant_count': int(np.max(abundances)),
                    'rare_species_count': int(np.sum(abundances <= 5))
                },
                'calculation_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🧬 Alpha diversity calculated: Shannon={shannon:.3f}, Simpson={simpson:.3f}")
            return alpha_results
            
        except Exception as e:
            logger.error(f"❌ Alpha diversity calculation failed: {e}")
            return {'error': str(e)}
    
    def calculate_beta_diversity(self, 
                                community_matrix: Dict[str, Dict[str, int]],
                                distance_metric: str = 'bray_curtis') -> Dict[str, Any]:
        """
        Calculate beta diversity between multiple samples
        
        Args:
            community_matrix: Dict of sample_id -> {species -> abundance}
            distance_metric: Distance metric to use
            
        Returns:
            Beta diversity results
        """
        try:
            if len(community_matrix) < 2:
                return {'error': 'At least 2 samples required for beta diversity'}
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(community_matrix).fillna(0).T
            sample_names = df.index.tolist()
            species_names = df.columns.tolist()
            abundance_matrix = df.values
            
            # Calculate distance matrix
            if distance_metric == 'jaccard':
                # Convert to presence/absence
                binary_matrix = (abundance_matrix > 0).astype(int)
                distances = pdist(binary_matrix, metric='jaccard')
            
            elif distance_metric == 'bray_curtis':
                distances = pdist(abundance_matrix, metric=self._bray_curtis)
            
            elif distance_metric == 'sorensen':
                binary_matrix = (abundance_matrix > 0).astype(int)
                distances = pdist(binary_matrix, metric=self._sorensen)
            
            elif distance_metric == 'horn_morisita':
                distances = pdist(abundance_matrix, metric=self._horn_morisita)
            
            else:
                return {'error': f'Unsupported distance metric: {distance_metric}'}
            
            # Convert to square matrix
            distance_matrix = squareform(distances)
            
            # Calculate additional beta diversity metrics
            whittaker_beta = self._calculate_whittaker_beta(abundance_matrix)
            
            # Cluster analysis
            linkage_matrix = linkage(distances, method='average')
            
            beta_results = {
                'distance_metric': distance_metric,
                'sample_names': sample_names,
                'distance_matrix': distance_matrix.tolist(),
                'distance_statistics': {
                    'mean_distance': round(np.mean(distances), 4),
                    'std_distance': round(np.std(distances), 4),
                    'min_distance': round(np.min(distances), 4),
                    'max_distance': round(np.max(distances), 4)
                },
                'whittaker_beta': round(whittaker_beta, 4),
                'cluster_analysis': {
                    'linkage_matrix': linkage_matrix.tolist(),
                    'cophenetic_correlation': self._calculate_cophenetic_correlation(distances, linkage_matrix)
                },
                'calculation_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🧬 Beta diversity calculated: {distance_metric}, {len(sample_names)} samples")
            return beta_results
            
        except Exception as e:
            logger.error(f"❌ Beta diversity calculation failed: {e}")
            return {'error': str(e)}
    
    def calculate_gamma_diversity(self, 
                                 community_matrix: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        """
        Calculate gamma diversity for the entire community
        
        Args:
            community_matrix: Dict of sample_id -> {species -> abundance}
            
        Returns:
            Gamma diversity results
        """
        try:
            if not community_matrix:
                return {'error': 'No community data provided'}
            
            # Aggregate all species across samples
            all_species = set()
            total_abundance = Counter()
            
            for sample_data in community_matrix.values():
                for species, count in sample_data.items():
                    all_species.add(species)
                    total_abundance[species] += count
            
            # Calculate gamma diversity (Shannon for entire community)
            gamma_alpha = self.calculate_alpha_diversity(dict(total_abundance), 'pooled_community')
            
            # Calculate mean alpha diversity
            sample_alphas = []
            for sample_id, sample_data in community_matrix.items():
                alpha_result = self.calculate_alpha_diversity(sample_data, sample_id)
                if 'diversity_indices' in alpha_result:
                    sample_alphas.append(alpha_result['diversity_indices']['shannon'])
            
            mean_alpha = np.mean(sample_alphas) if sample_alphas else 0
            
            # Beta diversity as gamma - alpha
            beta_diversity = gamma_alpha['diversity_indices']['shannon'] - mean_alpha
            
            # Multiplicative beta diversity
            multiplicative_beta = gamma_alpha['diversity_indices']['shannon_exp'] / np.exp(mean_alpha) if mean_alpha > 0 else 0
            
            gamma_results = {
                'gamma_diversity': gamma_alpha['diversity_indices'],
                'mean_alpha_diversity': round(mean_alpha, 4),
                'additive_beta_diversity': round(beta_diversity, 4),
                'multiplicative_beta_diversity': round(multiplicative_beta, 4),
                'total_species_pool': len(all_species),
                'total_individuals': sum(total_abundance.values()),
                'sample_count': len(community_matrix),
                'sample_alpha_diversities': [
                    {
                        'sample_id': sample_id,
                        'shannon': alpha_result.get('diversity_indices', {}).get('shannon', 0)
                    }
                    for sample_id, alpha_result in 
                    [(sid, self.calculate_alpha_diversity(sdata, sid)) 
                     for sid, sdata in community_matrix.items()]
                    if 'error' not in alpha_result
                ],
                'calculation_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🧬 Gamma diversity calculated: γ={gamma_alpha['diversity_indices']['shannon']:.3f}")
            return gamma_results
            
        except Exception as e:
            logger.error(f"❌ Gamma diversity calculation failed: {e}")
            return {'error': str(e)}
    
    def generate_rarefaction_curve(self, 
                                  abundance_data: Dict[str, int],
                                  max_samples: Optional[int] = None,
                                  step_size: int = 1) -> Dict[str, Any]:
        """
        Generate rarefaction curve for species accumulation
        
        Args:
            abundance_data: Dictionary of species -> abundance
            max_samples: Maximum number of individuals to sample
            step_size: Step size for rarefaction
            
        Returns:
            Rarefaction curve data
        """
        try:
            # Convert to individual occurrences
            individuals = []
            for species, count in abundance_data.items():
                individuals.extend([species] * int(count))
            
            total_individuals = len(individuals)
            
            if total_individuals == 0:
                return {'error': 'No individuals in sample'}
            
            # Set maximum samples if not provided
            if max_samples is None:
                max_samples = min(total_individuals, 1000)  # Reasonable limit
            
            # Generate rarefaction points
            sample_sizes = list(range(step_size, min(max_samples + 1, total_individuals + 1), step_size))
            
            rarefaction_points = []
            
            for n in sample_sizes:
                # Multiple random subsamples for each point
                species_counts = []
                iterations = min(100, max(10, 1000 // n))  # Adjust iterations based on sample size
                
                for _ in range(iterations):
                    # Random subsample without replacement
                    subsample = np.random.choice(individuals, size=n, replace=False)
                    unique_species = len(set(subsample))
                    species_counts.append(unique_species)
                
                mean_species = np.mean(species_counts)
                std_species = np.std(species_counts)
                
                rarefaction_points.append({
                    'sample_size': n,
                    'mean_species': round(mean_species, 2),
                    'std_species': round(std_species, 2),
                    'min_species': int(np.min(species_counts)),
                    'max_species': int(np.max(species_counts))
                })
            
            # Calculate expected richness at different sampling intensities
            expected_richness = self._calculate_expected_richness(abundance_data, sample_sizes)
            
            rarefaction_results = {
                'total_individuals': total_individuals,
                'observed_species': len(abundance_data),
                'rarefaction_curve': rarefaction_points,
                'expected_richness': expected_richness,
                'sampling_parameters': {
                    'max_samples': max_samples,
                    'step_size': step_size,
                    'iterations_per_point': iterations
                },
                'calculation_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🧬 Rarefaction curve generated: {len(sample_sizes)} points")
            return rarefaction_results
            
        except Exception as e:
            logger.error(f"❌ Rarefaction curve generation failed: {e}")
            return {'error': str(e)}
    
    def community_comparison(self, 
                           sample1: Dict[str, int],
                           sample2: Dict[str, int],
                           sample1_name: str = "Sample 1",
                           sample2_name: str = "Sample 2") -> Dict[str, Any]:
        """
        Compare two communities comprehensively
        
        Args:
            sample1: First community abundance data
            sample2: Second community abundance data
            sample1_name: Name of first sample
            sample2_name: Name of second sample
            
        Returns:
            Comprehensive comparison results
        """
        try:
            # Calculate alpha diversity for both samples
            alpha1 = self.calculate_alpha_diversity(sample1, sample1_name)
            alpha2 = self.calculate_alpha_diversity(sample2, sample2_name)
            
            if 'error' in alpha1 or 'error' in alpha2:
                return {'error': 'Failed to calculate alpha diversity for one or both samples'}
            
            # Calculate beta diversity
            community_matrix = {sample1_name: sample1, sample2_name: sample2}
            beta_results = self.calculate_beta_diversity(community_matrix, 'bray_curtis')
            
            # Species overlap analysis
            species1 = set(sample1.keys())
            species2 = set(sample2.keys())
            
            shared_species = species1.intersection(species2)
            unique_to_1 = species1 - species2
            unique_to_2 = species2 - species1
            
            # Jaccard similarity
            jaccard_similarity = len(shared_species) / len(species1.union(species2))
            
            # Abundance-weighted overlap
            total_abundance_1 = sum(sample1.values())
            total_abundance_2 = sum(sample2.values())
            
            shared_abundance_1 = sum(sample1.get(sp, 0) for sp in shared_species)
            shared_abundance_2 = sum(sample2.get(sp, 0) for sp in shared_species)
            
            abundance_overlap_1 = shared_abundance_1 / total_abundance_1 if total_abundance_1 > 0 else 0
            abundance_overlap_2 = shared_abundance_2 / total_abundance_2 if total_abundance_2 > 0 else 0
            
            comparison_results = {
                'sample_names': [sample1_name, sample2_name],
                'alpha_diversity_comparison': {
                    sample1_name: alpha1['diversity_indices'],
                    sample2_name: alpha2['diversity_indices'],
                    'shannon_difference': alpha2['diversity_indices']['shannon'] - alpha1['diversity_indices']['shannon'],
                    'simpson_difference': alpha2['diversity_indices']['simpson'] - alpha1['diversity_indices']['simpson']
                },
                'beta_diversity': {
                    'bray_curtis_distance': beta_results['distance_matrix'][0][1] if 'error' not in beta_results else None,
                    'whittaker_beta': beta_results.get('whittaker_beta')
                },
                'species_overlap': {
                    'shared_species_count': len(shared_species),
                    'unique_to_sample1': len(unique_to_1),
                    'unique_to_sample2': len(unique_to_2),
                    'jaccard_similarity': round(jaccard_similarity, 4),
                    'shared_species_list': list(shared_species)[:20]  # Limit to first 20
                },
                'abundance_overlap': {
                    'sample1_shared_abundance_fraction': round(abundance_overlap_1, 4),
                    'sample2_shared_abundance_fraction': round(abundance_overlap_2, 4),
                    'total_individuals': {
                        sample1_name: total_abundance_1,
                        sample2_name: total_abundance_2
                    }
                },
                'comparison_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"🧬 Community comparison completed: Jaccard={jaccard_similarity:.3f}")
            return comparison_results
            
        except Exception as e:
            logger.error(f"❌ Community comparison failed: {e}")
            return {'error': str(e)}
    
    def _calculate_fisher_alpha(self, abundances: np.ndarray) -> Optional[float]:
        """Calculate Fisher's alpha using iterative method"""
        try:
            n = np.sum(abundances)  # Total individuals
            s = len(abundances)     # Number of species
            
            # Initial estimate
            alpha = s / np.log(1 + n/s)
            
            # Iterative refinement
            for _ in range(100):
                old_alpha = alpha
                f_alpha = alpha * np.log(1 + n/alpha) - s
                df_alpha = np.log(1 + n/alpha) - n/(alpha + n)
                
                if abs(df_alpha) < 1e-10:
                    break
                    
                alpha = alpha - f_alpha / df_alpha
                
                if abs(alpha - old_alpha) < 1e-8:
                    break
            
            return alpha if alpha > 0 else None
            
        except:
            return None
    
    def _calculate_brillouin(self, abundances: np.ndarray) -> float:
        """Calculate Brillouin diversity index"""
        n = np.sum(abundances)
        
        if n <= 1:
            return 0.0
        
        log_factorial_sum = np.sum([math.lgamma(ni + 1) for ni in abundances])
        brillouin = (math.lgamma(n + 1) - log_factorial_sum) / n
        
        return brillouin
    
    def _calculate_mcintosh(self, abundances: np.ndarray) -> float:
        """Calculate McIntosh diversity index"""
        n = np.sum(abundances)
        u = np.sqrt(np.sum(abundances ** 2))
        
        if n == u:  # All abundance in one species
            return 0.0
        
        mcintosh = (n - u) / (n - np.sqrt(n))
        return mcintosh
    
    def _calculate_chao1(self, abundances: np.ndarray) -> Optional[float]:
        """Calculate Chao1 richness estimator"""
        try:
            s_obs = len(abundances)
            f1 = np.sum(abundances == 1)  # Singletons
            f2 = np.sum(abundances == 2)  # Doubletons
            
            if f2 > 0:
                chao1 = s_obs + (f1 ** 2) / (2 * f2)
            elif f1 > 0:
                chao1 = s_obs + f1 * (f1 - 1) / 2
            else:
                chao1 = s_obs
            
            return chao1
            
        except:
            return None
    
    def _calculate_ace(self, abundances: np.ndarray, threshold: int = 10) -> Optional[float]:
        """Calculate ACE richness estimator"""
        try:
            s_obs = len(abundances)
            
            # Abundant species (> threshold)
            s_abund = np.sum(abundances > threshold)
            
            # Rare species (<= threshold)
            rare_abundances = abundances[abundances <= threshold]
            s_rare = len(rare_abundances)
            
            if s_rare == 0:
                return float(s_obs)
            
            # Calculate coefficient of variation for rare species
            n_rare = np.sum(rare_abundances)
            c_ace = 1 - np.sum(rare_abundances == 1) / n_rare if n_rare > 0 else 0
            
            # Gamma calculation for ACE
            gamma_ace = max(0, (s_rare / c_ace) * np.sum(rare_abundances * (rare_abundances - 1)) / (n_rare * (n_rare - 1)) - 1) if n_rare > 1 and c_ace > 0 else 0
            
            ace = s_abund + (s_rare / c_ace) + (np.sum(rare_abundances == 1) / c_ace) * gamma_ace if c_ace > 0 else s_obs
            
            return ace
            
        except:
            return None
    
    def _calculate_coverage(self, abundances: np.ndarray) -> float:
        """Calculate Good's coverage estimator"""
        n = np.sum(abundances)
        f1 = np.sum(abundances == 1)
        
        coverage = 1 - (f1 / n) if n > 0 else 0
        return coverage
    
    def _bray_curtis(self, u: np.ndarray, v: np.ndarray) -> float:
        """Calculate Bray-Curtis dissimilarity"""
        return np.sum(np.abs(u - v)) / np.sum(u + v) if np.sum(u + v) > 0 else 0
    
    def _sorensen(self, u: np.ndarray, v: np.ndarray) -> float:
        """Calculate Sørensen dissimilarity"""
        intersection = np.sum(np.minimum(u, v))
        total = np.sum(u) + np.sum(v)
        return 1 - (2 * intersection / total) if total > 0 else 0
    
    def _horn_morisita(self, u: np.ndarray, v: np.ndarray) -> float:
        """Calculate Horn-Morisita dissimilarity"""
        sum_u = np.sum(u)
        sum_v = np.sum(v)
        
        if sum_u == 0 or sum_v == 0:
            return 1.0
        
        lambda_u = np.sum(u * (u - 1)) / (sum_u * (sum_u - 1)) if sum_u > 1 else 0
        lambda_v = np.sum(v * (v - 1)) / (sum_v * (sum_v - 1)) if sum_v > 1 else 0
        
        numerator = 2 * np.sum((u / sum_u) * (v / sum_v))
        denominator = lambda_u + lambda_v
        
        return 1 - (numerator / denominator) if denominator > 0 else 1.0
    
    def _calculate_whittaker_beta(self, abundance_matrix: np.ndarray) -> float:
        """Calculate Whittaker's beta diversity"""
        # Total species across all samples
        total_species = np.sum(np.sum(abundance_matrix, axis=0) > 0)
        
        # Mean species per sample
        mean_alpha = np.mean([np.sum(sample > 0) for sample in abundance_matrix])
        
        return (total_species / mean_alpha) - 1 if mean_alpha > 0 else 0
    
    def _calculate_cophenetic_correlation(self, distances: np.ndarray, linkage_matrix: np.ndarray) -> float:
        """Calculate cophenetic correlation coefficient"""
        try:
            from scipy.cluster.hierarchy import cophenet
            cophenetic_distances, _ = cophenet(linkage_matrix, distances)
            correlation = np.corrcoef(distances, cophenetic_distances)[0, 1]
            return round(correlation, 4)
        except:
            return 0.0
    
    def _calculate_expected_richness(self, abundance_data: Dict[str, int], sample_sizes: List[int]) -> List[Dict[str, Any]]:
        """Calculate expected richness using analytical formula"""
        expected_points = []
        
        total_individuals = sum(abundance_data.values())
        
        for n in sample_sizes:
            if n >= total_individuals:
                expected_richness = len(abundance_data)
            else:
                # Use hypergeometric expectation
                expected_richness = 0
                for species, count in abundance_data.items():
                    # Probability that species is NOT represented in sample of size n
                    prob_absent = self._hypergeometric_prob_absent(count, total_individuals - count, n)
                    expected_richness += 1 - prob_absent
            
            expected_points.append({
                'sample_size': n,
                'expected_richness': round(expected_richness, 2)
            })
        
        return expected_points
    
    def _hypergeometric_prob_absent(self, k: int, n_minus_k: int, n: int) -> float:
        """Calculate probability of species being absent in hypergeometric sampling"""
        try:
            # Probability = C(N-K, n) / C(N, n)
            # Where K is species abundance, N is total individuals
            from math import comb
            total = k + n_minus_k
            
            if n > n_minus_k:
                return 0.0  # Impossible to not include species
            
            prob = comb(n_minus_k, n) / comb(total, n)
            return min(1.0, max(0.0, prob))
            
        except:
            return 0.0

# Global diversity calculator instance
diversity_calculator = DiversityCalculator()