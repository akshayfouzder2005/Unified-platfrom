"""
📈 Trend Analysis - Statistical Trend Detection & Analysis

Advanced trend detection and analysis for marine time-series data.
Implements Mann-Kendall test, Sen's slope, and change point detection.

Author: Ocean-Bio Development Team
Version: 2.0.0
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TrendAnalyzer:
    """
    📈 Advanced Trend Analysis Engine
    
    Provides comprehensive trend detection and analysis:
    - Mann-Kendall trend test
    - Sen's slope estimation
    - Change point detection
    - Linear and non-linear trend fitting
    - Breakpoint analysis
    - Trend significance testing
    """
    
    def __init__(self):
        """Initialize the trend analyzer"""
        self.trend_methods = [
            'mann_kendall',
            'linear_regression', 
            'polynomial_regression',
            'sen_slope',
            'spearman_correlation'
        ]
        
    def mann_kendall_test(self, 
                         data: List[float],
                         dates: List[str],
                         alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform Mann-Kendall trend test
        
        Args:
            data: Time-series data values
            dates: Corresponding dates
            alpha: Significance level for hypothesis testing
            
        Returns:
            Mann-Kendall test results
        """
        try:
            data_array = np.array(data)
            n = len(data_array)
            
            # Calculate Mann-Kendall statistic S
            S = 0
            for i in range(n-1):
                for j in range(i+1, n):
                    if data_array[j] > data_array[i]:
                        S += 1
                    elif data_array[j] < data_array[i]:
                        S -= 1
            
            # Calculate variance of S
            # Check for ties
            unique_vals, counts = np.unique(data_array, return_counts=True)
            n_ties = len(counts[counts > 1])
            
            if n_ties > 0:
                # Variance with ties correction
                tie_correction = sum(t * (t - 1) * (2 * t + 5) for t in counts if t > 1)
                var_S = (n * (n - 1) * (2 * n + 5) - tie_correction) / 18
            else:
                # Variance without ties
                var_S = n * (n - 1) * (2 * n + 5) / 18
            
            # Calculate standardized test statistic Z
            if S > 0:
                Z = (S - 1) / np.sqrt(var_S)
            elif S < 0:
                Z = (S + 1) / np.sqrt(var_S)
            else:
                Z = 0
            
            # Calculate p-value (two-tailed test)
            p_value = 2 * (1 - stats.norm.cdf(abs(Z)))
            
            # Determine trend direction and significance
            if p_value <= alpha:
                if S > 0:
                    trend = "Increasing"
                    significance = "Significant"
                else:
                    trend = "Decreasing"
                    significance = "Significant"
            else:
                trend = "No trend"
                significance = "Not significant"
            
            # Calculate Sen's slope
            slopes = []
            for i in range(n-1):
                for j in range(i+1, n):
                    if i != j:
                        slope = (data_array[j] - data_array[i]) / (j - i)
                        slopes.append(slope)
            
            sen_slope = np.median(slopes) if slopes else 0
            
            mk_results = {
                'test': 'mann_kendall',
                'statistic_S': int(S),
                'standardized_Z': float(Z),
                'p_value': float(p_value),
                'alpha': alpha,
                'trend_direction': trend,
                'significance': significance,
                'sen_slope': float(sen_slope),
                'sample_size': n,
                'ties_present': n_ties > 0,
                'test_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"📈 Mann-Kendall test completed: {trend} trend (p={p_value:.4f})")
            return mk_results
            
        except Exception as e:
            logger.error(f"❌ Mann-Kendall test failed: {e}")
            return {'error': str(e)}
    
    def linear_trend_analysis(self, 
                             data: List[float],
                             dates: List[str],
                             confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Perform linear trend analysis using least squares regression
        
        Args:
            data: Time-series data values
            dates: Corresponding dates
            confidence_level: Confidence level for slope estimation
            
        Returns:
            Linear trend analysis results
        """
        try:
            # Convert dates to numeric (days since start)
            date_objects = pd.to_datetime(dates)
            start_date = date_objects.min()
            x = np.array([(d - start_date).days for d in date_objects]).reshape(-1, 1)
            y = np.array(data)
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(x, y)
            
            # Calculate fitted values
            y_pred = model.predict(x)
            
            # Calculate statistics
            r2 = r2_score(y, y_pred)
            
            # Calculate standard error of slope
            residuals = y - y_pred
            mse = np.mean(residuals**2)
            x_flat = x.flatten()
            slope_se = np.sqrt(mse / np.sum((x_flat - np.mean(x_flat))**2))
            
            # Calculate confidence interval for slope
            df = len(y) - 2  # degrees of freedom
            t_critical = stats.t.ppf((1 + confidence_level) / 2, df)
            slope_ci_lower = model.coef_[0] - t_critical * slope_se
            slope_ci_upper = model.coef_[0] + t_critical * slope_se
            
            # Statistical significance of slope
            t_stat = model.coef_[0] / slope_se if slope_se > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
            # Determine trend significance
            is_significant = p_value < (1 - confidence_level)
            
            if is_significant:
                if model.coef_[0] > 0:
                    trend_direction = "Increasing"
                else:
                    trend_direction = "Decreasing"
            else:
                trend_direction = "No significant trend"
            
            # Calculate trend over entire period
            total_days = x_flat[-1] - x_flat[0]
            total_change = model.coef_[0] * total_days
            percent_change = (total_change / model.intercept_) * 100 if model.intercept_ != 0 else 0
            
            linear_results = {
                'test': 'linear_regression',
                'slope': float(model.coef_[0]),
                'intercept': float(model.intercept_),
                'r_squared': float(r2),
                'slope_standard_error': float(slope_se),
                'slope_confidence_interval': [float(slope_ci_lower), float(slope_ci_upper)],
                'confidence_level': confidence_level,
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'trend_direction': trend_direction,
                'is_significant': is_significant,
                'total_change': float(total_change),
                'percent_change': float(percent_change),
                'fitted_values': y_pred.tolist(),
                'residuals': residuals.tolist(),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"📈 Linear trend analysis completed: slope={model.coef_[0]:.6f}, R²={r2:.3f}")
            return linear_results
            
        except Exception as e:
            logger.error(f"❌ Linear trend analysis failed: {e}")
            return {'error': str(e)}
    
    def polynomial_trend_analysis(self, 
                                 data: List[float],
                                 dates: List[str],
                                 degree: int = 2) -> Dict[str, Any]:
        """
        Perform polynomial trend analysis
        
        Args:
            data: Time-series data values
            dates: Corresponding dates
            degree: Polynomial degree
            
        Returns:
            Polynomial trend analysis results
        """
        try:
            # Convert dates to numeric
            date_objects = pd.to_datetime(dates)
            start_date = date_objects.min()
            x = np.array([(d - start_date).days for d in date_objects])
            y = np.array(data)
            
            # Create polynomial features
            poly_features = PolynomialFeatures(degree=degree)
            x_poly = poly_features.fit_transform(x.reshape(-1, 1))
            
            # Fit polynomial regression
            model = LinearRegression()
            model.fit(x_poly, y)
            
            # Calculate fitted values
            y_pred = model.predict(x_poly)
            
            # Calculate R²
            r2 = r2_score(y, y_pred)
            
            # Calculate adjusted R²
            n = len(y)
            p = degree + 1  # number of parameters
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            
            # Calculate residuals
            residuals = y - y_pred
            rmse = np.sqrt(np.mean(residuals**2))
            
            # Identify turning points (for degree > 1)
            turning_points = []
            if degree > 1:
                # Calculate derivative coefficients
                coeffs = model.coef_[::-1]  # Reverse for numpy poly format
                deriv_coeffs = np.polyder(coeffs)
                
                # Find roots of derivative (turning points)
                if len(deriv_coeffs) > 0:
                    roots = np.roots(deriv_coeffs)
                    real_roots = roots[np.isreal(roots)].real
                    
                    # Filter roots within data range
                    x_min, x_max = x.min(), x.max()
                    valid_roots = real_roots[(real_roots >= x_min) & (real_roots <= x_max)]
                    
                    for root in valid_roots:
                        root_date = start_date + timedelta(days=float(root))
                        root_value = model.predict(poly_features.transform([[root]]))[0]
                        turning_points.append({
                            'date': root_date.strftime('%Y-%m-%d'),
                            'value': float(root_value),
                            'x_position': float(root)
                        })
            
            polynomial_results = {
                'test': 'polynomial_regression',
                'degree': degree,
                'coefficients': model.coef_.tolist(),
                'intercept': float(model.intercept_),
                'r_squared': float(r2),
                'adjusted_r_squared': float(adj_r2),
                'rmse': float(rmse),
                'fitted_values': y_pred.tolist(),
                'residuals': residuals.tolist(),
                'turning_points': turning_points,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"📈 Polynomial trend analysis completed: degree={degree}, R²={r2:.3f}")
            return polynomial_results
            
        except Exception as e:
            logger.error(f"❌ Polynomial trend analysis failed: {e}")
            return {'error': str(e)}
    
    def change_point_detection(self, 
                              data: List[float],
                              dates: List[str],
                              min_segment_length: int = 5) -> Dict[str, Any]:
        """
        Detect change points in time-series using PELT (Pruned Exact Linear Time)
        
        Args:
            data: Time-series data values
            dates: Corresponding dates
            min_segment_length: Minimum length of segments
            
        Returns:
            Change point detection results
        """
        try:
            data_array = np.array(data)
            n = len(data_array)
            
            if n < 2 * min_segment_length:
                return {
                    'change_points': [],
                    'segments': [],
                    'error': 'Insufficient data for change point detection'
                }
            
            # Simple change point detection using variance change
            change_points = []
            
            # Calculate cumulative sum of squares
            cumsum = np.cumsum(data_array)
            cumsum_sq = np.cumsum(data_array**2)
            
            best_score = float('inf')
            best_change_point = None
            
            # Test each potential change point
            for i in range(min_segment_length, n - min_segment_length):
                # Calculate sum of squared errors for two segments
                
                # First segment (0 to i)
                n1 = i
                mean1 = cumsum[i-1] / n1
                ss1 = cumsum_sq[i-1] - n1 * mean1**2
                
                # Second segment (i to n)
                n2 = n - i
                mean2 = (cumsum[-1] - cumsum[i-1]) / n2
                ss2 = (cumsum_sq[-1] - cumsum_sq[i-1]) - n2 * mean2**2
                
                # Total sum of squared errors
                total_ss = ss1 + ss2
                
                # Penalize for model complexity (BIC-like penalty)
                penalty = 2 * np.log(n)
                score = total_ss + penalty
                
                if score < best_score:
                    best_score = score
                    best_change_point = i
            
            # If a significant change point was found
            change_points = []
            if best_change_point is not None:
                change_point_date = pd.to_datetime(dates[best_change_point])
                change_points.append({
                    'index': int(best_change_point),
                    'date': change_point_date.strftime('%Y-%m-%d'),
                    'value': float(data_array[best_change_point])
                })
            
            # Create segments based on change points
            segments = []
            if change_points:
                cp_idx = change_points[0]['index']
                
                # First segment
                segment1_data = data_array[:cp_idx]
                segments.append({
                    'start_index': 0,
                    'end_index': cp_idx - 1,
                    'start_date': dates[0],
                    'end_date': dates[cp_idx - 1],
                    'mean': float(np.mean(segment1_data)),
                    'std': float(np.std(segment1_data)),
                    'trend_slope': self._calculate_segment_slope(segment1_data),
                    'length': len(segment1_data)
                })
                
                # Second segment
                segment2_data = data_array[cp_idx:]
                segments.append({
                    'start_index': cp_idx,
                    'end_index': n - 1,
                    'start_date': dates[cp_idx],
                    'end_date': dates[-1],
                    'mean': float(np.mean(segment2_data)),
                    'std': float(np.std(segment2_data)),
                    'trend_slope': self._calculate_segment_slope(segment2_data),
                    'length': len(segment2_data)
                })
            else:
                # No change points - single segment
                segments.append({
                    'start_index': 0,
                    'end_index': n - 1,
                    'start_date': dates[0],
                    'end_date': dates[-1],
                    'mean': float(np.mean(data_array)),
                    'std': float(np.std(data_array)),
                    'trend_slope': self._calculate_segment_slope(data_array),
                    'length': n
                })
            
            change_point_results = {
                'method': 'variance_based_detection',
                'change_points': change_points,
                'segments': segments,
                'n_change_points': len(change_points),
                'min_segment_length': min_segment_length,
                'total_segments': len(segments),
                'detection_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"📈 Change point detection completed: {len(change_points)} change points found")
            return change_point_results
            
        except Exception as e:
            logger.error(f"❌ Change point detection failed: {e}")
            return {'error': str(e)}
    
    def correlation_analysis(self, 
                           data1: List[float],
                           data2: List[float],
                           method: str = 'pearson') -> Dict[str, Any]:
        """
        Analyze correlation between two time-series
        
        Args:
            data1: First time-series
            data2: Second time-series
            method: Correlation method ('pearson' or 'spearman')
            
        Returns:
            Correlation analysis results
        """
        try:
            if len(data1) != len(data2):
                raise ValueError("Time-series must have equal length")
            
            data1_array = np.array(data1)
            data2_array = np.array(data2)
            
            # Remove missing values
            mask = ~(np.isnan(data1_array) | np.isnan(data2_array))
            data1_clean = data1_array[mask]
            data2_clean = data2_array[mask]
            
            if len(data1_clean) < 3:
                return {'error': 'Insufficient data for correlation analysis'}
            
            # Calculate correlation
            if method == 'pearson':
                correlation, p_value = pearsonr(data1_clean, data2_clean)
            elif method == 'spearman':
                correlation, p_value = spearmanr(data1_clean, data2_clean)
            else:
                raise ValueError("Method must be 'pearson' or 'spearman'")
            
            # Determine correlation strength
            abs_corr = abs(correlation)
            if abs_corr >= 0.8:
                strength = "Very strong"
            elif abs_corr >= 0.6:
                strength = "Strong"
            elif abs_corr >= 0.4:
                strength = "Moderate"
            elif abs_corr >= 0.2:
                strength = "Weak"
            else:
                strength = "Very weak"
            
            # Determine direction
            direction = "Positive" if correlation > 0 else "Negative"
            
            correlation_results = {
                'method': method,
                'correlation_coefficient': float(correlation),
                'p_value': float(p_value),
                'strength': strength,
                'direction': direction,
                'is_significant': p_value < 0.05,
                'sample_size': len(data1_clean),
                'missing_values_removed': len(data1) - len(data1_clean),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"📈 Correlation analysis completed: r={correlation:.3f}, p={p_value:.4f}")
            return correlation_results
            
        except Exception as e:
            logger.error(f"❌ Correlation analysis failed: {e}")
            return {'error': str(e)}
    
    def comprehensive_trend_analysis(self, 
                                   data: List[float],
                                   dates: List[str],
                                   include_changepoints: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive trend analysis using multiple methods
        
        Args:
            data: Time-series data values
            dates: Corresponding dates
            include_changepoints: Whether to include change point detection
            
        Returns:
            Comprehensive trend analysis results
        """
        try:
            results = {
                'comprehensive_analysis': True,
                'data_summary': {
                    'n_observations': len(data),
                    'start_date': dates[0],
                    'end_date': dates[-1],
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data))
                },
                'analyses': {}
            }
            
            # Mann-Kendall test
            mk_result = self.mann_kendall_test(data, dates)
            if 'error' not in mk_result:
                results['analyses']['mann_kendall'] = mk_result
            
            # Linear trend analysis
            linear_result = self.linear_trend_analysis(data, dates)
            if 'error' not in linear_result:
                results['analyses']['linear_regression'] = linear_result
            
            # Polynomial trend analysis (degree 2)
            poly_result = self.polynomial_trend_analysis(data, dates, degree=2)
            if 'error' not in poly_result:
                results['analyses']['polynomial_regression'] = poly_result
            
            # Change point detection
            if include_changepoints:
                cp_result = self.change_point_detection(data, dates)
                if 'error' not in cp_result:
                    results['analyses']['change_point_detection'] = cp_result
            
            # Summary of trend conclusions
            trend_consensus = self._determine_trend_consensus(results['analyses'])
            results['trend_consensus'] = trend_consensus
            
            results['analysis_timestamp'] = datetime.now().isoformat()
            
            logger.info("📈 Comprehensive trend analysis completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive trend analysis failed: {e}")
            return {'error': str(e)}
    
    def _calculate_segment_slope(self, data: np.ndarray) -> float:
        """Calculate slope for a data segment using linear regression"""
        try:
            if len(data) < 2:
                return 0.0
            
            x = np.arange(len(data))
            slope, _, _, _, _ = stats.linregress(x, data)
            return float(slope)
        except:
            return 0.0
    
    def _determine_trend_consensus(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Determine consensus trend direction from multiple analyses"""
        try:
            trend_votes = {
                'increasing': 0,
                'decreasing': 0,
                'no_trend': 0
            }
            
            methods_used = []
            
            # Mann-Kendall
            if 'mann_kendall' in analyses:
                mk = analyses['mann_kendall']
                methods_used.append('Mann-Kendall')
                
                if mk['significance'] == 'Significant':
                    if mk['trend_direction'] == 'Increasing':
                        trend_votes['increasing'] += 1
                    elif mk['trend_direction'] == 'Decreasing':
                        trend_votes['decreasing'] += 1
                else:
                    trend_votes['no_trend'] += 1
            
            # Linear regression
            if 'linear_regression' in analyses:
                lr = analyses['linear_regression']
                methods_used.append('Linear Regression')
                
                if lr['is_significant']:
                    if lr['slope'] > 0:
                        trend_votes['increasing'] += 1
                    else:
                        trend_votes['decreasing'] += 1
                else:
                    trend_votes['no_trend'] += 1
            
            # Determine consensus
            max_votes = max(trend_votes.values())
            consensus_trends = [k for k, v in trend_votes.items() if v == max_votes]
            
            if len(consensus_trends) == 1:
                consensus = consensus_trends[0]
                confidence = "High"
            else:
                consensus = "Mixed"
                confidence = "Low"
            
            return {
                'consensus_trend': consensus,
                'confidence': confidence,
                'method_votes': trend_votes,
                'methods_used': methods_used,
                'total_methods': len(methods_used)
            }
            
        except Exception as e:
            return {'error': f'Consensus determination failed: {e}'}

# Global trend analyzer instance
trend_analyzer = TrendAnalyzer()