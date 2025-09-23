"""
📈 Population Models - Population Dynamics & Demographic Analysis

Advanced population modeling for marine species.
Implements age-structured models, recruitment analysis, and demographic projections.

Author: Ocean-Bio Development Team
Version: 2.0.0
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PopulationModeler:
    """
    📈 Advanced Population Dynamics Modeler
    
    Provides comprehensive population modeling capabilities:
    - Age-structured population models
    - Leslie matrix projections
    - Recruitment analysis
    - Mortality estimation
    - Population growth rate analysis
    - Demographic transition projections
    """
    
    def __init__(self):
        """Initialize the population modeler"""
        self.models = ['leslie_matrix', 'exponential', 'logistic', 'age_structured']
        self.demographic_rates = {}
        
    def leslie_matrix_projection(self, 
                                life_table: Dict[str, List[float]],
                                initial_abundance: List[float],
                                projection_years: int = 20) -> Dict[str, Any]:
        """
        Perform Leslie matrix population projection
        
        Args:
            life_table: Dictionary with 'survival_rates' and 'fertility_rates'
            initial_abundance: Initial abundance by age class
            projection_years: Number of years to project
            
        Returns:
            Leslie matrix projection results
        """
        try:
            survival_rates = np.array(life_table['survival_rates'])
            fertility_rates = np.array(life_table['fertility_rates'])
            initial_pop = np.array(initial_abundance)
            
            n_ages = len(survival_rates)
            
            # Validate input dimensions
            if len(fertility_rates) != n_ages or len(initial_pop) != n_ages:
                raise ValueError("All arrays must have the same length")
            
            # Construct Leslie matrix
            leslie_matrix = np.zeros((n_ages, n_ages))
            
            # Fill first row with fertility rates
            leslie_matrix[0, :] = fertility_rates
            
            # Fill subdiagonal with survival rates (except last age class)
            for i in range(1, n_ages):
                leslie_matrix[i, i-1] = survival_rates[i-1]
            
            # Project population forward
            population_trajectory = np.zeros((projection_years + 1, n_ages))
            population_trajectory[0, :] = initial_pop
            
            for year in range(1, projection_years + 1):
                population_trajectory[year, :] = leslie_matrix @ population_trajectory[year - 1, :]
            
            # Calculate total population by year
            total_population = np.sum(population_trajectory, axis=1)
            
            # Calculate population growth rates
            growth_rates = np.diff(total_population) / total_population[:-1]
            
            # Calculate stable age distribution (dominant eigenvector)
            eigenvalues, eigenvectors = np.linalg.eig(leslie_matrix)
            dominant_eigenvalue = np.max(np.real(eigenvalues))
            dominant_eigenvector = np.real(eigenvectors[:, np.argmax(np.real(eigenvalues))])
            stable_age_dist = dominant_eigenvector / np.sum(dominant_eigenvector)
            
            # Calculate reproductive value (left eigenvector)
            _, left_eigenvectors = np.linalg.eig(leslie_matrix.T)
            reproductive_values = np.real(left_eigenvectors[:, np.argmax(np.real(eigenvalues))])
            reproductive_values = reproductive_values / reproductive_values[0]  # Normalize to age 0 = 1
            
            # Generation time calculation
            mean_age_reproduction = np.sum(np.arange(n_ages) * fertility_rates * stable_age_dist) / np.sum(fertility_rates * stable_age_dist)
            
            leslie_results = {
                'model': 'leslie_matrix',
                'leslie_matrix': leslie_matrix.tolist(),
                'projection_years': projection_years,
                'population_trajectory': {
                    'years': list(range(projection_years + 1)),
                    'age_structure': population_trajectory.tolist(),
                    'total_population': total_population.tolist()
                },
                'growth_analysis': {
                    'annual_growth_rates': growth_rates.tolist(),
                    'mean_growth_rate': float(np.mean(growth_rates)),
                    'asymptotic_growth_rate': float(dominant_eigenvalue)
                },
                'demographic_parameters': {
                    'stable_age_distribution': stable_age_dist.tolist(),
                    'reproductive_values': reproductive_values.tolist(),
                    'generation_time': float(mean_age_reproduction),
                    'net_reproductive_rate': float(dominant_eigenvalue)
                },
                'life_table_input': life_table,
                'projection_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"📈 Leslie matrix projection completed: λ={dominant_eigenvalue:.3f}")
            return leslie_results
            
        except Exception as e:
            logger.error(f"❌ Leslie matrix projection failed: {e}")
            return {'error': str(e)}
    
    def exponential_growth_model(self, 
                                abundance_data: List[float],
                                time_points: List[float],
                                forecast_periods: int = 10) -> Dict[str, Any]:
        """
        Fit exponential growth model: N(t) = N0 * exp(r*t)
        
        Args:
            abundance_data: Observed abundance values
            time_points: Corresponding time points
            forecast_periods: Number of periods to forecast
            
        Returns:
            Exponential growth model results
        """
        try:
            N_obs = np.array(abundance_data)
            t_obs = np.array(time_points)
            
            # Log-transform for linear regression
            ln_N = np.log(N_obs)
            
            # Fit linear model: ln(N) = ln(N0) + r*t
            coeffs = np.polyfit(t_obs, ln_N, 1)
            r = coeffs[0]  # Growth rate
            ln_N0 = coeffs[1]  # Log initial abundance
            N0 = np.exp(ln_N0)
            
            # Calculate fitted values
            N_fitted = N0 * np.exp(r * t_obs)
            
            # Calculate R²
            ss_res = np.sum((N_obs - N_fitted) ** 2)
            ss_tot = np.sum((N_obs - np.mean(N_obs)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Generate forecast
            if forecast_periods > 0:
                t_forecast = np.arange(t_obs[-1] + 1, t_obs[-1] + forecast_periods + 1)
                N_forecast = N0 * np.exp(r * t_forecast)
            else:
                t_forecast = np.array([])
                N_forecast = np.array([])
            
            # Calculate doubling time (if r > 0) or half-life (if r < 0)
            if r > 0:
                doubling_time = np.log(2) / r
                time_metric = 'doubling_time'
            elif r < 0:
                doubling_time = np.log(0.5) / r  # Actually half-life
                time_metric = 'half_life'
            else:
                doubling_time = float('inf')
                time_metric = 'no_change'
            
            exp_results = {
                'model': 'exponential_growth',
                'parameters': {
                    'initial_abundance_N0': float(N0),
                    'growth_rate_r': float(r),
                    'intrinsic_growth_rate': float(r)
                },
                'model_fit': {
                    'r_squared': float(r_squared),
                    'fitted_values': N_fitted.tolist()
                },
                'biological_interpretation': {
                    time_metric: float(doubling_time),
                    'population_trend': 'increasing' if r > 0 else 'decreasing' if r < 0 else 'stable'
                },
                'forecast': {
                    'time_points': t_forecast.tolist(),
                    'predicted_abundance': N_forecast.tolist(),
                    'forecast_periods': forecast_periods
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"📈 Exponential growth model fitted: r={r:.4f}, R²={r_squared:.3f}")
            return exp_results
            
        except Exception as e:
            logger.error(f"❌ Exponential growth modeling failed: {e}")
            return {'error': str(e)}
    
    def logistic_growth_model(self, 
                             abundance_data: List[float],
                             time_points: List[float],
                             forecast_periods: int = 10) -> Dict[str, Any]:
        """
        Fit logistic growth model: dN/dt = r*N*(1 - N/K)
        
        Args:
            abundance_data: Observed abundance values
            time_points: Corresponding time points
            forecast_periods: Number of periods to forecast
            
        Returns:
            Logistic growth model results
        """
        try:
            N_obs = np.array(abundance_data)
            t_obs = np.array(time_points)
            
            # Define logistic growth function
            def logistic(t, N0, r, K):
                return K / (1 + ((K - N0) / N0) * np.exp(-r * t))
            
            # Define objective function for optimization
            def objective(params):
                N0, r, K = params
                if N0 <= 0 or r <= 0 or K <= 0:
                    return 1e10
                try:
                    N_pred = logistic(t_obs, N0, r, K)
                    return np.sum((N_obs - N_pred) ** 2)
                except:
                    return 1e10
            
            # Initial parameter estimates
            N0_init = N_obs[0]
            K_init = max(N_obs) * 1.2  # Assume K is slightly higher than max observed
            r_init = 0.1
            
            # Parameter bounds
            bounds = [
                (N_obs.min() * 0.1, N_obs.max() * 2),  # N0
                (0.001, 2.0),  # r
                (N_obs.max(), N_obs.max() * 5)  # K
            ]
            
            # Optimize parameters
            result = differential_evolution(objective, bounds, seed=42)
            
            if result.success:
                N0, r, K = result.x
                
                # Calculate fitted values
                N_fitted = logistic(t_obs, N0, r, K)
                
                # Calculate R²
                ss_res = np.sum((N_obs - N_fitted) ** 2)
                ss_tot = np.sum((N_obs - np.mean(N_obs)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # Generate forecast
                if forecast_periods > 0:
                    t_forecast = np.arange(t_obs[-1] + 1, t_obs[-1] + forecast_periods + 1)
                    N_forecast = logistic(t_forecast, N0, r, K)
                else:
                    t_forecast = np.array([])
                    N_forecast = np.array([])
                
                # Calculate time to reach K/2
                time_to_half_K = np.log((K - N0) / N0) / r
                
                logistic_results = {
                    'model': 'logistic_growth',
                    'parameters': {
                        'initial_abundance_N0': float(N0),
                        'intrinsic_growth_rate_r': float(r),
                        'carrying_capacity_K': float(K)
                    },
                    'model_fit': {
                        'r_squared': float(r_squared),
                        'fitted_values': N_fitted.tolist(),
                        'optimization_success': True
                    },
                    'biological_interpretation': {
                        'carrying_capacity': float(K),
                        'maximum_growth_rate': float(r * K / 4),  # At N = K/2
                        'time_to_half_capacity': float(time_to_half_K),
                        'current_capacity_utilization': float(N_obs[-1] / K * 100)
                    },
                    'forecast': {
                        'time_points': t_forecast.tolist(),
                        'predicted_abundance': N_forecast.tolist(),
                        'forecast_periods': forecast_periods
                    },
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"📈 Logistic growth model fitted: r={r:.4f}, K={K:.2f}, R²={r_squared:.3f}")
                return logistic_results
                
            else:
                return {'error': 'Logistic model optimization failed to converge'}
                
        except Exception as e:
            logger.error(f"❌ Logistic growth modeling failed: {e}")
            return {'error': str(e)}
    
    def mortality_analysis(self, 
                          survival_data: Dict[str, List[float]],
                          age_classes: List[int]) -> Dict[str, Any]:
        """
        Analyze mortality patterns from survival data
        
        Args:
            survival_data: Dictionary with survival rates by age/time
            age_classes: Age class identifiers
            
        Returns:
            Mortality analysis results
        """
        try:
            survival_rates = np.array(survival_data.get('survival_rates', []))
            ages = np.array(age_classes)
            
            if len(survival_rates) != len(ages):
                raise ValueError("Survival rates and age classes must have equal length")
            
            # Calculate mortality rates
            mortality_rates = 1 - survival_rates
            
            # Calculate instantaneous mortality rates
            instantaneous_mortality = -np.log(survival_rates)
            
            # Calculate life expectancy at each age
            life_expectancy = np.zeros_like(ages, dtype=float)
            for i, age in enumerate(ages):
                # Sum of future survival probabilities
                if i < len(ages) - 1:
                    future_survival = np.prod(survival_rates[i:])
                    life_expectancy[i] = future_survival
                else:
                    life_expectancy[i] = 0
            
            # Calculate survivorship curve (lx)
            survivorship = np.ones_like(ages, dtype=float)
            for i in range(1, len(ages)):
                survivorship[i] = survivorship[i-1] * survival_rates[i-1]
            
            # Calculate deaths at each age (dx)
            deaths = np.zeros_like(ages, dtype=float)
            deaths[:-1] = survivorship[:-1] * mortality_rates[:-1]
            deaths[-1] = survivorship[-1]  # All remaining die in last age class
            
            # Calculate mortality hazard (qx = mx / (1 + mx/2))
            hazard_rates = mortality_rates / (1 + mortality_rates / 2)
            
            mortality_results = {
                'analysis_type': 'mortality_analysis',
                'age_classes': ages.tolist(),
                'survival_rates': survival_rates.tolist(),
                'mortality_rates': mortality_rates.tolist(),
                'instantaneous_mortality': instantaneous_mortality.tolist(),
                'survivorship_curve': survivorship.tolist(),
                'deaths_by_age': deaths.tolist(),
                'hazard_rates': hazard_rates.tolist(),
                'life_table_statistics': {
                    'mean_mortality_rate': float(np.mean(mortality_rates)),
                    'peak_mortality_age': int(ages[np.argmax(mortality_rates)]),
                    'total_mortality': float(np.sum(deaths))
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info("📈 Mortality analysis completed")
            return mortality_results
            
        except Exception as e:
            logger.error(f"❌ Mortality analysis failed: {e}")
            return {'error': str(e)}
    
    def recruitment_analysis(self, 
                            recruitment_data: List[float],
                            spawning_biomass: List[float],
                            years: List[int]) -> Dict[str, Any]:
        """
        Analyze stock-recruitment relationship
        
        Args:
            recruitment_data: Recruitment values
            spawning_biomass: Spawning stock biomass values
            years: Year identifiers
            
        Returns:
            Stock-recruitment analysis results
        """
        try:
            R = np.array(recruitment_data)
            SSB = np.array(spawning_biomass)
            
            if len(R) != len(SSB) or len(R) != len(years):
                raise ValueError("All arrays must have equal length")
            
            # Beverton-Holt model: R = a*SSB / (1 + b*SSB)
            def beverton_holt(SSB, a, b):
                return a * SSB / (1 + b * SSB)
            
            # Ricker model: R = a*SSB*exp(-b*SSB)
            def ricker(SSB, a, b):
                return a * SSB * np.exp(-b * SSB)
            
            # Fit Beverton-Holt model
            try:
                from scipy.optimize import curve_fit
                bh_params, _ = curve_fit(beverton_holt, SSB, R, p0=[1, 0.001])
                bh_fitted = beverton_holt(SSB, *bh_params)
                bh_r2 = 1 - np.sum((R - bh_fitted)**2) / np.sum((R - np.mean(R))**2)
                bh_success = True
            except:
                bh_params = [np.nan, np.nan]
                bh_fitted = np.full_like(R, np.nan)
                bh_r2 = np.nan
                bh_success = False
            
            # Fit Ricker model
            try:
                ricker_params, _ = curve_fit(ricker, SSB, R, p0=[1, 0.001])
                ricker_fitted = ricker(SSB, *ricker_params)
                ricker_r2 = 1 - np.sum((R - ricker_fitted)**2) / np.sum((R - np.mean(R))**2)
                ricker_success = True
            except:
                ricker_params = [np.nan, np.nan]
                ricker_fitted = np.full_like(R, np.nan)
                ricker_r2 = np.nan
                ricker_success = False
            
            # Simple linear relationship
            linear_slope, linear_intercept = np.polyfit(SSB, R, 1)
            linear_fitted = linear_slope * SSB + linear_intercept
            linear_r2 = 1 - np.sum((R - linear_fitted)**2) / np.sum((R - np.mean(R))**2)
            
            # Determine best model
            models = {
                'linear': {'r2': linear_r2, 'success': True},
                'beverton_holt': {'r2': bh_r2, 'success': bh_success},
                'ricker': {'r2': ricker_r2, 'success': ricker_success}
            }
            
            valid_models = {k: v for k, v in models.items() if v['success'] and not np.isnan(v['r2'])}
            best_model = max(valid_models.keys(), key=lambda k: valid_models[k]['r2']) if valid_models else 'linear'
            
            recruitment_results = {
                'analysis_type': 'stock_recruitment',
                'years': years,
                'data': {
                    'recruitment': R.tolist(),
                    'spawning_biomass': SSB.tolist()
                },
                'models': {
                    'linear': {
                        'slope': float(linear_slope),
                        'intercept': float(linear_intercept),
                        'fitted_values': linear_fitted.tolist(),
                        'r_squared': float(linear_r2)
                    },
                    'beverton_holt': {
                        'parameters': {'a': float(bh_params[0]), 'b': float(bh_params[1])},
                        'fitted_values': bh_fitted.tolist(),
                        'r_squared': float(bh_r2),
                        'success': bh_success
                    },
                    'ricker': {
                        'parameters': {'a': float(ricker_params[0]), 'b': float(ricker_params[1])},
                        'fitted_values': ricker_fitted.tolist(),
                        'r_squared': float(ricker_r2),
                        'success': ricker_success
                    }
                },
                'best_model': best_model,
                'recruitment_statistics': {
                    'mean_recruitment': float(np.mean(R)),
                    'recruitment_variability': float(np.std(R) / np.mean(R)),
                    'recruitment_range': [float(np.min(R)), float(np.max(R))]
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"📈 Recruitment analysis completed: best model = {best_model}")
            return recruitment_results
            
        except Exception as e:
            logger.error(f"❌ Recruitment analysis failed: {e}")
            return {'error': str(e)}

# Global population modeler instance
population_modeler = PopulationModeler()