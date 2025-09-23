"""
📈 Stock Assessment Engine - Fisheries Stock Evaluation & Management

Advanced stock assessment models for marine fisheries.
Implements surplus production models, VPA, and yield-per-recruit analysis.

Author: Ocean-Bio Development Team
Version: 2.0.0
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
import warnings
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class StockAssessmentEngine:
    """
    📈 Advanced Stock Assessment Engine
    
    Provides comprehensive fisheries stock assessment capabilities:
    - Surplus production models (Schaefer, Fox, Pella-Tomlinson)
    - Virtual Population Analysis (VPA)
    - Yield-per-recruit analysis
    - Reference point estimation (MSY, FMSY, BMSY)
    - Stock status evaluation and forecasting
    """
    
    def __init__(self):
        """Initialize the stock assessment engine"""
        self.models = {
            'schaefer': self._schaefer_model,
            'fox': self._fox_model,
            'pella_tomlinson': self._pella_tomlinson_model
        }
        self.reference_points = {}
        self.assessment_results = {}
        
    def schaefer_assessment(self, 
                           catch_data: List[float],
                           abundance_index: List[float],
                           years: List[int],
                           initial_params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Perform Schaefer surplus production model assessment
        
        Args:
            catch_data: Annual catch data
            abundance_index: Abundance index (CPUE, survey data)
            years: Year vector
            initial_params: Initial parameter guesses
            
        Returns:
            Assessment results with parameters and reference points
        """
        try:
            # Validate input data
            if len(catch_data) != len(abundance_index) or len(catch_data) != len(years):
                raise ValueError("Data vectors must have equal length")
            
            # Convert to pandas DataFrame
            data = pd.DataFrame({
                'year': years,
                'catch': catch_data,
                'abundance': abundance_index
            })
            
            # Set initial parameters if not provided
            if initial_params is None:
                initial_params = {
                    'r': 0.2,  # Intrinsic growth rate
                    'K': max(abundance_index) * 2,  # Carrying capacity
                    'q': 0.001,  # Catchability coefficient
                    'sigma': 0.1  # Process error
                }
            
            # Define parameter bounds
            bounds = [
                (0.01, 2.0),  # r
                (max(abundance_index), max(abundance_index) * 10),  # K
                (1e-6, 1.0),  # q
                (0.01, 1.0)   # sigma
            ]
            
            # Fit the model using maximum likelihood
            result = differential_evolution(
                self._schaefer_likelihood,
                bounds,
                args=(data,),
                seed=42,
                maxiter=1000
            )
            
            if result.success:
                r, K, q, sigma = result.x
                
                # Calculate derived quantities
                MSY = r * K / 4  # Maximum Sustainable Yield
                BMSY = K / 2     # Biomass at MSY
                FMSY = r / 2     # Fishing mortality at MSY
                
                # Calculate biomass trajectory
                biomass = self._calculate_biomass_trajectory(data['catch'].values, r, K)
                
                # Calculate fishing mortality
                F = data['catch'].values / biomass[:-1]  # Remove last biomass (no catch after)
                
                # Current stock status
                current_biomass = biomass[-1]
                current_F = F[-1] if len(F) > 0 else 0
                
                # Stock status indicators
                B_ratio = current_biomass / BMSY
                F_ratio = current_F / FMSY if FMSY > 0 else 0
                
                # Determine stock status
                if B_ratio >= 1.0 and F_ratio <= 1.0:
                    status = "Healthy"
                elif B_ratio >= 0.8 and F_ratio <= 1.2:
                    status = "Caution"
                elif B_ratio >= 0.5:
                    status = "Overfished"
                else:
                    status = "Severely Depleted"
                
                assessment_results = {
                    'model': 'schaefer',
                    'parameters': {
                        'r': float(r),
                        'K': float(K),
                        'q': float(q),
                        'sigma': float(sigma)
                    },
                    'reference_points': {
                        'MSY': float(MSY),
                        'BMSY': float(BMSY),
                        'FMSY': float(FMSY)
                    },
                    'time_series': {
                        'years': years,
                        'catch': catch_data,
                        'abundance_index': abundance_index,
                        'estimated_biomass': biomass.tolist(),
                        'fishing_mortality': F.tolist(),
                        'predicted_abundance': (q * biomass).tolist()
                    },
                    'current_status': {
                        'biomass': float(current_biomass),
                        'fishing_mortality': float(current_F),
                        'B_BMSY_ratio': float(B_ratio),
                        'F_FMSY_ratio': float(F_ratio),
                        'status': status
                    },
                    'model_fit': {
                        'log_likelihood': float(-result.fun),
                        'AIC': float(2 * 4 - 2 * (-result.fun)),  # 4 parameters
                        'convergence': result.success
                    }
                }
                
                logger.info(f"📈 Schaefer assessment completed: MSY={MSY:.2f}, Status={status}")
                return assessment_results
                
            else:
                logger.error("❌ Schaefer model optimization failed")
                return {'error': 'Optimization failed', 'details': result.message}
                
        except Exception as e:
            logger.error(f"❌ Schaefer assessment failed: {e}")
            return {'error': str(e)}
    
    def fox_assessment(self, 
                      catch_data: List[float],
                      abundance_index: List[float],
                      years: List[int]) -> Dict[str, Any]:
        """Perform Fox surplus production model assessment"""
        try:
            data = pd.DataFrame({
                'year': years,
                'catch': catch_data,
                'abundance': abundance_index
            })
            
            # Initial parameters for Fox model
            initial_params = {
                'r': 0.2,
                'K': max(abundance_index) * 2,
                'q': 0.001,
                'sigma': 0.1
            }
            
            bounds = [
                (0.01, 2.0),
                (max(abundance_index), max(abundance_index) * 10),
                (1e-6, 1.0),
                (0.01, 1.0)
            ]
            
            result = differential_evolution(
                self._fox_likelihood,
                bounds,
                args=(data,),
                seed=42,
                maxiter=1000
            )
            
            if result.success:
                r, K, q, sigma = result.x
                
                # Fox model reference points
                MSY = r * K / np.e  # Maximum Sustainable Yield for Fox
                BMSY = K / np.e     # Biomass at MSY for Fox
                FMSY = r / np.e     # Fishing mortality at MSY for Fox
                
                # Calculate biomass trajectory using Fox dynamics
                biomass = self._calculate_fox_biomass_trajectory(data['catch'].values, r, K)
                
                current_biomass = biomass[-1]
                F = data['catch'].values / biomass[:-1]
                current_F = F[-1] if len(F) > 0 else 0
                
                B_ratio = current_biomass / BMSY
                F_ratio = current_F / FMSY if FMSY > 0 else 0
                
                # Stock status
                if B_ratio >= 1.0 and F_ratio <= 1.0:
                    status = "Healthy"
                elif B_ratio >= 0.8:
                    status = "Caution"
                elif B_ratio >= 0.5:
                    status = "Overfished"
                else:
                    status = "Severely Depleted"
                
                return {
                    'model': 'fox',
                    'parameters': {
                        'r': float(r),
                        'K': float(K),
                        'q': float(q),
                        'sigma': float(sigma)
                    },
                    'reference_points': {
                        'MSY': float(MSY),
                        'BMSY': float(BMSY),
                        'FMSY': float(FMSY)
                    },
                    'time_series': {
                        'years': years,
                        'catch': catch_data,
                        'abundance_index': abundance_index,
                        'estimated_biomass': biomass.tolist(),
                        'fishing_mortality': F.tolist()
                    },
                    'current_status': {
                        'biomass': float(current_biomass),
                        'fishing_mortality': float(current_F),
                        'B_BMSY_ratio': float(B_ratio),
                        'F_FMSY_ratio': float(F_ratio),
                        'status': status
                    },
                    'model_fit': {
                        'log_likelihood': float(-result.fun),
                        'AIC': float(2 * 4 - 2 * (-result.fun)),
                        'convergence': result.success
                    }
                }
                
            else:
                return {'error': 'Fox model optimization failed'}
                
        except Exception as e:
            logger.error(f"❌ Fox assessment failed: {e}")
            return {'error': str(e)}
    
    def yield_per_recruit_analysis(self, 
                                 life_history_params: Dict[str, float],
                                 F_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Perform yield-per-recruit and spawning-stock-biomass-per-recruit analysis
        
        Args:
            life_history_params: Dictionary with M, growth parameters, maturity, etc.
            F_range: Range of fishing mortality rates to evaluate
            
        Returns:
            YPR and SPR analysis results
        """
        try:
            # Default parameters
            M = life_history_params.get('M', 0.2)  # Natural mortality
            Linf = life_history_params.get('Linf', 100)  # Asymptotic length
            K = life_history_params.get('K', 0.2)  # Growth rate
            t0 = life_history_params.get('t0', 0)  # Age at length 0
            a = life_history_params.get('a', 0.01)  # Length-weight parameter
            b = life_history_params.get('b', 3.0)  # Length-weight exponent
            age_mat = life_history_params.get('age_mat', 3)  # Age at maturity
            max_age = life_history_params.get('max_age', 20)  # Maximum age
            
            if F_range is None:
                F_range = (0, 2.0)
            
            F_values = np.linspace(F_range[0], F_range[1], 100)
            ages = np.arange(0, max_age + 1)
            
            YPR_values = []
            SPR_values = []
            
            for F in F_values:
                # Calculate survivorship
                Z = M + F  # Total mortality
                survivorship = np.exp(-Z * ages)
                
                # Calculate length-at-age (von Bertalanffy)
                length_at_age = Linf * (1 - np.exp(-K * (ages - t0)))
                
                # Calculate weight-at-age
                weight_at_age = a * (length_at_age ** b)
                
                # Calculate maturity-at-age (knife-edge at age_mat)
                maturity_at_age = np.where(ages >= age_mat, 1.0, 0.0)
                
                # Calculate fishing selectivity (assume full selectivity at age 1+)
                selectivity = np.where(ages >= 1, 1.0, 0.0)
                
                # Yield per recruit
                if F > 0:
                    fishing_mortality_at_age = F * selectivity
                    catch_at_age = (fishing_mortality_at_age / Z) * (1 - survivorship) * weight_at_age
                    catch_at_age = np.where(Z > 0, catch_at_age, 0)  # Avoid division by zero
                    YPR = np.sum(catch_at_age)
                else:
                    YPR = 0
                
                # Spawning stock biomass per recruit
                SSB = np.sum(survivorship * weight_at_age * maturity_at_age)
                
                YPR_values.append(YPR)
                SPR_values.append(SSB)
            
            # Find reference points
            max_YPR_idx = np.argmax(YPR_values)
            F_max = F_values[max_YPR_idx]
            max_YPR = YPR_values[max_YPR_idx]
            
            # F_0.1 (F where slope is 10% of origin slope)
            if len(YPR_values) > 1:
                origin_slope = YPR_values[1] / F_values[1] if F_values[1] > 0 else 0
                target_slope = 0.1 * origin_slope
                
                # Find F_0.1 by finding where slope drops to 10% of origin
                F_01_idx = 0
                for i in range(1, len(YPR_values) - 1):
                    if F_values[i] > 0:
                        current_slope = (YPR_values[i+1] - YPR_values[i]) / (F_values[i+1] - F_values[i])
                        if current_slope <= target_slope:
                            F_01_idx = i
                            break
                
                F_01 = F_values[F_01_idx]
                YPR_01 = YPR_values[F_01_idx]
            else:
                F_01 = 0
                YPR_01 = 0
            
            # SPR reference points
            virgin_SSB = SPR_values[0]  # SSB when F=0
            SPR_ratios = np.array(SPR_values) / virgin_SSB if virgin_SSB > 0 else np.zeros_like(SPR_values)
            
            # Find F corresponding to SPR30% and SPR20%
            F_SPR30_idx = np.argmin(np.abs(SPR_ratios - 0.3))
            F_SPR20_idx = np.argmin(np.abs(SPR_ratios - 0.2))
            
            F_SPR30 = F_values[F_SPR30_idx]
            F_SPR20 = F_values[F_SPR20_idx]
            
            analysis_results = {
                'life_history_params': life_history_params,
                'F_range': F_range,
                'curves': {
                    'F_values': F_values.tolist(),
                    'YPR_values': YPR_values,
                    'SPR_values': SPR_values,
                    'SPR_ratios': SPR_ratios.tolist()
                },
                'reference_points': {
                    'F_max': float(F_max),
                    'max_YPR': float(max_YPR),
                    'F_01': float(F_01),
                    'YPR_01': float(YPR_01),
                    'F_SPR30': float(F_SPR30),
                    'F_SPR20': float(F_SPR20),
                    'virgin_SSB': float(virgin_SSB)
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"📈 YPR analysis completed: F_max={F_max:.3f}, F_0.1={F_01:.3f}")
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ YPR analysis failed: {e}")
            return {'error': str(e)}
    
    def stock_projection(self, 
                        assessment_results: Dict[str, Any],
                        projection_years: int = 10,
                        catch_scenarios: Optional[List[float]] = None) -> Dict[str, Any]:
        """Project stock biomass under different catch scenarios"""
        try:
            if 'parameters' not in assessment_results:
                raise ValueError("Assessment results must contain parameters")
            
            params = assessment_results['parameters']
            r = params['r']
            K = params['K']
            
            # Get current biomass
            current_biomass = assessment_results['current_status']['biomass']
            
            # Default catch scenarios
            if catch_scenarios is None:
                current_catch = assessment_results['time_series']['catch'][-1]
                catch_scenarios = [0, current_catch * 0.5, current_catch, current_catch * 1.5]
            
            projections = {}
            
            for scenario_idx, catch_level in enumerate(catch_scenarios):
                biomass_trajectory = [current_biomass]
                
                for year in range(projection_years):
                    B = biomass_trajectory[-1]
                    
                    # Schaefer growth
                    if assessment_results['model'] == 'schaefer':
                        growth = r * B * (1 - B / K)
                    elif assessment_results['model'] == 'fox':
                        growth = r * B * np.log(K / B) if B > 0 and B < K else 0
                    else:
                        growth = r * B * (1 - B / K)  # Default to Schaefer
                    
                    # Next year biomass
                    next_biomass = max(0, B + growth - catch_level)
                    biomass_trajectory.append(next_biomass)
                
                projections[f'scenario_{scenario_idx}'] = {
                    'catch_level': catch_level,
                    'biomass_trajectory': biomass_trajectory,
                    'final_biomass': biomass_trajectory[-1],
                    'biomass_change': biomass_trajectory[-1] - current_biomass
                }
            
            return {
                'projection_years': projection_years,
                'current_biomass': current_biomass,
                'scenarios': projections,
                'projection_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Stock projection failed: {e}")
            return {'error': str(e)}
    
    def _schaefer_likelihood(self, params: np.ndarray, data: pd.DataFrame) -> float:
        """Calculate negative log-likelihood for Schaefer model"""
        try:
            r, K, q, sigma = params
            
            # Calculate predicted biomass
            biomass = self._calculate_biomass_trajectory(data['catch'].values, r, K)
            
            # Calculate predicted abundance index
            predicted_abundance = q * biomass[:-1]  # Remove last biomass (no observation)
            observed_abundance = data['abundance'].values
            
            # Calculate log-likelihood (assuming log-normal errors)
            if len(predicted_abundance) != len(observed_abundance):
                return 1e10  # Large penalty for dimension mismatch
            
            # Avoid log of negative or zero values
            predicted_abundance = np.maximum(predicted_abundance, 1e-10)
            observed_abundance = np.maximum(observed_abundance, 1e-10)
            
            log_residuals = np.log(observed_abundance) - np.log(predicted_abundance)
            neg_log_likelihood = 0.5 * len(log_residuals) * np.log(2 * np.pi * sigma**2) + \
                               0.5 * np.sum(log_residuals**2) / sigma**2
            
            return neg_log_likelihood
            
        except Exception as e:
            return 1e10  # Large penalty for errors
    
    def _fox_likelihood(self, params: np.ndarray, data: pd.DataFrame) -> float:
        """Calculate negative log-likelihood for Fox model"""
        try:
            r, K, q, sigma = params
            
            biomass = self._calculate_fox_biomass_trajectory(data['catch'].values, r, K)
            predicted_abundance = q * biomass[:-1]
            observed_abundance = data['abundance'].values
            
            if len(predicted_abundance) != len(observed_abundance):
                return 1e10
            
            predicted_abundance = np.maximum(predicted_abundance, 1e-10)
            observed_abundance = np.maximum(observed_abundance, 1e-10)
            
            log_residuals = np.log(observed_abundance) - np.log(predicted_abundance)
            neg_log_likelihood = 0.5 * len(log_residuals) * np.log(2 * np.pi * sigma**2) + \
                               0.5 * np.sum(log_residuals**2) / sigma**2
            
            return neg_log_likelihood
            
        except Exception as e:
            return 1e10
    
    def _calculate_biomass_trajectory(self, catches: np.ndarray, r: float, K: float) -> np.ndarray:
        """Calculate biomass trajectory for Schaefer model"""
        n_years = len(catches)
        biomass = np.zeros(n_years + 1)
        biomass[0] = K  # Assume unfished biomass at start
        
        for t in range(n_years):
            B = biomass[t]
            growth = r * B * (1 - B / K)
            biomass[t + 1] = max(0, B + growth - catches[t])
        
        return biomass
    
    def _calculate_fox_biomass_trajectory(self, catches: np.ndarray, r: float, K: float) -> np.ndarray:
        """Calculate biomass trajectory for Fox model"""
        n_years = len(catches)
        biomass = np.zeros(n_years + 1)
        biomass[0] = K  # Assume unfished biomass at start
        
        for t in range(n_years):
            B = biomass[t]
            if B > 0 and B < K:
                growth = r * B * np.log(K / B)
            else:
                growth = 0
            biomass[t + 1] = max(0, B + growth - catches[t])
        
        return biomass
    
    def compare_models(self, 
                      catch_data: List[float],
                      abundance_index: List[float],
                      years: List[int]) -> Dict[str, Any]:
        """Compare different stock assessment models and select best"""
        try:
            models_results = {}
            
            # Run Schaefer assessment
            schaefer_result = self.schaefer_assessment(catch_data, abundance_index, years)
            if 'error' not in schaefer_result:
                models_results['schaefer'] = schaefer_result
            
            # Run Fox assessment
            fox_result = self.fox_assessment(catch_data, abundance_index, years)
            if 'error' not in fox_result:
                models_results['fox'] = fox_result
            
            if not models_results:
                return {'error': 'No models converged successfully'}
            
            # Compare models using AIC
            best_model = None
            best_aic = float('inf')
            
            for model_name, result in models_results.items():
                aic = result.get('model_fit', {}).get('AIC', float('inf'))
                if aic < best_aic:
                    best_aic = aic
                    best_model = model_name
            
            return {
                'models_compared': list(models_results.keys()),
                'model_results': models_results,
                'best_model': best_model,
                'best_aic': best_aic,
                'comparison_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Model comparison failed: {e}")
            return {'error': str(e)}

# Global stock assessment instance
stock_assessor = StockAssessmentEngine()