"""
📈 Forecasting Engine - Time-series Analysis & Prediction

Advanced time-series forecasting for marine data.
Implements ARIMA, Prophet, and seasonal decomposition models.

Author: Ocean-Bio Development Team
Version: 2.0.0
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Initialize logger early
logger = logging.getLogger(__name__)

# Statistical and ML libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import VAR
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Try to import Prophet, handle if not available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not available. Time-series forecasting will use ARIMA only.")

class ForecastingEngine:
    """
    📈 Advanced Time-series Forecasting Engine
    
    Provides comprehensive forecasting capabilities:
    - ARIMA modeling for univariate series
    - Prophet for seasonal and trend decomposition
    - VAR for multivariate time-series
    - Seasonal decomposition and analysis
    - Forecast accuracy assessment
    """
    
    def __init__(self):
        """Initialize the forecasting engine"""
        self.models = {}
        self.forecasts = {}
        self.supported_models = ['arima', 'prophet', 'var', 'seasonal_naive']
        if PROPHET_AVAILABLE:
            self.supported_models.append('prophet')
    
    def arima_forecast(self, 
                      data: List[float],
                      dates: List[str],
                      forecast_periods: int = 12,
                      order: Optional[Tuple[int, int, int]] = None,
                      seasonal_order: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Perform ARIMA forecasting
        
        Args:
            data: Time-series data values
            dates: Corresponding dates
            forecast_periods: Number of periods to forecast
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal ARIMA order (P, D, Q, s)
            
        Returns:
            Forecast results with confidence intervals
        """
        try:
            # Create DataFrame
            df = pd.DataFrame({
                'date': pd.to_datetime(dates),
                'value': data
            })
            df = df.set_index('date').sort_index()
            
            # Check for stationarity
            stationarity_test = self._test_stationarity(df['value'])
            
            # Auto-determine ARIMA order if not provided
            if order is None:
                order = self._auto_arima_order(df['value'])
            
            # Fit ARIMA model
            model = ARIMA(df['value'], order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit()
            
            # Generate forecasts
            forecast_result = fitted_model.forecast(steps=forecast_periods, alpha=0.05)
            
            # Get confidence intervals
            forecast_ci = fitted_model.get_forecast(steps=forecast_periods).conf_int()
            
            # Create forecast dates
            last_date = df.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.infer_freq(df.index),
                periods=forecast_periods,
                freq=pd.infer_freq(df.index)
            )
            
            # Calculate in-sample fit
            fitted_values = fitted_model.fittedvalues
            residuals = fitted_model.resid
            
            # Model diagnostics
            aic = fitted_model.aic
            bic = fitted_model.bic
            
            # Accuracy metrics on training data
            mse = np.mean(residuals**2)
            mae = np.mean(np.abs(residuals))
            mape = np.mean(np.abs(residuals / df['value'])) * 100
            
            forecast_results = {
                'model': 'arima',
                'model_order': order,
                'seasonal_order': seasonal_order,
                'stationarity_test': stationarity_test,
                'historical_data': {
                    'dates': df.index.strftime('%Y-%m-%d').tolist(),
                    'values': df['value'].tolist(),
                    'fitted_values': fitted_values.tolist()
                },
                'forecast': {
                    'dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
                    'values': forecast_result.tolist(),
                    'lower_ci': forecast_ci.iloc[:, 0].tolist(),
                    'upper_ci': forecast_ci.iloc[:, 1].tolist()
                },
                'model_diagnostics': {
                    'aic': float(aic),
                    'bic': float(bic),
                    'mse': float(mse),
                    'mae': float(mae),
                    'mape': float(mape)
                },
                'residuals': {
                    'values': residuals.tolist(),
                    'ljung_box_p': float(fitted_model.test_serial_correlation('ljungbox')[0].iloc[-1, 1])
                },
                'forecast_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"📈 ARIMA forecast completed: {forecast_periods} periods, AIC={aic:.2f}")
            return forecast_results
            
        except Exception as e:
            logger.error(f"❌ ARIMA forecasting failed: {e}")
            return {'error': str(e)}
    
    def prophet_forecast(self, 
                        data: List[float],
                        dates: List[str],
                        forecast_periods: int = 12) -> Dict[str, Any]:
        """
        Perform Prophet forecasting (if available)
        
        Args:
            data: Time-series data values
            dates: Corresponding dates
            forecast_periods: Number of periods to forecast
            
        Returns:
            Prophet forecast results
        """
        try:
            if not PROPHET_AVAILABLE:
                return {'error': 'Prophet is not available'}
            
            # Prepare data for Prophet
            df = pd.DataFrame({
                'ds': pd.to_datetime(dates),
                'y': data
            })
            
            # Initialize and fit Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            # Fit the model
            model.fit(df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=forecast_periods)
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Split historical and forecast data
            historical_size = len(df)
            
            historical_forecast = forecast.iloc[:historical_size]
            future_forecast = forecast.iloc[historical_size:]
            
            # Calculate accuracy metrics
            residuals = df['y'] - historical_forecast['yhat']
            mse = np.mean(residuals**2)
            mae = np.mean(np.abs(residuals))
            mape = np.mean(np.abs(residuals / df['y'])) * 100
            
            forecast_results = {
                'model': 'prophet',
                'historical_data': {
                    'dates': df['ds'].dt.strftime('%Y-%m-%d').tolist(),
                    'values': df['y'].tolist(),
                    'fitted_values': historical_forecast['yhat'].tolist(),
                    'trend': historical_forecast['trend'].tolist(),
                    'seasonal': (historical_forecast['yearly'] + historical_forecast.get('weekly', 0)).tolist()
                },
                'forecast': {
                    'dates': future_forecast['ds'].dt.strftime('%Y-%m-%d').tolist(),
                    'values': future_forecast['yhat'].tolist(),
                    'lower_ci': future_forecast['yhat_lower'].tolist(),
                    'upper_ci': future_forecast['yhat_upper'].tolist(),
                    'trend': future_forecast['trend'].tolist(),
                    'seasonal': (future_forecast['yearly'] + future_forecast.get('weekly', 0)).tolist()
                },
                'components': {
                    'trend_changepoints': model.changepoints.strftime('%Y-%m-%d').tolist(),
                    'seasonality_components': ['yearly'] + (['weekly'] if model.weekly_seasonality else [])
                },
                'model_diagnostics': {
                    'mse': float(mse),
                    'mae': float(mae),
                    'mape': float(mape)
                },
                'forecast_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"📈 Prophet forecast completed: {forecast_periods} periods")
            return forecast_results
            
        except Exception as e:
            logger.error(f"❌ Prophet forecasting failed: {e}")
            return {'error': str(e)}
    
    def seasonal_decomposition(self, 
                              data: List[float],
                              dates: List[str],
                              model_type: str = 'additive',
                              period: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform seasonal decomposition of time-series
        
        Args:
            data: Time-series data values
            dates: Corresponding dates
            model_type: 'additive' or 'multiplicative'
            period: Seasonal period (auto-detected if None)
            
        Returns:
            Decomposition results
        """
        try:
            # Create DataFrame
            df = pd.DataFrame({
                'date': pd.to_datetime(dates),
                'value': data
            })
            df = df.set_index('date').sort_index()
            
            # Auto-detect period if not provided
            if period is None:
                freq = pd.infer_freq(df.index)
                if freq:
                    if 'M' in freq:  # Monthly data
                        period = 12
                    elif 'Q' in freq:  # Quarterly data
                        period = 4
                    elif 'D' in freq:  # Daily data
                        period = 365
                    else:
                        period = min(len(data) // 2, 12)  # Default fallback
                else:
                    period = min(len(data) // 2, 12)
            
            # Perform decomposition
            decomposition = seasonal_decompose(
                df['value'], 
                model=model_type, 
                period=period,
                extrapolate_trend='freq'
            )
            
            # Calculate seasonal strength
            seasonal_strength = 1 - np.var(decomposition.resid.dropna()) / np.var(decomposition.seasonal + decomposition.resid).dropna()
            
            # Calculate trend strength
            trend_strength = 1 - np.var(decomposition.resid.dropna()) / np.var(decomposition.trend + decomposition.resid).dropna()
            
            decomposition_results = {
                'model_type': model_type,
                'period': period,
                'dates': df.index.strftime('%Y-%m-%d').tolist(),
                'components': {
                    'original': df['value'].tolist(),
                    'trend': decomposition.trend.tolist(),
                    'seasonal': decomposition.seasonal.tolist(),
                    'residual': decomposition.resid.tolist()
                },
                'strength_metrics': {
                    'seasonal_strength': float(seasonal_strength),
                    'trend_strength': float(trend_strength)
                },
                'seasonal_pattern': {
                    'period': period,
                    'average_seasonal': decomposition.seasonal[:period].tolist()
                },
                'decomposition_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"📈 Seasonal decomposition completed: period={period}, type={model_type}")
            return decomposition_results
            
        except Exception as e:
            logger.error(f"❌ Seasonal decomposition failed: {e}")
            return {'error': str(e)}
    
    def var_forecast(self, 
                    data_dict: Dict[str, List[float]],
                    dates: List[str],
                    forecast_periods: int = 12,
                    max_lags: int = 5) -> Dict[str, Any]:
        """
        Perform Vector Autoregression (VAR) forecasting for multivariate time-series
        
        Args:
            data_dict: Dictionary of variable names and their time-series data
            dates: Corresponding dates
            forecast_periods: Number of periods to forecast
            max_lags: Maximum number of lags to consider
            
        Returns:
            VAR forecast results
        """
        try:
            # Create DataFrame
            df = pd.DataFrame(data_dict)
            df['date'] = pd.to_datetime(dates)
            df = df.set_index('date').sort_index()
            
            # Fit VAR model
            model = VAR(df)
            
            # Select optimal lag order
            lag_order = model.select_order(maxlags=max_lags)
            optimal_lags = lag_order.aic
            
            # Fit model with optimal lags
            fitted_model = model.fit(optimal_lags)
            
            # Generate forecasts
            forecast_result = fitted_model.forecast(df.values, steps=forecast_periods)
            
            # Create forecast dates
            last_date = df.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.infer_freq(df.index),
                periods=forecast_periods,
                freq=pd.infer_freq(df.index)
            )
            
            # Calculate fitted values and residuals
            fitted_values = fitted_model.fittedvalues
            residuals = fitted_model.resid
            
            # Prepare results
            var_results = {
                'model': 'var',
                'variables': list(data_dict.keys()),
                'optimal_lags': optimal_lags,
                'historical_data': {
                    'dates': df.index.strftime('%Y-%m-%d').tolist(),
                    'values': {var: df[var].tolist() for var in df.columns}
                },
                'fitted_values': {
                    var: fitted_values[var].tolist() for var in df.columns
                },
                'forecast': {
                    'dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
                    'values': {
                        var: forecast_result[:, i].tolist() 
                        for i, var in enumerate(df.columns)
                    }
                },
                'model_diagnostics': {
                    'aic': float(fitted_model.aic),
                    'bic': float(fitted_model.bic),
                    'log_likelihood': float(fitted_model.llf)
                },
                'residuals': {
                    var: residuals[var].tolist() for var in df.columns
                },
                'forecast_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"📈 VAR forecast completed: {len(df.columns)} variables, {optimal_lags} lags")
            return var_results
            
        except Exception as e:
            logger.error(f"❌ VAR forecasting failed: {e}")
            return {'error': str(e)}
    
    def forecast_accuracy(self, 
                         actual: List[float],
                         predicted: List[float]) -> Dict[str, float]:
        """Calculate forecast accuracy metrics"""
        try:
            actual = np.array(actual)
            predicted = np.array(predicted)
            
            # Basic error metrics
            errors = actual - predicted
            abs_errors = np.abs(errors)
            squared_errors = errors**2
            
            # Calculate metrics
            mae = np.mean(abs_errors)
            mse = np.mean(squared_errors)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs(errors / actual)) * 100 if np.all(actual != 0) else np.inf
            
            # Additional metrics
            mean_actual = np.mean(actual)
            ss_res = np.sum(squared_errors)
            ss_tot = np.sum((actual - mean_actual)**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Directional accuracy
            actual_changes = np.diff(actual)
            predicted_changes = np.diff(predicted)
            directional_accuracy = np.mean(
                np.sign(actual_changes) == np.sign(predicted_changes)
            ) * 100 if len(actual_changes) > 0 else 0
            
            return {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'r_squared': float(r2),
                'directional_accuracy': float(directional_accuracy)
            }
            
        except Exception as e:
            logger.error(f"❌ Accuracy calculation failed: {e}")
            return {'error': str(e)}
    
    def _test_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """Test time-series stationarity using ADF and KPSS tests"""
        try:
            # Augmented Dickey-Fuller test
            adf_result = adfuller(series.dropna())
            
            # KPSS test
            kpss_result = kpss(series.dropna())
            
            return {
                'adf_test': {
                    'statistic': float(adf_result[0]),
                    'p_value': float(adf_result[1]),
                    'critical_values': {k: float(v) for k, v in adf_result[4].items()},
                    'is_stationary': adf_result[1] < 0.05
                },
                'kpss_test': {
                    'statistic': float(kpss_result[0]),
                    'p_value': float(kpss_result[1]),
                    'critical_values': {k: float(v) for k, v in kpss_result[3].items()},
                    'is_stationary': kpss_result[1] > 0.05
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Stationarity test failed: {e}")
            return {'error': str(e)}
    
    def _auto_arima_order(self, series: pd.Series, max_p: int = 3, max_q: int = 3) -> Tuple[int, int, int]:
        """Automatically determine optimal ARIMA order"""
        try:
            best_aic = float('inf')
            best_order = (1, 1, 1)
            
            # Test stationarity and determine d
            adf_result = adfuller(series.dropna())
            if adf_result[1] > 0.05:  # Non-stationary
                # Try first difference
                diff_series = series.diff().dropna()
                adf_diff = adfuller(diff_series)
                d = 1 if adf_diff[1] <= 0.05 else 2
            else:
                d = 0
            
            # Grid search for p and q
            for p in range(0, max_p + 1):
                for q in range(0, max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
            
            return best_order
            
        except Exception as e:
            logger.error(f"❌ Auto ARIMA order selection failed: {e}")
            return (1, 1, 1)  # Default fallback
    
    def create_forecast_visualization(self, 
                                    forecast_results: Dict[str, Any],
                                    title: str = "Time Series Forecast") -> Dict[str, Any]:
        """Create interactive visualization of forecast results"""
        try:
            if 'historical_data' not in forecast_results or 'forecast' not in forecast_results:
                return {'error': 'Invalid forecast results format'}
            
            # Create subplot figure
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Forecast', 'Residuals'),
                row_heights=[0.7, 0.3]
            )
            
            hist_data = forecast_results['historical_data']
            forecast_data = forecast_results['forecast']
            
            # Historical data
            fig.add_trace(
                go.Scatter(
                    x=hist_data['dates'],
                    y=hist_data['values'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Fitted values (if available)
            if 'fitted_values' in hist_data:
                fig.add_trace(
                    go.Scatter(
                        x=hist_data['dates'],
                        y=hist_data['fitted_values'],
                        mode='lines',
                        name='Fitted',
                        line=dict(color='red', dash='dash')
                    ),
                    row=1, col=1
                )
            
            # Forecast
            fig.add_trace(
                go.Scatter(
                    x=forecast_data['dates'],
                    y=forecast_data['values'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='green')
                ),
                row=1, col=1
            )
            
            # Confidence intervals (if available)
            if 'upper_ci' in forecast_data and 'lower_ci' in forecast_data:
                fig.add_trace(
                    go.Scatter(
                        x=forecast_data['dates'],
                        y=forecast_data['upper_ci'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=forecast_data['dates'],
                        y=forecast_data['lower_ci'],
                        mode='lines',
                        line=dict(width=0),
                        name='Confidence Interval',
                        fill='tonexty',
                        fillcolor='rgba(0,100,80,0.2)'
                    ),
                    row=1, col=1
                )
            
            # Residuals (if available)
            if 'residuals' in forecast_results:
                fig.add_trace(
                    go.Scatter(
                        x=hist_data['dates'],
                        y=forecast_results['residuals']['values'],
                        mode='lines',
                        name='Residuals',
                        line=dict(color='orange')
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                title=title,
                height=600,
                showlegend=True
            )
            
            # Convert to JSON for API response
            fig_json = fig.to_json()
            
            return {
                'visualization': fig_json,
                'created_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Forecast visualization failed: {e}")
            return {'error': str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get forecasting engine status"""
        return {
            'service': 'forecasting_engine',
            'status': 'healthy',
            'supported_models': self.supported_models,
            'prophet_available': PROPHET_AVAILABLE,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_available_forecast_types(self) -> List[str]:
        """Get list of available forecast types"""
        types = ['arima', 'seasonal_decompose']
        if PROPHET_AVAILABLE:
            types.append('prophet')
        return types
    
    async def generate_forecast(self,
                              forecast_type: str,
                              time_series_data: List[Dict[str, Any]],
                              forecast_horizon: int,
                              confidence_level: float = 0.95,
                              parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate forecast using specified method"""
        try:
            # Extract data from time series
            dates = [d.get('date', '') for d in time_series_data]
            values = [float(d.get('value', 0)) for d in time_series_data]
            
            if not dates or not values:
                return {'error': 'Invalid time series data'}
            
            # Call appropriate forecasting method
            if forecast_type == 'arima':
                order = parameters.get('order') if parameters else None
                seasonal_order = parameters.get('seasonal_order') if parameters else None
                result = self.arima_forecast(
                    data=values,
                    dates=dates,
                    forecast_periods=forecast_horizon,
                    order=order,
                    seasonal_order=seasonal_order
                )
            elif forecast_type == 'prophet' and PROPHET_AVAILABLE:
                result = self.prophet_forecast(
                    data=values,
                    dates=dates,
                    forecast_periods=forecast_horizon
                )
            elif forecast_type == 'seasonal_decompose':
                result = self.seasonal_decomposition(
                    data=values,
                    dates=dates,
                    model='additive',
                    period=parameters.get('period', 12) if parameters else 12
                )
            else:
                return {'error': f'Unsupported forecast type: {forecast_type}'}
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Forecast generation failed: {e}")
            return {'error': str(e)}

# Global forecasting engine instance
forecaster = ForecastingEngine()
forecasting_engine = forecaster  # Alias for compatibility
