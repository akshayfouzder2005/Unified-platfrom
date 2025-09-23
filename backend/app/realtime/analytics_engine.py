"""
Real-time Analytics Engine for Ocean-Bio Platform
Generates live insights, statistics, and monitoring data
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
import json

from app.core.database import get_db
from .websocket_manager import websocket_manager

logger = logging.getLogger(__name__)

class AnalyticsEngine:
    """
    Real-time analytics and monitoring engine
    Provides live statistics, insights, and system monitoring
    """
    
    def __init__(self):
        self.is_running = False
        self.update_interval = 30  # seconds
        self.analytics_cache = {}
        self.last_update = None
        
        # Metrics to track
        self.metrics = {
            'species_identification': {
                'total_identifications': 0,
                'success_rate': 0.0,
                'avg_processing_time': 0.0,
                'top_species': [],
                'identifications_per_hour': 0
            },
            'data_ingestion': {
                'total_records': 0,
                'records_today': 0,
                'error_rate': 0.0,
                'active_uploads': 0,
                'data_quality_score': 0.0
            },
            'fisheries_activity': {
                'active_vessels': 0,
                'total_catch_weight': 0.0,
                'catch_per_hour': 0.0,
                'top_fishing_areas': [],
                'quota_utilization': 0.0
            },
            'research_activity': {
                'active_studies': 0,
                'samples_processed': 0,
                'researchers_online': 0,
                'data_collaborations': 0
            },
            'system_health': {
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'database_connections': 0,
                'response_time': 0.0,
                'error_rate': 0.0
            }
        }
        
        self.is_initialized = False
        self.redis_url = None
    
    async def initialize(self, redis_url: Optional[str] = None):
        """Initialize the analytics engine"""
        try:
            logger.info("📈 Initializing analytics engine...")
            
            self.redis_url = redis_url
            
            if redis_url:
                logger.info("Redis URL provided for analytics persistence")
                # In a full implementation, we would connect to Redis here
            else:
                logger.info("Running analytics in memory-only mode")
            
            self.is_initialized = True
            logger.info("✅ Analytics engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize analytics engine: {e}")
            return False
    
    async def start(self):
        """Start the real-time analytics engine"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("Starting real-time analytics engine...")
        
        # Start background tasks
        asyncio.create_task(self._update_metrics_loop())
        asyncio.create_task(self._broadcast_analytics_loop())
        asyncio.create_task(self._monitor_system_health())
        
        logger.info("Real-time analytics engine started")
    
    async def stop(self):
        """Stop the analytics engine"""
        self.is_running = False
        logger.info("Real-time analytics engine stopped")
    
    async def _update_metrics_loop(self):
        """Background task to continuously update metrics"""
        while self.is_running:
            try:
                await self._update_all_metrics()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error updating metrics: {str(e)}")
                await asyncio.sleep(10)
    
    async def _broadcast_analytics_loop(self):
        """Background task to broadcast analytics to subscribers"""
        while self.is_running:
            try:
                if self.analytics_cache:
                    await websocket_manager.send_analytics_update(self.analytics_cache)
                await asyncio.sleep(15)  # Broadcast every 15 seconds
            except Exception as e:
                logger.error(f"Error broadcasting analytics: {str(e)}")
                await asyncio.sleep(10)
    
    async def _monitor_system_health(self):
        """Monitor system health metrics"""
        while self.is_running:
            try:
                await self._update_system_health_metrics()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error monitoring system health: {str(e)}")
                await asyncio.sleep(30)
    
    async def _update_all_metrics(self):
        """Update all analytics metrics"""
        try:
            # Get database session (in a real implementation, use dependency injection)
            db = next(get_db())
            
            # Update species identification metrics
            await self._update_species_metrics(db)
            
            # Update data ingestion metrics
            await self._update_ingestion_metrics(db)
            
            # Update fisheries metrics
            await self._update_fisheries_metrics(db)
            
            # Update research activity metrics
            await self._update_research_metrics(db)
            
            # Generate summary analytics
            self.analytics_cache = self._generate_analytics_summary()
            self.last_update = datetime.now()
            
            logger.debug("Updated all analytics metrics")
            
        except Exception as e:
            logger.error(f"Error updating analytics metrics: {str(e)}")
        finally:
            if 'db' in locals():
                db.close()
    
    async def _update_species_metrics(self, db: Session):
        """Update species identification metrics"""
        try:
            # Get total species records (demo data)
            total_count = 1250  # Demo value
            
            # Calculate success rate (demo)
            success_rate = 0.89  # Demo value
            
            # Average processing time (demo)
            avg_processing_time = 2.3  # Demo value in seconds
            
            # Top identified species (demo)
            top_species = [
                {"name": "Atlantic Cod", "scientific": "Gadus morhua", "count": 145},
                {"name": "Yellowfin Tuna", "scientific": "Thunnus albacares", "count": 123},
                {"name": "Red Snapper", "scientific": "Lutjanus campechanus", "count": 98},
                {"name": "Atlantic Salmon", "scientific": "Salmo salar", "count": 87},
                {"name": "Mackerel", "scientific": "Scomber scombrus", "count": 76}
            ]
            
            # Identifications per hour (demo)
            identifications_per_hour = 45
            
            self.metrics['species_identification'].update({
                'total_identifications': total_count,
                'success_rate': success_rate,
                'avg_processing_time': avg_processing_time,
                'top_species': top_species,
                'identifications_per_hour': identifications_per_hour
            })
            
        except Exception as e:
            logger.error(f"Error updating species metrics: {str(e)}")
    
    async def _update_ingestion_metrics(self, db: Session):
        """Update data ingestion metrics"""
        try:
            # Total records (demo)
            total_records = 45623
            
            # Records today (demo)
            records_today = 234
            
            # Error rate (demo)
            error_rate = 0.023
            
            # Active uploads (demo)
            active_uploads = 3
            
            # Data quality score (demo)
            data_quality_score = 0.94
            
            self.metrics['data_ingestion'].update({
                'total_records': total_records,
                'records_today': records_today,
                'error_rate': error_rate,
                'active_uploads': active_uploads,
                'data_quality_score': data_quality_score
            })
            
        except Exception as e:
            logger.error(f"Error updating ingestion metrics: {str(e)}")
    
    async def _update_fisheries_metrics(self, db: Session):
        """Update fisheries activity metrics"""
        try:
            # Active vessels (demo)
            active_vessels = 28
            
            # Total catch weight today (demo)
            total_catch_weight = 1456.7  # kg
            
            # Catch per hour (demo)
            catch_per_hour = 89.3  # kg/hour
            
            # Top fishing areas (demo)
            top_fishing_areas = [
                {"area": "North Atlantic - Zone 12", "catch_weight": 567.2},
                {"area": "Celtic Sea - Zone 7", "catch_weight": 423.8},
                {"area": "Bay of Biscay - Zone 8", "catch_weight": 345.1},
                {"area": "North Sea - Zone 4", "catch_weight": 289.6}
            ]
            
            # Quota utilization (demo)
            quota_utilization = 0.67
            
            self.metrics['fisheries_activity'].update({
                'active_vessels': active_vessels,
                'total_catch_weight': total_catch_weight,
                'catch_per_hour': catch_per_hour,
                'top_fishing_areas': top_fishing_areas,
                'quota_utilization': quota_utilization
            })
            
        except Exception as e:
            logger.error(f"Error updating fisheries metrics: {str(e)}")
    
    async def _update_research_metrics(self, db: Session):
        """Update research activity metrics"""
        try:
            # Active studies (demo)
            active_studies = 12
            
            # Samples processed today (demo)
            samples_processed = 67
            
            # Researchers online (demo)
            researchers_online = 8
            
            # Data collaborations (demo)
            data_collaborations = 4
            
            self.metrics['research_activity'].update({
                'active_studies': active_studies,
                'samples_processed': samples_processed,
                'researchers_online': researchers_online,
                'data_collaborations': data_collaborations
            })
            
        except Exception as e:
            logger.error(f"Error updating research metrics: {str(e)}")
    
    async def _update_system_health_metrics(self):
        """Update system health metrics"""
        try:
            import psutil
            
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Database connections (demo)
            database_connections = 15
            
            # Response time (demo)
            response_time = 0.23  # seconds
            
            # Error rate (demo)
            error_rate = 0.012
            
            self.metrics['system_health'].update({
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'database_connections': database_connections,
                'response_time': response_time,
                'error_rate': error_rate
            })
            
        except ImportError:
            # psutil not available, use demo values
            self.metrics['system_health'].update({
                'cpu_usage': 23.5,
                'memory_usage': 67.2,
                'database_connections': 15,
                'response_time': 0.23,
                'error_rate': 0.012
            })
        except Exception as e:
            logger.error(f"Error updating system health metrics: {str(e)}")
    
    def _generate_analytics_summary(self) -> Dict[str, Any]:
        """Generate comprehensive analytics summary"""
        return {
            'timestamp': datetime.now().isoformat(),
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'metrics': self.metrics,
            'summary': {
                'total_activity_score': self._calculate_activity_score(),
                'system_status': self._determine_system_status(),
                'key_insights': self._generate_key_insights(),
                'recommendations': self._generate_recommendations()
            },
            'real_time_stats': {
                'active_websocket_connections': len(websocket_manager.connections),
                'total_subscribers': sum(len(subs) for subs in websocket_manager.subscriptions.values()),
                'platform_uptime': websocket_manager.get_stats()['uptime']
            }
        }
    
    def _calculate_activity_score(self) -> float:
        """Calculate overall platform activity score (0-100)"""
        try:
            # Weighted activity score based on different metrics
            species_score = min(100, self.metrics['species_identification']['identifications_per_hour'] * 2)
            ingestion_score = min(100, self.metrics['data_ingestion']['records_today'] / 5)
            fisheries_score = min(100, self.metrics['fisheries_activity']['active_vessels'] * 3)
            research_score = min(100, self.metrics['research_activity']['samples_processed'] * 1.5)
            
            # Weighted average
            total_score = (
                species_score * 0.3 +
                ingestion_score * 0.25 +
                fisheries_score * 0.25 +
                research_score * 0.2
            )
            
            return round(total_score, 1)
            
        except Exception as e:
            logger.error(f"Error calculating activity score: {str(e)}")
            return 50.0
    
    def _determine_system_status(self) -> str:
        """Determine overall system status"""
        try:
            health_metrics = self.metrics['system_health']
            
            if (health_metrics['cpu_usage'] > 80 or 
                health_metrics['memory_usage'] > 85 or 
                health_metrics['error_rate'] > 0.05):
                return 'warning'
            elif (health_metrics['cpu_usage'] > 90 or 
                  health_metrics['memory_usage'] > 95 or 
                  health_metrics['error_rate'] > 0.1):
                return 'critical'
            else:
                return 'healthy'
                
        except Exception as e:
            logger.error(f"Error determining system status: {str(e)}")
            return 'unknown'
    
    def _generate_key_insights(self) -> List[str]:
        """Generate key insights from current metrics"""
        insights = []
        
        try:
            # Species identification insights
            if self.metrics['species_identification']['success_rate'] > 0.9:
                insights.append("🎯 AI species identification performing excellently (>90% accuracy)")
            
            # Data quality insights
            if self.metrics['data_ingestion']['data_quality_score'] > 0.9:
                insights.append("✨ Data quality is exceptionally high")
            
            # Fisheries insights
            if self.metrics['fisheries_activity']['quota_utilization'] > 0.8:
                insights.append("⚠️ Fishing quotas nearing limits - monitor closely")
            
            # Research activity insights
            if self.metrics['research_activity']['active_studies'] > 10:
                insights.append("🔬 High research activity - excellent platform adoption")
            
            # System health insights
            if self.metrics['system_health']['response_time'] < 0.5:
                insights.append("⚡ System response times are optimal")
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            insights.append("📊 Analytics engine is monitoring platform performance")
        
        return insights
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        try:
            # Performance recommendations
            if self.metrics['system_health']['cpu_usage'] > 70:
                recommendations.append("Consider scaling compute resources")
            
            # Data quality recommendations
            if self.metrics['data_ingestion']['error_rate'] > 0.05:
                recommendations.append("Review data validation rules to reduce error rate")
            
            # Research optimization
            if self.metrics['research_activity']['researchers_online'] > 15:
                recommendations.append("Peak research activity - consider adding collaboration features")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
        
        return recommendations
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current analytics metrics"""
        return {
            'metrics': self.metrics,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'is_running': self.is_running
        }
    
    async def get_historical_data(self, metric_type: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical data for a specific metric (demo implementation)"""
        try:
            # In a real implementation, this would query historical data from database
            # For now, generate demo historical data
            historical_data = []
            current_time = datetime.now()
            
            for i in range(hours):
                timestamp = current_time - timedelta(hours=i)
                
                # Generate demo data based on metric type
                if metric_type == 'species_identification':
                    value = max(0, 45 + (i * 2) + (i % 3 * 10))
                elif metric_type == 'data_ingestion':
                    value = max(0, 20 + (i * 1.5) + (i % 4 * 5))
                elif metric_type == 'fisheries_activity':
                    value = max(0, 30 + (i * 1.2) + (i % 5 * 8))
                else:
                    value = max(0, 25 + (i * 1.8) + (i % 6 * 6))
                
                historical_data.append({
                    'timestamp': timestamp.isoformat(),
                    'value': round(value, 1)
                })
            
            return list(reversed(historical_data))
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return []
    
    async def trigger_alert(self, alert_type: str, message: str, severity: str = 'info'):
        """Trigger a system alert"""
        try:
            alert_data = {
                'alert_type': alert_type,
                'message': message,
                'severity': severity,
                'timestamp': datetime.now().isoformat(),
                'metrics_snapshot': self.metrics
            }
            
            # Send alert via WebSocket
            await websocket_manager.send_system_alert(alert_data, severity)
            
            logger.info(f"Alert triggered: {alert_type} - {message}")
            
        except Exception as e:
            logger.error(f"Error triggering alert: {str(e)}")
    
    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data"""
        try:
            return {
                'system_status': self._determine_system_status(),
                'activity_score': self._calculate_activity_score(),
                'metrics': self.metrics,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'is_initialized': self.is_initialized,
                'is_running': self.is_running
            }
        except Exception as e:
            logger.error(f"Error getting dashboard data: {str(e)}")
            return {'system_status': 'unknown', 'error': str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            return {
                'system_health': {
                    'status': self._determine_system_status(),
                    'cpu_usage': self.metrics['system_health']['cpu_usage'],
                    'memory_usage': self.metrics['system_health']['memory_usage'],
                    'response_time': self.metrics['system_health']['response_time'],
                    'error_rate': self.metrics['system_health']['error_rate']
                },
                'processing_performance': {
                    'species_identification_rate': self.metrics['species_identification']['success_rate'],
                    'avg_processing_time': self.metrics['species_identification']['avg_processing_time'],
                    'data_quality_score': self.metrics['data_ingestion']['data_quality_score']
                },
                'activity_metrics': {
                    'active_connections': len(websocket_manager.connections),
                    'identifications_per_hour': self.metrics['species_identification']['identifications_per_hour'],
                    'records_processed_today': self.metrics['data_ingestion']['records_today']
                }
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {'system_health': {'status': 'unknown'}, 'error': str(e)}

# Global analytics engine instance
analytics_engine = AnalyticsEngine()
