#!/usr/bin/env python3
"""
Phase 1 - Intelligence Enhancement Initialization Script
Initializes AI/ML infrastructure, real-time analytics, and Elasticsearch search
"""
import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))

from app.ml.model_manager import model_manager
from app.realtime.websocket_manager import websocket_manager
from app.realtime.analytics_engine import analytics_engine
from app.search.elasticsearch_service import elasticsearch_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Phase1Initializer:
    """
    Phase 1 initialization manager for Intelligence Enhancement features
    """
    
    def __init__(self):
        self.components = {
            'ml_models': False,
            'websocket_manager': False,
            'analytics_engine': False,
            'elasticsearch': False
        }
        self.services_started = []
    
    async def initialize_all(self):
        """Initialize all Phase 1 components"""
        logger.info("🚀 Starting Phase 1 - Intelligence Enhancement initialization...")
        
        try:
            # 1. Initialize ML/AI infrastructure
            await self.initialize_ml_infrastructure()
            
            # 2. Initialize WebSocket manager
            await self.initialize_websocket_manager()
            
            # 3. Initialize analytics engine
            await self.initialize_analytics_engine()
            
            # 4. Initialize Elasticsearch service
            await self.initialize_elasticsearch()
            
            # 5. Verify all components
            await self.verify_components()
            
            # 6. Generate initialization report
            self.generate_report()
            
        except Exception as e:
            logger.error(f"❌ Phase 1 initialization failed: {str(e)}")
            await self.cleanup_on_failure()
            raise
    
    async def initialize_ml_infrastructure(self):
        """Initialize ML/AI model infrastructure"""
        logger.info("🧠 Initializing ML/AI infrastructure...")
        
        try:
            # Load all 5 data model types
            default_models = [
                {
                    'model_id': 'edna_primary',
                    'model_type': 'edna',
                    'description': 'eDNA analysis and species detection model'
                },
                {
                    'model_id': 'oceanographic_primary',
                    'model_type': 'oceanographic',
                    'description': 'Oceanographic data analysis model'
                },
                {
                    'model_id': 'otolith_primary', 
                    'model_type': 'otolith',
                    'description': 'Otolith morphology analysis model'
                },
                {
                    'model_id': 'taxonomy_primary',
                    'model_type': 'taxonomy',
                    'description': 'Taxonomic classification model'
                },
                {
                    'model_id': 'fisheries_primary',
                    'model_type': 'fisheries',
                    'description': 'Fisheries data analysis model'
                }
            ]
            
            for model_config in default_models:
                try:
                    success = model_manager.load_model_sync(
                        model_config['model_id'],
                        model_config['model_type']
                    )
                    
                    if success:
                        logger.info(f"✅ Loaded model: {model_config['model_id']}")
                    else:
                        logger.warning(f"⚠️ Could not load model: {model_config['model_id']} (demo models not available)")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Model loading error for {model_config['model_id']}: {str(e)}")
            
            # Get system status
            system_status = model_manager.get_system_status()
            logger.info(f"📊 ML System Status: {system_status}")
            
            self.components['ml_models'] = True
            self.services_started.append('ML/AI Infrastructure')
            logger.info("✅ ML/AI infrastructure initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ML infrastructure: {str(e)}")
            raise
    
    async def initialize_websocket_manager(self):
        """Initialize WebSocket manager for real-time communication"""
        logger.info("🔗 Initializing WebSocket manager...")
        
        try:
            # Initialize WebSocket manager (Redis URL optional for development)
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            
            try:
                await websocket_manager.initialize(redis_url)
                logger.info("✅ WebSocket manager initialized with Redis pub/sub")
            except Exception as redis_error:
                logger.warning(f"⚠️ Redis not available, using local WebSocket only: {redis_error}")
                await websocket_manager.initialize()
            
            # Test WebSocket functionality
            stats = websocket_manager.get_connection_stats()
            logger.info(f"📊 WebSocket Stats: {stats}")
            
            self.components['websocket_manager'] = True
            self.services_started.append('WebSocket Manager')
            logger.info("✅ WebSocket manager initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize WebSocket manager: {str(e)}")
            raise
    
    async def initialize_analytics_engine(self):
        """Initialize real-time analytics engine"""
        logger.info("📈 Initializing analytics engine...")
        
        try:
            # Initialize analytics engine
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            
            try:
                await analytics_engine.initialize(redis_url)
                logger.info("✅ Analytics engine initialized with Redis persistence")
            except Exception as redis_error:
                logger.warning(f"⚠️ Redis not available for analytics, using memory only: {redis_error}")
                await analytics_engine.initialize()
            
            # Start the analytics engine
            await analytics_engine.start()
            
            # Test analytics functionality
            dashboard_data = analytics_engine.get_real_time_dashboard_data()
            logger.info(f"📊 Analytics Dashboard Preview: System Status = {dashboard_data.get('system_status')}")
            
            self.components['analytics_engine'] = True
            self.services_started.append('Analytics Engine')
            logger.info("✅ Analytics engine initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize analytics engine: {str(e)}")
            raise
    
    async def initialize_elasticsearch(self):
        """Initialize Elasticsearch service for intelligent search"""
        logger.info("🔍 Initializing Elasticsearch service...")
        
        try:
            # Get Elasticsearch URL from environment or use default
            elasticsearch_url = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')
            
            # Initialize Elasticsearch service
            elasticsearch_service.elasticsearch_url = elasticsearch_url
            
            success = await elasticsearch_service.initialize()
            
            if success:
                logger.info("✅ Elasticsearch service initialized successfully")
                
                # Get cluster health
                health = await elasticsearch_service.get_cluster_health()
                logger.info(f"📊 Elasticsearch Health: {health.get('cluster_status', 'unknown')}")
                
                # Test search functionality
                test_search = await elasticsearch_service.search("*", size=1)
                logger.info(f"🔍 Search Test: Found {test_search.get('hits', {}).get('total', {}).get('value', 0)} total documents")
                
                self.components['elasticsearch'] = True
                self.services_started.append('Elasticsearch Service')
                
            else:
                logger.warning("⚠️ Elasticsearch not available - search features will be limited")
                logger.info("💡 To enable full search functionality, ensure Elasticsearch is running on http://localhost:9200")
                
        except Exception as e:
            logger.warning(f"⚠️ Elasticsearch initialization failed: {str(e)}")
            logger.info("💡 Search functionality will be limited without Elasticsearch")
    
    async def verify_components(self):
        """Verify all components are working correctly"""
        logger.info("🔍 Verifying all components...")
        
        # Test ML components
        if self.components['ml_models']:
            loaded_models = model_manager.get_loaded_models()
            logger.info(f"🧠 ML Models: {len(loaded_models)} models loaded")
        
        # Test WebSocket
        if self.components['websocket_manager']:
            total_connections = websocket_manager.connection_manager.get_total_connections()
            logger.info(f"🔗 WebSocket: Ready for {total_connections} active connections")
        
        # Test Analytics
        if self.components['analytics_engine']:
            performance_metrics = analytics_engine.get_performance_metrics()
            logger.info(f"📈 Analytics: System health = {performance_metrics.get('system_health', {}).get('status', 'unknown')}")
        
        # Test Elasticsearch
        if self.components['elasticsearch']:
            if elasticsearch_service.is_connected:
                indices = list(elasticsearch_service.index_configs.keys())
                logger.info(f"🔍 Elasticsearch: {len(indices)} data types configured for search")
        
        logger.info("✅ Component verification completed")
    
    def generate_report(self):
        """Generate initialization report"""
        logger.info("\n" + "="*60)
        logger.info("🎉 PHASE 1 - INTELLIGENCE ENHANCEMENT INITIALIZED")
        logger.info("="*60)
        
        logger.info("📋 Services Started:")
        for service in self.services_started:
            logger.info(f"   ✅ {service}")
        
        logger.info(f"\n🔧 Component Status:")
        for component, status in self.components.items():
            status_icon = "✅" if status else "⚠️"
            logger.info(f"   {status_icon} {component.replace('_', ' ').title()}: {'ACTIVE' if status else 'INACTIVE'}")
        
        logger.info(f"\n🚀 Available Features:")
        logger.info("   🧠 AI-powered species identification")
        logger.info("   📈 Real-time analytics and monitoring")
        logger.info("   🔗 WebSocket connections for live updates")
        
        if self.components['elasticsearch']:
            logger.info("   🔍 Intelligent search with Elasticsearch")
        else:
            logger.info("   ⚠️ Search limited (Elasticsearch not available)")
        
        logger.info(f"\n📚 API Endpoints Available:")
        logger.info("   POST /api/ml/identify/species - Species identification")
        logger.info("   GET  /api/ml/models/status - ML model status") 
        logger.info("   WebSocket /ws/{room} - Real-time connections")
        
        if self.components['elasticsearch']:
            logger.info("   POST /api/search/ - Intelligent search")
            logger.info("   GET  /api/search/suggest/{query} - Search suggestions")
        
        logger.info(f"\n🎯 Next Steps:")
        logger.info("   1. Start the FastAPI server")
        logger.info("   2. Test API endpoints")
        logger.info("   3. Monitor real-time analytics")
        logger.info("   4. Begin data ingestion and indexing")
        
        logger.info("="*60)
    
    async def cleanup_on_failure(self):
        """Clean up resources on initialization failure"""
        logger.info("🧹 Cleaning up resources due to initialization failure...")
        
        try:
            if analytics_engine.is_running:
                await analytics_engine.stop()
            
            if elasticsearch_service.is_connected:
                await elasticsearch_service.close()
                
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

async def main():
    """Main initialization function"""
    initializer = Phase1Initializer()
    
    try:
        await initializer.initialize_all()
        logger.info("🎉 Phase 1 initialization completed successfully!")
        return True
        
    except KeyboardInterrupt:
        logger.info("🛑 Initialization interrupted by user")
        return False
    except Exception as e:
        logger.error(f"💥 Initialization failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)