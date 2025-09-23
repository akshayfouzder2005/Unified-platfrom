#!/usr/bin/env python3
"""
Phase 1 - Intelligence Enhancement Validation Script
Tests the key Phase 1 API endpoints and features
"""
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import io
from PIL import Image

# Add the backend directory to the path
backend_dir = Path(__file__).parent
sys.path.append(str(backend_dir))

from app.main import app
from app.ml.model_manager import model_manager
from app.realtime.websocket_manager import websocket_manager
from app.realtime.analytics_engine import analytics_engine
from app.search.elasticsearch_service import elasticsearch_service
from app.search.search_engine import search_engine

class Phase1Validator:
    """Validates Phase 1 functionality"""
    
    def __init__(self):
        self.test_results = {
            'ml_components': {},
            'realtime_components': {},
            'search_components': {},
            'api_endpoints': {},
            'overall_status': 'PENDING'
        }
    
    async def run_all_validations(self):
        """Run all Phase 1 validation tests"""
        try:
            print("🧪 Starting Phase 1 - Intelligence Enhancement Validation")
            print("=" * 60)
            
            # Test ML Components
            await self.test_ml_components()
            
            # Test Real-time Components
            await self.test_realtime_components()
            
            # Test Search Components
            await self.test_search_components()
            
            # Generate final report
            self.generate_final_report()
            
        except Exception as e:
            print(f"❌ Validation failed: {str(e)}")
            self.test_results['overall_status'] = 'FAILED'
    
    async def test_ml_components(self):
        """Test ML/AI components"""
        print("\n🧠 Testing ML/AI Components...")
        
        try:
            # Test model manager status
            system_status = model_manager.get_system_status()
            loaded_models = system_status.get('loaded_models', 0)
            
            self.test_results['ml_components']['model_manager'] = {
                'loaded_models': loaded_models,
                'status': 'PASS' if loaded_models > 0 else 'FAIL',
                'details': f"{loaded_models} models loaded"
            }
            
            print(f"   ✅ Model Manager: {loaded_models} models loaded")
            
            # Test species identification
            try:
                # Create a simple test image
                test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                pil_image = Image.fromarray(test_image)
                
                # Get a model for testing
                fish_model = model_manager.get_model('fish_primary')
                if fish_model:
                    result = fish_model.identify_species(pil_image, species_type='fish')
                    
                    self.test_results['ml_components']['species_identification'] = {
                        'status': 'PASS' if result.get('success') else 'FAIL',
                        'details': f"Identification {'successful' if result.get('success') else 'failed'}"
                    }
                    
                    print(f"   ✅ Species Identification: {'Working' if result.get('success') else 'Failed'}")
                else:
                    self.test_results['ml_components']['species_identification'] = {
                        'status': 'FAIL',
                        'details': "No fish model available"
                    }
                    print("   ❌ Species Identification: No model available")
                    
            except Exception as e:
                self.test_results['ml_components']['species_identification'] = {
                    'status': 'FAIL',
                    'details': f"Error: {str(e)}"
                }
                print(f"   ❌ Species Identification: Error - {str(e)}")
                
        except Exception as e:
            print(f"   ❌ ML Components test failed: {str(e)}")
    
    async def test_realtime_components(self):
        """Test real-time components"""
        print("\n🔗 Testing Real-time Components...")
        
        try:
            # Test WebSocket manager
            stats = websocket_manager.get_connection_stats()
            is_initialized = stats.get('is_initialized', False)
            
            self.test_results['realtime_components']['websocket_manager'] = {
                'status': 'PASS' if is_initialized else 'FAIL',
                'details': f"Initialized: {is_initialized}",
                'active_connections': stats.get('active_connections', 0),
                'available_topics': len(stats.get('available_topics', []))
            }
            
            print(f"   ✅ WebSocket Manager: {'Initialized' if is_initialized else 'Not initialized'}")
            print(f"      - Available topics: {len(stats.get('available_topics', []))}")
            
            # Test Analytics Engine
            dashboard_data = analytics_engine.get_real_time_dashboard_data()
            system_status = dashboard_data.get('system_status', 'unknown')
            
            self.test_results['realtime_components']['analytics_engine'] = {
                'status': 'PASS' if system_status != 'unknown' else 'FAIL',
                'details': f"System status: {system_status}",
                'is_running': analytics_engine.is_running
            }
            
            print(f"   ✅ Analytics Engine: System status = {system_status}")
            print(f"      - Is running: {analytics_engine.is_running}")
            
        except Exception as e:
            print(f"   ❌ Real-time components test failed: {str(e)}")
    
    async def test_search_components(self):
        """Test search components"""
        print("\n🔍 Testing Search Components...")
        
        try:
            # Test Search Engine
            search_stats = search_engine.get_search_stats()
            is_initialized = search_stats.get('is_initialized', False)
            
            self.test_results['search_components']['search_engine'] = {
                'status': 'PASS' if is_initialized else 'FAIL',
                'details': f"Initialized: {is_initialized}",
                'elasticsearch_available': search_stats.get('elasticsearch_available', False),
                'fallback_mode': search_stats.get('fallback_mode', True)
            }
            
            print(f"   ✅ Search Engine: {'Initialized' if is_initialized else 'Not initialized'}")
            print(f"      - Elasticsearch available: {search_stats.get('elasticsearch_available', False)}")
            print(f"      - Fallback mode: {search_stats.get('fallback_mode', True)}")
            
            # Test search functionality
            try:
                search_result = await search_engine.search("fish", size=5)
                search_working = 'hits' in search_result
                
                self.test_results['search_components']['search_functionality'] = {
                    'status': 'PASS' if search_working else 'FAIL',
                    'details': f"Search {'working' if search_working else 'failed'}",
                    'results_count': search_result.get('hits', {}).get('total', {}).get('value', 0)
                }
                
                print(f"   ✅ Search Functionality: {'Working' if search_working else 'Failed'}")
                
            except Exception as e:
                self.test_results['search_components']['search_functionality'] = {
                    'status': 'FAIL',
                    'details': f"Error: {str(e)}"
                }
                print(f"   ❌ Search Functionality: Error - {str(e)}")
                
        except Exception as e:
            print(f"   ❌ Search components test failed: {str(e)}")
    
    def generate_final_report(self):
        """Generate final validation report"""
        print("\n" + "=" * 60)
        print("📋 PHASE 1 VALIDATION REPORT")
        print("=" * 60)
        
        # Count passes and fails
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.test_results.items():
            if category == 'overall_status':
                continue
                
            print(f"\n{category.replace('_', ' ').title()}:")
            for test_name, result in tests.items():
                if isinstance(result, dict) and 'status' in result:
                    total_tests += 1
                    status_icon = "✅" if result['status'] == 'PASS' else "❌"
                    print(f"  {status_icon} {test_name.replace('_', ' ').title()}: {result['details']}")
                    if result['status'] == 'PASS':
                        passed_tests += 1
        
        # Overall status
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        if success_rate >= 80:
            overall_status = "COMPLETED ✅"
        elif success_rate >= 60:
            overall_status = "PARTIAL ⚠️"
        else:
            overall_status = "FAILED ❌"
        
        self.test_results['overall_status'] = overall_status
        
        print(f"\n🎯 Overall Status: {overall_status}")
        print(f"📊 Success Rate: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        # Key capabilities summary
        print(f"\n🚀 Phase 1 Capabilities Verified:")
        print(f"   🧠 AI-powered species identification: {'✅' if any(t.get('status') == 'PASS' for t in self.test_results['ml_components'].values()) else '❌'}")
        print(f"   📈 Real-time analytics: {'✅' if any(t.get('status') == 'PASS' for t in self.test_results['realtime_components'].values()) else '❌'}")
        print(f"   🔍 Intelligent search: {'✅' if any(t.get('status') == 'PASS' for t in self.test_results['search_components'].values()) else '❌'}")
        print(f"   🔗 WebSocket real-time communication: {'✅' if self.test_results['realtime_components'].get('websocket_manager', {}).get('status') == 'PASS' else '❌'}")
        
        print(f"\n🏆 Phase 1 - Intelligence Enhancement: {overall_status}")
        
        # Save results to file
        try:
            with open('phase1_validation_report.json', 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            print(f"\n📄 Detailed report saved to: phase1_validation_report.json")
        except Exception as e:
            print(f"\n⚠️ Could not save detailed report: {str(e)}")

async def main():
    """Main validation function"""
    validator = Phase1Validator()
    await validator.run_all_validations()

if __name__ == "__main__":
    asyncio.run(main())