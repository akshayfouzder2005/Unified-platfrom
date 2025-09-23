#!/usr/bin/env python3
"""
🧪 Ocean-Bio Phase 2 Test Runner

Comprehensive test execution script for all Phase 2 components including:
- Geospatial analysis
- Predictive modeling  
- Genomics and eDNA analysis

Features:
- Parallel test execution
- Coverage reporting
- Performance benchmarking
- Test result aggregation
- CI/CD integration support

Author: Ocean-Bio Development Team
Version: 2.0.0
"""

import os
import sys
import subprocess
import argparse
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Add parent directory to path for imports
current_dir = Path(__file__).parent
backend_dir = current_dir.parent
sys.path.insert(0, str(backend_dir))

class TestRunner:
    """Comprehensive test runner for Ocean-Bio Phase 2"""
    
    def __init__(self, verbose: bool = False):
        """Initialize test runner"""
        self.verbose = verbose
        self.test_dir = current_dir
        self.backend_dir = backend_dir
        self.results = {}
        self.start_time = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_environment(self) -> bool:
        """Setup test environment and dependencies"""
        self.logger.info("🔧 Setting up test environment...")
        
        # Check if virtual environment is active
        if not hasattr(sys, 'real_prefix') and not (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        ):
            self.logger.warning("⚠️  No virtual environment detected. Consider using one.")
        
        # Install test dependencies if requirements file exists
        requirements_file = self.test_dir / "requirements-test.txt"
        if requirements_file.exists():
            self.logger.info("📦 Installing test dependencies...")
            try:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
                ], check=True, capture_output=True)
                self.logger.info("✅ Test dependencies installed successfully")
                return True
            except subprocess.CalledProcessError as e:
                self.logger.error(f"❌ Failed to install test dependencies: {e}")
                return False
        
        return True
    
    def run_geospatial_tests(self) -> Dict[str, Any]:
        """Run geospatial analysis tests"""
        self.logger.info("🗺️  Running geospatial analysis tests...")
        
        cmd = [
            sys.executable, '-m', 'pytest', 
            str(self.test_dir / 'test_geospatial.py'),
            '-v', '--tb=short',
            '--cov=backend.app.geospatial',
            '--cov-report=term-missing',
            '--cov-report=json:coverage_geospatial.json',
            '--junit-xml=results_geospatial.xml',
            '-p', 'no:warnings'
        ]
        
        return self._execute_test_command(cmd, "geospatial")
    
    def run_predictive_tests(self) -> Dict[str, Any]:
        """Run predictive modeling tests"""
        self.logger.info("📈 Running predictive modeling tests...")
        
        cmd = [
            sys.executable, '-m', 'pytest',
            str(self.test_dir / 'test_predictive.py'),
            '-v', '--tb=short',
            '--cov=backend.app.predictive',
            '--cov-report=term-missing',
            '--cov-report=json:coverage_predictive.json',
            '--junit-xml=results_predictive.xml',
            '-p', 'no:warnings'
        ]
        
        return self._execute_test_command(cmd, "predictive")
    
    def run_genomics_tests(self) -> Dict[str, Any]:
        """Run genomics and eDNA analysis tests"""
        self.logger.info("🧬 Running genomics and eDNA analysis tests...")
        
        cmd = [
            sys.executable, '-m', 'pytest',
            str(self.test_dir / 'test_genomics.py'),
            '-v', '--tb=short',
            '--cov=backend.app.genomics',
            '--cov-report=term-missing',
            '--cov-report=json:coverage_genomics.json',
            '--junit-xml=results_genomics.xml',
            '-p', 'no:warnings'
        ]
        
        return self._execute_test_command(cmd, "genomics")
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests across all components"""
        self.logger.info("🔗 Running integration tests...")
        
        cmd = [
            sys.executable, '-m', 'pytest',
            str(self.test_dir),
            '-v', '--tb=short',
            '-m', 'integration',
            '--cov=backend.app',
            '--cov-report=term-missing',
            '--cov-report=json:coverage_integration.json',
            '--junit-xml=results_integration.xml',
            '-p', 'no:warnings'
        ]
        
        return self._execute_test_command(cmd, "integration")
    
    def run_all_tests(self, parallel: bool = True) -> Dict[str, Any]:
        """Run all tests with optional parallelization"""
        self.logger.info("🚀 Running all Phase 2 tests...")
        
        cmd = [
            sys.executable, '-m', 'pytest',
            str(self.test_dir),
            '-v', '--tb=short',
            '--cov=backend.app',
            '--cov-report=term-missing',
            '--cov-report=html:htmlcov',
            '--cov-report=json:coverage_all.json',
            '--cov-report=xml:coverage.xml',
            '--junit-xml=results_all.xml',
            '--cov-fail-under=80',
            '-p', 'no:warnings'
        ]
        
        if parallel:
            # Add parallel execution with auto-detection of CPU cores
            cmd.extend(['-n', 'auto'])
            
        return self._execute_test_command(cmd, "all")
    
    def _execute_test_command(self, cmd: List[str], test_type: str) -> Dict[str, Any]:
        """Execute a test command and collect results"""
        start_time = time.time()
        
        try:
            # Change to backend directory for proper module discovery
            original_cwd = os.getcwd()
            os.chdir(self.backend_dir)
            
            if self.verbose:
                self.logger.info(f"Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            os.chdir(original_cwd)
            
            execution_time = time.time() - start_time
            
            # Parse test results
            test_results = {
                'test_type': test_type,
                'execution_time': execution_time,
                'return_code': result.returncode,
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'timestamp': datetime.now().isoformat()
            }
            
            # Extract key metrics from pytest output
            test_results.update(self._parse_pytest_output(result.stdout))
            
            if test_results['success']:
                self.logger.info(f"✅ {test_type} tests completed successfully in {execution_time:.2f}s")
            else:
                self.logger.error(f"❌ {test_type} tests failed (return code: {result.returncode})")
                if self.verbose:
                    self.logger.error(f"STDERR: {result.stderr}")
            
            return test_results
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"⏰ {test_type} tests timed out after 30 minutes")
            return {
                'test_type': test_type,
                'execution_time': 1800,
                'return_code': -1,
                'success': False,
                'error': 'Test execution timed out',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"💥 Error running {test_type} tests: {e}")
            return {
                'test_type': test_type,
                'execution_time': time.time() - start_time,
                'return_code': -2,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _parse_pytest_output(self, output: str) -> Dict[str, Any]:
        """Parse pytest output to extract key metrics"""
        metrics = {}
        
        lines = output.split('\n')
        for line in lines:
            # Parse test counts
            if 'passed' in line or 'failed' in line or 'error' in line:
                if '::' not in line and ('passed' in line or 'failed' in line):
                    # Summary line like "5 passed, 2 failed, 1 error"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.isdigit() and i + 1 < len(parts):
                            next_word = parts[i + 1].rstrip(',')
                            if next_word in ['passed', 'failed', 'error', 'skipped']:
                                metrics[f'{next_word}_count'] = int(part)
            
            # Parse coverage percentage
            if 'TOTAL' in line and '%' in line:
                try:
                    coverage_match = line.split()[-1]
                    if '%' in coverage_match:
                        metrics['coverage_percentage'] = float(coverage_match.rstrip('%'))
                except:
                    pass
            
            # Parse execution time
            if 'seconds' in line and ('passed' in line or 'failed' in line):
                try:
                    # Look for pattern like "5 passed in 12.34s"
                    if 'in ' in line and 's' in line:
                        time_part = line.split('in ')[-1].split()[0].rstrip('s')
                        if time_part.replace('.', '').isdigit():
                            metrics['pytest_execution_time'] = float(time_part)
                except:
                    pass
        
        return metrics
    
    def generate_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive test report"""
        self.logger.info("📊 Generating test report...")
        
        # Create report directory
        report_dir = self.test_dir / 'reports'
        report_dir.mkdir(exist_ok=True)
        
        # Generate JSON report
        json_report = {
            'test_run': {
                'timestamp': datetime.now().isoformat(),
                'total_execution_time': time.time() - self.start_time if self.start_time else 0,
                'platform': sys.platform,
                'python_version': sys.version
            },
            'results': results,
            'summary': self._generate_summary(results)
        }
        
        json_report_file = report_dir / f'test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(json_report_file, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        # Generate HTML report (simple)
        html_report = self._generate_html_report(json_report)
        html_report_file = report_dir / f'test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        with open(html_report_file, 'w') as f:
            f.write(html_report)
        
        self.logger.info(f"📁 Reports saved to {report_dir}")
        self.logger.info(f"📄 JSON Report: {json_report_file}")
        self.logger.info(f"🌐 HTML Report: {html_report_file}")
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from test results"""
        summary = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'error_tests': 0,
            'skipped_tests': 0,
            'success_rate': 0.0,
            'average_coverage': 0.0,
            'total_execution_time': 0.0,
            'successful_test_suites': 0,
            'failed_test_suites': 0
        }
        
        coverage_values = []
        
        for result in results.values():
            if isinstance(result, dict):
                # Aggregate test counts
                summary['total_tests'] += result.get('passed_count', 0) + result.get('failed_count', 0) + result.get('error_count', 0)
                summary['passed_tests'] += result.get('passed_count', 0)
                summary['failed_tests'] += result.get('failed_count', 0)
                summary['error_tests'] += result.get('error_count', 0)
                summary['skipped_tests'] += result.get('skipped_count', 0)
                
                # Aggregate execution time
                summary['total_execution_time'] += result.get('execution_time', 0)
                
                # Track coverage
                if 'coverage_percentage' in result:
                    coverage_values.append(result['coverage_percentage'])
                
                # Track suite success
                if result.get('success', False):
                    summary['successful_test_suites'] += 1
                else:
                    summary['failed_test_suites'] += 1
        
        # Calculate rates
        if summary['total_tests'] > 0:
            summary['success_rate'] = (summary['passed_tests'] / summary['total_tests']) * 100
        
        if coverage_values:
            summary['average_coverage'] = sum(coverage_values) / len(coverage_values)
        
        return summary
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate simple HTML report"""
        summary = report_data['summary']
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Ocean-Bio Phase 2 Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: linear-gradient(45deg, #2196F3, #21CBF3); color: white; padding: 20px; border-radius: 10px; }}
                .summary {{ background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 5px; }}
                .success {{ color: #4CAF50; }}
                .error {{ color: #f44336; }}
                .warning {{ color: #ff9800; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌊 Ocean-Bio Phase 2 Test Report</h1>
                <p>Generated on {report_data['test_run']['timestamp']}</p>
            </div>
            
            <div class="summary">
                <h2>📊 Summary</h2>
                <div class="metric"><strong>Total Tests:</strong> {summary['total_tests']}</div>
                <div class="metric success"><strong>Passed:</strong> {summary['passed_tests']}</div>
                <div class="metric error"><strong>Failed:</strong> {summary['failed_tests']}</div>
                <div class="metric warning"><strong>Errors:</strong> {summary['error_tests']}</div>
                <div class="metric"><strong>Success Rate:</strong> {summary['success_rate']:.1f}%</div>
                <div class="metric"><strong>Average Coverage:</strong> {summary['average_coverage']:.1f}%</div>
                <div class="metric"><strong>Total Time:</strong> {summary['total_execution_time']:.2f}s</div>
            </div>
            
            <h2>🧪 Test Suite Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Test Suite</th>
                        <th>Status</th>
                        <th>Passed</th>
                        <th>Failed</th>
                        <th>Coverage</th>
                        <th>Time (s)</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for test_type, result in report_data['results'].items():
            if isinstance(result, dict):
                status = "✅ Success" if result.get('success', False) else "❌ Failed"
                status_class = "success" if result.get('success', False) else "error"
                
                html += f"""
                    <tr>
                        <td>{test_type.title()}</td>
                        <td class="{status_class}">{status}</td>
                        <td>{result.get('passed_count', 'N/A')}</td>
                        <td>{result.get('failed_count', 'N/A')}</td>
                        <td>{result.get('coverage_percentage', 'N/A')}%</td>
                        <td>{result.get('execution_time', 'N/A'):.2f}</td>
                    </tr>
                """
        
        html += """
                </tbody>
            </table>
            
            <div style="margin-top: 40px; padding: 20px; background: #e3f2fd; border-radius: 5px;">
                <h3>🏆 Phase 2 Testing Complete!</h3>
                <p>This report covers comprehensive testing of all Phase 2 features:</p>
                <ul>
                    <li><strong>Geospatial Analysis:</strong> GIS integration, spatial analysis, mapping</li>
                    <li><strong>Predictive Modeling:</strong> Stock assessment, forecasting, ML models</li>
                    <li><strong>Genomics & eDNA:</strong> Sequence processing, taxonomic classification, diversity analysis</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def cleanup(self) -> None:
        """Cleanup temporary files and artifacts"""
        self.logger.info("🧹 Cleaning up temporary files...")
        
        # Clean up coverage files
        coverage_files = [
            'coverage_geospatial.json',
            'coverage_predictive.json', 
            'coverage_genomics.json',
            'coverage_integration.json',
            'coverage_all.json',
            'coverage.xml',
            '.coverage'
        ]
        
        for file in coverage_files:
            file_path = self.backend_dir / file
            if file_path.exists():
                try:
                    file_path.unlink()
                except:
                    pass
        
        # Clean up pytest cache
        pytest_cache = self.backend_dir / '.pytest_cache'
        if pytest_cache.exists():
            try:
                import shutil
                shutil.rmtree(pytest_cache)
            except:
                pass


def main():
    """Main entry point for test runner"""
    parser = argparse.ArgumentParser(description="Ocean-Bio Phase 2 Test Runner")
    
    parser.add_argument(
        '--component', 
        choices=['geospatial', 'predictive', 'genomics', 'integration', 'all'],
        default='all',
        help='Component to test (default: all)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--no-setup',
        action='store_true',
        help='Skip environment setup'
    )
    
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel test execution'
    )
    
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Skip cleanup of temporary files'
    )
    
    parser.add_argument(
        '--report-only',
        action='store_true',
        help='Generate report from existing results without running tests'
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = TestRunner(verbose=args.verbose)
    runner.start_time = time.time()
    
    print("🌊 Ocean-Bio Phase 2 Test Runner")
    print("=" * 50)
    
    try:
        # Setup environment
        if not args.no_setup and not args.report_only:
            if not runner.setup_environment():
                print("❌ Environment setup failed")
                sys.exit(1)
        
        # Run tests based on component selection
        if not args.report_only:
            if args.component == 'geospatial':
                runner.results['geospatial'] = runner.run_geospatial_tests()
            elif args.component == 'predictive':
                runner.results['predictive'] = runner.run_predictive_tests()
            elif args.component == 'genomics':
                runner.results['genomics'] = runner.run_genomics_tests()
            elif args.component == 'integration':
                runner.results['integration'] = runner.run_integration_tests()
            elif args.component == 'all':
                runner.results['all'] = runner.run_all_tests(parallel=not args.no_parallel)
            
            # Generate report
            runner.generate_report(runner.results)
        
        # Cleanup
        if not args.no_cleanup:
            runner.cleanup()
        
        # Print final status
        total_time = time.time() - runner.start_time
        print(f"\n🏁 Test execution completed in {total_time:.2f} seconds")
        
        # Exit with appropriate code
        success = all(
            result.get('success', False) if isinstance(result, dict) else False 
            for result in runner.results.values()
        )
        
        if success:
            print("✅ All tests passed!")
            sys.exit(0)
        else:
            print("❌ Some tests failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()