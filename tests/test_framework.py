"""
Tests for the core framework components.
"""

import unittest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from framework.config_manager import ConfigManager
from framework.logger import BenchmarkLogger
from framework.parallel_executor import ParallelExecutor, ProgressTracker
from framework.data_validator import DataValidator, DataManager
from framework.results_analyzer import ResultsAnalyzer


class TestConfigManager(unittest.TestCase):
    """Test configuration management."""
    
    def setUp(self):
        """Set up test configuration."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        
        # Create test config
        test_config = {
            'models': {
                'openai': {
                    'default_model': 'gpt-4o-mini',
                    'api_timeout': 30
                }
            },
            'benchmarks': {
                'test_benchmark': {
                    'enabled': True,
                    'data_path': 'test_data'
                }
            },
            'evaluation': {
                'parallel_benchmarks': True
            },
            'logging': {
                'level': 'INFO'
            }
        }
        
        with open(self.config_file, 'w') as f:
            import yaml
            yaml.dump(test_config, f)
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_config_loading(self):
        """Test configuration loading."""
        config = ConfigManager(self.config_file)
        
        self.assertEqual(config.get('models.openai.default_model'), 'gpt-4o-mini')
        self.assertEqual(config.get('benchmarks.test_benchmark.enabled'), True)
        self.assertEqual(config.get('nonexistent.key', 'default'), 'default')
    
    def test_model_config(self):
        """Test model configuration retrieval."""
        config = ConfigManager(self.config_file)
        model_config = config.get_model_config('gpt-4o-mini')
        
        self.assertIn('model_name', model_config)
        self.assertIn('params', model_config)
        self.assertEqual(model_config['model_name'], 'gpt-4o-mini')
    
    def test_benchmark_config(self):
        """Test benchmark configuration retrieval."""
        config = ConfigManager(self.config_file)
        benchmark_config = config.get_benchmark_config('test_benchmark')
        
        self.assertEqual(benchmark_config['data_path'], 'test_data')
    
    def test_enabled_benchmarks(self):
        """Test enabled benchmarks listing."""
        config = ConfigManager(self.config_file)
        enabled = config.get_enabled_benchmarks()
        
        self.assertIn('test_benchmark', enabled)
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = ConfigManager(self.config_file)
        issues = config.validate_config()
        
        self.assertEqual(len(issues), 0)


class TestBenchmarkLogger(unittest.TestCase):
    """Test logging functionality."""
    
    def setUp(self):
        """Set up test logger."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, 'test.log')
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_logger_creation(self):
        """Test logger creation."""
        logger = BenchmarkLogger('test_logger', log_file=self.log_file)
        
        self.assertEqual(logger.name, 'test_logger')
        self.assertIsNotNone(logger.logger)
    
    def test_logging_levels(self):
        """Test different logging levels."""
        logger = BenchmarkLogger('test_logger', log_file=self.log_file)
        
        # Test that logging methods don't raise exceptions
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
    
    def test_structured_logging(self):
        """Test structured logging with context."""
        logger = BenchmarkLogger('test_logger', log_file=self.log_file)
        
        logger.info("Test message", key1="value1", key2="value2")
        
        # Check that log file was created and contains the message
        self.assertTrue(os.path.exists(self.log_file))
        
        with open(self.log_file, 'r') as f:
            content = f.read()
            self.assertIn("Test message", content)
            self.assertIn("key1=value1", content)


class TestParallelExecutor(unittest.TestCase):
    """Test parallel execution functionality."""
    
    def setUp(self):
        """Set up test executor."""
        self.executor = ParallelExecutor(max_workers=2)
    
    def test_executor_creation(self):
        """Test executor creation."""
        self.assertEqual(self.executor.max_workers, 2)
        self.assertIsNotNone(self.executor.logger)
    
    def test_progress_tracker(self):
        """Test progress tracking."""
        with ProgressTracker(10, "Test progress") as progress:
            self.assertEqual(progress.total, 10)
            self.assertEqual(progress.current, 0)
            
            progress.update(5)
            self.assertEqual(progress.current, 5)
            
            stats = progress.get_stats()
            self.assertEqual(stats['current'], 5)
            self.assertEqual(stats['total'], 10)
            self.assertEqual(stats['progress'], 50.0)


class TestDataValidator(unittest.TestCase):
    """Test data validation functionality."""
    
    def setUp(self):
        """Set up test validator."""
        self.validator = DataValidator()
    
    def test_empathy_validation(self):
        """Test empathy benchmark data validation."""
        valid_data = [
            {
                'id': 1,
                'patient': 'I feel sad',
                'empathetic': 'I understand you feel sad',
                'non_empathetic': 'Just get over it'
            }
        ]
        
        result = self.validator.validate_benchmark_data('empathy', valid_data)
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['errors']), 0)
    
    def test_empathy_validation_errors(self):
        """Test empathy validation with errors."""
        invalid_data = [
            {
                'id': 1,
                'patient': 'I feel sad',
                # Missing required fields
            }
        ]
        
        result = self.validator.validate_benchmark_data('empathy', invalid_data)
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)
    
    def test_commonsense_validation(self):
        """Test commonsense benchmark data validation."""
        valid_data = [
            {
                'id': 1,
                'question': 'What is 2+2?',
                'options': {'A': '3', 'B': '4', 'C': '5'},
                'correct_answer': 'B'
            }
        ]
        
        result = self.validator.validate_benchmark_data('commonsense', valid_data)
        self.assertTrue(result['valid'])
    
    def test_code_generation_validation(self):
        """Test code generation benchmark data validation."""
        valid_data = [
            {
                'id': 1,
                'title': 'Test function',
                'description': 'Write a test function',
                'function_signature': 'def test():',
                'test_cases': [{'input': {}, 'expected_output': 'test'}],
                'language': 'python'
            }
        ]
        
        result = self.validator.validate_benchmark_data('code_generation', valid_data)
        self.assertTrue(result['valid'])


class TestDataManager(unittest.TestCase):
    """Test data management functionality."""
    
    def setUp(self):
        """Set up test data manager."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_file = os.path.join(self.temp_dir, 'test_data.json')
        
        # Create test data
        test_data = [
            {
                'id': 1,
                'patient': 'I feel sad',
                'empathetic': 'I understand you feel sad',
                'non_empathetic': 'Just get over it'
            }
        ]
        
        with open(self.data_file, 'w') as f:
            json.dump(test_data, f)
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_data_loading(self):
        """Test data loading functionality."""
        manager = DataManager()
        
        data, metadata = manager.load_benchmark_data('empathy', self.data_file)
        
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['id'], 1)
        self.assertIn('benchmark_name', metadata)
        self.assertEqual(metadata['benchmark_name'], 'empathy')
    
    def test_cache_functionality(self):
        """Test data caching functionality."""
        manager = DataManager()
        
        # Load data first time
        data1, metadata1 = manager.load_benchmark_data('empathy', self.data_file)
        
        # Load data second time (should use cache)
        data2, metadata2 = manager.load_benchmark_data('empathy', self.data_file)
        
        self.assertEqual(data1, data2)
        self.assertEqual(metadata1['benchmark_name'], metadata2['benchmark_name'])


class TestResultsAnalyzer(unittest.TestCase):
    """Test results analysis functionality."""
    
    def setUp(self):
        """Set up test analyzer."""
        self.temp_dir = tempfile.mkdtemp()
        self.results_dir = os.path.join(self.temp_dir, 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create test results
        test_results = {
            'benchmark': 'empathy',
            'model': 'gpt-4o-mini',
            'timestamp': '2024-01-01T00:00:00',
            'metrics': {
                'accuracy': 0.85,
                'f1_score': 0.82,
                'precision': 0.80,
                'recall': 0.84
            },
            'num_examples': 100,
            'duration': 30.5
        }
        
        with open(os.path.join(self.results_dir, 'test_result.json'), 'w') as f:
            json.dump(test_results, f)
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_analyzer_creation(self):
        """Test analyzer creation."""
        analyzer = ResultsAnalyzer(self.results_dir)
        
        self.assertEqual(str(analyzer.results_dir), self.results_dir)
        self.assertIsNotNone(analyzer.logger)
    
    def test_results_loading(self):
        """Test results loading."""
        analyzer = ResultsAnalyzer(self.results_dir)
        results = analyzer._load_all_results()
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['benchmark'], 'empathy')
    
    def test_dataframe_conversion(self):
        """Test conversion to DataFrame."""
        analyzer = ResultsAnalyzer(self.results_dir)
        results = analyzer._load_all_results()
        df = analyzer._results_to_dataframe(results)
        
        self.assertEqual(len(df), 1)
        self.assertIn('benchmark', df.columns)
        self.assertIn('model', df.columns)
        self.assertIn('accuracy', df.columns)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_analysis_generation(self, mock_close, mock_savefig):
        """Test analysis generation."""
        analyzer = ResultsAnalyzer(self.results_dir)
        analysis = analyzer.analyze_all_results()
        
        self.assertIn('overview', analysis)
        self.assertIn('model_comparison', analysis)
        self.assertIn('benchmark_comparison', analysis)
        
        # Check that visualizations were generated
        self.assertTrue(mock_savefig.called)


if __name__ == '__main__':
    unittest.main()
