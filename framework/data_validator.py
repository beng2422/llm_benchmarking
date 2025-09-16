"""
Data validation and management utilities for the LLM Benchmarking Suite.
"""

import json
import os
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from .logger import logger
from .config_manager import config


class DataValidator:
    """Validates benchmark data against schemas and requirements."""
    
    def __init__(self):
        self.logger = logger
        self.validation_config = config.get('data_validation', {})
        self.strict_mode = self.validation_config.get('strict_mode', False)
        self.validate_schemas = self.validation_config.get('validate_schemas', True)
        
        # Define schemas for each benchmark type
        self.schemas = {
            'empathy': {
                'required_fields': ['id', 'patient', 'empathetic', 'non_empathetic'],
                'field_types': {
                    'id': (int, str),
                    'patient': str,
                    'empathetic': str,
                    'non_empathetic': str
                },
                'optional_fields': ['category', 'difficulty']
            },
            'commonsense': {
                'required_fields': ['id', 'question', 'options', 'correct_answer'],
                'field_types': {
                    'id': (int, str),
                    'question': str,
                    'options': dict,
                    'correct_answer': str
                },
                'optional_fields': ['category', 'difficulty', 'explanation']
            },
            'math_reasoning': {
                'required_fields': ['id', 'question', 'answer', 'category', 'difficulty'],
                'field_types': {
                    'id': (int, str),
                    'question': str,
                    'answer': (int, float, str),
                    'category': str,
                    'difficulty': str
                },
                'optional_fields': ['explanation', 'steps']
            },
            'code_generation': {
                'required_fields': ['id', 'title', 'description', 'function_signature', 'test_cases', 'language'],
                'field_types': {
                    'id': (int, str),
                    'title': str,
                    'description': str,
                    'function_signature': str,
                    'test_cases': list,
                    'language': str
                },
                'optional_fields': ['difficulty', 'category', 'expected_code']
            }
        }
    
    def validate_benchmark_data(self, benchmark_name: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate benchmark data against its schema.
        
        Args:
            benchmark_name: Name of the benchmark
            data: List of data examples to validate
            
        Returns:
            Validation results with errors and warnings
        """
        if not self.validate_schemas:
            return {'valid': True, 'errors': [], 'warnings': []}
        
        if benchmark_name not in self.schemas:
            return {
                'valid': False,
                'errors': [f"No schema defined for benchmark: {benchmark_name}"],
                'warnings': []
            }
        
        schema = self.schemas[benchmark_name]
        errors = []
        warnings = []
        
        for i, example in enumerate(data):
            example_errors, example_warnings = self._validate_example(example, schema, i)
            errors.extend(example_errors)
            warnings.extend(example_warnings)
        
        is_valid = len(errors) == 0 or not self.strict_mode
        
        self.logger.info(f"Data validation for {benchmark_name}", 
                        valid=is_valid, errors=len(errors), warnings=len(warnings))
        
        return {
            'valid': is_valid,
            'errors': errors,
            'warnings': warnings,
            'total_examples': len(data)
        }
    
    def _validate_example(self, example: Dict[str, Any], schema: Dict[str, Any], index: int) -> Tuple[List[str], List[str]]:
        """Validate a single example against its schema."""
        errors = []
        warnings = []
        
        # Check required fields
        for field in schema['required_fields']:
            if field not in example:
                errors.append(f"Example {index}: Missing required field '{field}'")
            elif example[field] is None or example[field] == '':
                errors.append(f"Example {index}: Required field '{field}' is empty")
        
        # Check field types
        for field, expected_types in schema['field_types'].items():
            if field in example and example[field] is not None:
                if not isinstance(example[field], expected_types):
                    errors.append(f"Example {index}: Field '{field}' has wrong type. Expected {expected_types}, got {type(example[field])}")
        
        # Check for unexpected fields
        allowed_fields = set(schema['required_fields'] + schema['optional_fields'])
        for field in example.keys():
            if field not in allowed_fields:
                warnings.append(f"Example {index}: Unexpected field '{field}'")
        
        # Benchmark-specific validations
        if schema == self.schemas['empathy']:
            errors.extend(self._validate_empathy_example(example, index))
        elif schema == self.schemas['commonsense']:
            errors.extend(self._validate_commonsense_example(example, index))
        elif schema == self.schemas['math_reasoning']:
            errors.extend(self._validate_math_example(example, index))
        elif schema == self.schemas['code_generation']:
            errors.extend(self._validate_code_example(example, index))
        
        return errors, warnings
    
    def _validate_empathy_example(self, example: Dict[str, Any], index: int) -> List[str]:
        """Validate empathy benchmark example."""
        errors = []
        
        if 'empathetic' in example and 'non_empathetic' in example:
            if example['empathetic'] == example['non_empathetic']:
                errors.append(f"Example {index}: Empathetic and non-empathetic responses are identical")
        
        return errors
    
    def _validate_commonsense_example(self, example: Dict[str, Any], index: int) -> List[str]:
        """Validate commonsense benchmark example."""
        errors = []
        
        if 'options' in example and 'correct_answer' in example:
            if not isinstance(example['options'], dict):
                errors.append(f"Example {index}: Options must be a dictionary")
            elif example['correct_answer'] not in example['options']:
                errors.append(f"Example {index}: Correct answer '{example['correct_answer']}' not found in options")
        
        return errors
    
    def _validate_math_example(self, example: Dict[str, Any], index: int) -> List[str]:
        """Validate math reasoning benchmark example."""
        errors = []
        
        if 'difficulty' in example:
            valid_difficulties = ['easy', 'medium', 'hard']
            if example['difficulty'] not in valid_difficulties:
                errors.append(f"Example {index}: Invalid difficulty '{example['difficulty']}'. Must be one of {valid_difficulties}")
        
        return errors
    
    def _validate_code_example(self, example: Dict[str, Any], index: int) -> List[str]:
        """Validate code generation benchmark example."""
        errors = []
        
        if 'language' in example:
            valid_languages = ['python', 'javascript', 'java']
            if example['language'] not in valid_languages:
                errors.append(f"Example {index}: Invalid language '{example['language']}'. Must be one of {valid_languages}")
        
        if 'test_cases' in example:
            if not isinstance(example['test_cases'], list):
                errors.append(f"Example {index}: Test cases must be a list")
            else:
                for j, test_case in enumerate(example['test_cases']):
                    if not isinstance(test_case, dict):
                        errors.append(f"Example {index}, Test case {j}: Must be a dictionary")
                    elif 'input' not in test_case or 'expected_output' not in test_case:
                        errors.append(f"Example {index}, Test case {j}: Missing 'input' or 'expected_output'")
        
        return errors


class DataManager:
    """Manages data loading, versioning, and caching."""
    
    def __init__(self):
        self.logger = logger
        self.validator = DataValidator()
        self.cache_dir = Path("data_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def load_benchmark_data(self, benchmark_name: str, data_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Load and validate benchmark data.
        
        Args:
            benchmark_name: Name of the benchmark
            data_path: Path to the data file
            
        Returns:
            Tuple of (data, metadata)
        """
        # Check cache first
        cache_key = self._get_cache_key(benchmark_name, data_path)
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data:
            self.logger.debug(f"Loaded {benchmark_name} data from cache")
            return cached_data['data'], cached_data['metadata']
        
        # Load from file
        data = self._load_from_file(data_path)
        
        # Validate data
        validation_result = self.validator.validate_benchmark_data(benchmark_name, data)
        
        if not validation_result['valid'] and self.validator.strict_mode:
            raise ValueError(f"Data validation failed for {benchmark_name}: {validation_result['errors']}")
        
        # Create metadata
        metadata = {
            'benchmark_name': benchmark_name,
            'data_path': data_path,
            'loaded_at': datetime.now().isoformat(),
            'num_examples': len(data),
            'validation_result': validation_result,
            'file_hash': self._calculate_file_hash(data_path)
        }
        
        # Cache the data
        self._save_to_cache(cache_key, data, metadata)
        
        # Log validation results
        if validation_result['errors']:
            self.logger.warning(f"Data validation warnings for {benchmark_name}", 
                              errors=len(validation_result['errors']))
        if validation_result['warnings']:
            self.logger.info(f"Data validation warnings for {benchmark_name}", 
                           warnings=len(validation_result['warnings']))
        
        return data, metadata
    
    def _load_from_file(self, data_path: str) -> List[Dict[str, Any]]:
        """Load data from file."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            if data_path.endswith('.json'):
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
        
        if not isinstance(data, list):
            raise ValueError("Data must be a list of examples")
        
        return data
    
    def _get_cache_key(self, benchmark_name: str, data_path: str) -> str:
        """Generate cache key for data."""
        file_hash = self._calculate_file_hash(data_path)
        return f"{benchmark_name}_{file_hash}"
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()[:16]
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load data from cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, data: List[Dict[str, Any]], metadata: Dict[str, Any]):
        """Save data to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'data': data,
                    'metadata': metadata
                }, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")
    
    def clear_cache(self, benchmark_name: Optional[str] = None):
        """Clear cache for a specific benchmark or all benchmarks."""
        if benchmark_name:
            pattern = f"{benchmark_name}_*.json"
        else:
            pattern = "*.json"
        
        for cache_file in self.cache_dir.glob(pattern):
            cache_file.unlink()
        
        self.logger.info(f"Cleared cache for {benchmark_name or 'all benchmarks'}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data."""
        cache_info = {}
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                metadata = data.get('metadata', {})
                benchmark_name = metadata.get('benchmark_name', 'unknown')
                
                if benchmark_name not in cache_info:
                    cache_info[benchmark_name] = []
                
                cache_info[benchmark_name].append({
                    'file': cache_file.name,
                    'loaded_at': metadata.get('loaded_at'),
                    'num_examples': metadata.get('num_examples', 0),
                    'file_hash': metadata.get('file_hash', ''),
                    'size_mb': cache_file.stat().st_size / (1024 * 1024)
                })
            except Exception as e:
                self.logger.warning(f"Failed to read cache info for {cache_file}: {e}")
        
        return cache_info
