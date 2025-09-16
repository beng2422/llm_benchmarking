"""
Configuration management for the LLM Benchmarking Suite.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """Centralized configuration management."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'models.openai.default_model')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        base_config = self.get('models.openai', {})
        model_config = base_config.get('models', {}).get(model_name, {})
        
        # Merge with default params
        default_params = base_config.get('default_params', {})
        model_params = {**default_params, **model_config}
        
        return {
            'model_name': model_name,
            'params': model_params,
            'timeout': base_config.get('api_timeout', 30),
            'max_retries': base_config.get('max_retries', 3),
            'retry_delay': base_config.get('retry_delay', 1.0)
        }
    
    def get_benchmark_config(self, benchmark_name: str) -> Dict[str, Any]:
        """Get configuration for a specific benchmark."""
        return self.get(f'benchmarks.{benchmark_name}', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.get('evaluation', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get('logging', {})
    
    def is_benchmark_enabled(self, benchmark_name: str) -> bool:
        """Check if a benchmark is enabled."""
        return self.get(f'benchmarks.{benchmark_name}.enabled', True)
    
    def get_enabled_benchmarks(self) -> list:
        """Get list of enabled benchmarks."""
        benchmarks = self.get('benchmarks', {})
        return [name for name, config in benchmarks.items() 
                if config.get('enabled', True)]
    
    def update_config(self, key: str, value: Any):
        """Update configuration value."""
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def save_config(self, path: Optional[str] = None):
        """Save configuration to file."""
        save_path = path or self.config_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    def validate_config(self) -> list:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check required sections
        required_sections = ['models', 'benchmarks', 'evaluation', 'logging']
        for section in required_sections:
            if section not in self._config:
                issues.append(f"Missing required section: {section}")
        
        # Validate model configurations
        if 'models' in self._config:
            models_config = self._config['models']
            if 'openai' not in models_config:
                issues.append("OpenAI model configuration is required")
        
        # Validate benchmark configurations
        if 'benchmarks' in self._config:
            for benchmark_name, config in self._config['benchmarks'].items():
                if not isinstance(config, dict):
                    issues.append(f"Benchmark '{benchmark_name}' configuration must be a dictionary")
                    continue
                
                if 'data_path' not in config:
                    issues.append(f"Benchmark '{benchmark_name}' missing data_path")
        
        return issues


# Global config instance
config = ConfigManager()
