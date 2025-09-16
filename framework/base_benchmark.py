"""
Base benchmark class that all benchmarks must inherit from.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
import json
import os
import time
from .logger import logger
from .config_manager import config


class BaseBenchmark(ABC):
    """
    Abstract base class for all benchmarks.
    
    Each benchmark must implement:
    - load_data(): Load the benchmark dataset
    - evaluate_model(): Run a model on the benchmark
    - calculate_metrics(): Compute benchmark-specific metrics
    """
    
    def __init__(self, name: str, description: str, data_path: str):
        self.name = name
        self.description = description
        self.data_path = data_path
        self._data = None
        self.logger = logger
        self.config = config.get_benchmark_config(name)
    
    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load the benchmark dataset.
        
        Returns:
            List of examples, each containing the necessary fields for evaluation
        """
        pass
    
    @abstractmethod
    def evaluate_model(self, model_name: str, model_func: callable, **kwargs) -> List[Dict[str, Any]]:
        """
        Run a model on the benchmark.
        
        Args:
            model_name: Name of the model being evaluated
            model_func: Function that takes a prompt and returns a response
            **kwargs: Additional model-specific parameters
            
        Returns:
            List of results for each example
        """
        pass
    
    @abstractmethod
    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate benchmark-specific metrics.
        
        Args:
            results: Results from evaluate_model()
            
        Returns:
            Dictionary of metric names to values
        """
        pass
    
    def get_data(self) -> List[Dict[str, Any]]:
        """Get cached data or load it if not already loaded."""
        if self._data is None:
            self._data = self.load_data()
            
            # Apply max_examples limit if configured
            max_examples = self.config.get('max_examples')
            if max_examples and len(self._data) > max_examples:
                self.logger.info(f"Limiting {self.name} to {max_examples} examples", 
                               total_examples=len(self._data))
                self._data = self._data[:max_examples]
        
        return self._data
    
    def evaluate_single_example(self, example: Dict[str, Any], model_func: Callable, **kwargs) -> Dict[str, Any]:
        """
        Evaluate a single example. This method can be overridden for custom single-example evaluation.
        By default, it calls the full evaluate_model method with a single example.
        
        Args:
            example: Single example to evaluate
            model_func: Function that takes a prompt and returns a response
            **kwargs: Additional model parameters
            
        Returns:
            Evaluation result for the single example
        """
        # Default implementation - subclasses should override this for better performance
        results = self.evaluate_model(example.get('model_name', 'unknown'), model_func, **kwargs)
        return results[0] if results else {}
    
    def get_info(self) -> Dict[str, str]:
        """Get benchmark information."""
        return {
            "name": self.name,
            "description": self.description,
            "data_path": self.data_path,
            "num_examples": len(self.get_data())
        }
