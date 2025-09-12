"""
Benchmark registry for managing available benchmarks.
"""

from typing import Dict, List, Type
from .base_benchmark import BaseBenchmark


class BenchmarkRegistry:
    """Registry for managing available benchmarks."""
    
    _benchmarks: Dict[str, Type[BaseBenchmark]] = {}
    
    @classmethod
    def register(cls, name: str, benchmark_class: Type[BaseBenchmark]):
        """Register a benchmark class."""
        cls._benchmarks[name] = benchmark_class
    
    @classmethod
    def get_benchmark(cls, name: str, **kwargs) -> BaseBenchmark:
        """Get a benchmark instance by name."""
        if name not in cls._benchmarks:
            raise ValueError(f"Benchmark '{name}' not found. Available: {list(cls._benchmarks.keys())}")
        
        benchmark_class = cls._benchmarks[name]
        return benchmark_class(**kwargs)
    
    @classmethod
    def list_benchmarks(cls) -> List[str]:
        """List all registered benchmark names."""
        return list(cls._benchmarks.keys())
    
    @classmethod
    def get_benchmark_info(cls, name: str) -> Dict[str, str]:
        """Get information about a benchmark."""
        if name not in cls._benchmarks:
            raise ValueError(f"Benchmark '{name}' not found")
        
        # Create a temporary instance to get info
        benchmark_class = cls._benchmarks[name]
        # We need the required kwargs, so we'll handle this differently
        return {"name": name, "class": benchmark_class.__name__}
