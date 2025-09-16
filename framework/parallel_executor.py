"""
Parallel execution utilities for the LLM Benchmarking Suite.
"""

import time
import asyncio
import concurrent.futures
from typing import List, Dict, Any, Callable, Optional, Tuple
from functools import partial
from tqdm import tqdm
from .logger import logger
from .config_manager import config


class ParallelExecutor:
    """Handles parallel execution of benchmarks and examples."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or config.get('evaluation.max_workers', 4)
        self.logger = logger
    
    def run_benchmarks_parallel(self, 
                               benchmark_configs: List[Dict[str, Any]], 
                               model_func: Callable,
                               **model_kwargs) -> Dict[str, Any]:
        """
        Run multiple benchmarks in parallel.
        
        Args:
            benchmark_configs: List of benchmark configurations
            model_func: Model function to call
            **model_kwargs: Additional model parameters
            
        Returns:
            Dictionary mapping benchmark names to results
        """
        self.logger.info(f"Running {len(benchmark_configs)} benchmarks in parallel", 
                        max_workers=self.max_workers)
        
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all benchmark tasks
            future_to_benchmark = {}
            for config in benchmark_configs:
                future = executor.submit(
                    self._run_single_benchmark,
                    config,
                    model_func,
                    **model_kwargs
                )
                future_to_benchmark[future] = config['name']
            
            # Collect results with progress bar
            with tqdm(total=len(benchmark_configs), desc="Running benchmarks") as pbar:
                for future in concurrent.futures.as_completed(future_to_benchmark):
                    benchmark_name = future_to_benchmark[future]
                    try:
                        result = future.result()
                        results[benchmark_name] = result
                        pbar.set_postfix({"completed": benchmark_name})
                    except Exception as e:
                        self.logger.error(f"Benchmark {benchmark_name} failed", error=str(e))
                        results[benchmark_name] = {"error": str(e)}
                    finally:
                        pbar.update(1)
        
        return results
    
    def run_examples_parallel(self, 
                             examples: List[Dict[str, Any]], 
                             model_func: Callable,
                             evaluation_func: Callable,
                             **model_kwargs) -> List[Dict[str, Any]]:
        """
        Run multiple examples in parallel.
        
        Args:
            examples: List of examples to evaluate
            model_func: Model function to call
            evaluation_func: Function to evaluate each example
            **model_kwargs: Additional model parameters
            
        Returns:
            List of evaluation results
        """
        self.logger.info(f"Running {len(examples)} examples in parallel", 
                        max_workers=self.max_workers)
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all example tasks
            future_to_example = {}
            for i, example in enumerate(examples):
                future = executor.submit(
                    self._run_single_example,
                    example,
                    model_func,
                    evaluation_func,
                    i,
                    **model_kwargs
                )
                future_to_example[future] = i
            
            # Collect results with progress bar
            with tqdm(total=len(examples), desc="Processing examples") as pbar:
                for future in concurrent.futures.as_completed(future_to_example):
                    example_idx = future_to_example[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Example {example_idx} failed", error=str(e))
                        results.append({
                            "id": examples[example_idx].get("id", example_idx),
                            "error": str(e),
                            "is_correct": False
                        })
                    finally:
                        pbar.update(1)
        
        # Sort results by original order
        results.sort(key=lambda x: x.get("id", 0))
        return results
    
    def _run_single_benchmark(self, 
                             config: Dict[str, Any], 
                             model_func: Callable,
                             **model_kwargs) -> Dict[str, Any]:
        """Run a single benchmark."""
        benchmark_name = config['name']
        benchmark_class = config['class']
        benchmark_instance = benchmark_class(**config.get('kwargs', {}))
        
        start_time = time.time()
        
        try:
            # Load data
            data = benchmark_instance.get_data()
            self.logger.debug(f"Loaded {len(data)} examples for {benchmark_name}")
            
            # Run evaluation
            if config.get('parallel_examples', False):
                results = self.run_examples_parallel(
                    data, model_func, benchmark_instance.evaluate_single_example, **model_kwargs
                )
            else:
                results = benchmark_instance.evaluate_model(
                    config['model_name'], model_func, **model_kwargs
                )
            
            # Calculate metrics
            metrics = benchmark_instance.calculate_metrics(results)
            
            duration = time.time() - start_time
            
            self.logger.log_benchmark_complete(benchmark_name, config['model_name'], metrics, duration)
            
            return {
                "benchmark": benchmark_name,
                "model": config['model_name'],
                "timestamp": time.time(),
                "metrics": metrics,
                "num_examples": len(results),
                "results": results,
                "duration": duration
            }
            
        except Exception as e:
            self.logger.log_error(e, f"benchmark {benchmark_name}")
            return {
                "benchmark": benchmark_name,
                "model": config['model_name'],
                "timestamp": time.time(),
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    def _run_single_example(self, 
                           example: Dict[str, Any], 
                           model_func: Callable,
                           evaluation_func: Callable,
                           example_idx: int,
                           **model_kwargs) -> Dict[str, Any]:
        """Run evaluation on a single example."""
        start_time = time.time()
        
        try:
            result = evaluation_func(example, model_func, **model_kwargs)
            duration = time.time() - start_time
            
            # Log model request
            self.logger.log_model_request(
                model_kwargs.get('model_name', 'unknown'),
                len(str(example)),
                len(str(result.get('model_response', ''))),
                duration,
                success=True
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.log_model_request(
                model_kwargs.get('model_name', 'unknown'),
                len(str(example)),
                0,
                duration,
                success=False
            )
            raise e


class ProgressTracker:
    """Tracks and displays progress for long-running operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self.pbar = None
        self.logger = logger
    
    def __enter__(self):
        self.pbar = tqdm(total=self.total, desc=self.description)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar:
            self.pbar.close()
    
    def update(self, n: int = 1, **kwargs):
        """Update progress."""
        self.current += n
        if self.pbar:
            self.pbar.update(n)
            if kwargs:
                self.pbar.set_postfix(kwargs)
        
        # Log progress every 10%
        if self.total > 0 and self.current % max(1, self.total // 10) == 0:
            progress = (self.current / self.total) * 100
            elapsed = time.time() - self.start_time
            eta = (elapsed / self.current) * (self.total - self.current) if self.current > 0 else 0
            
            self.logger.info(f"Progress: {self.description}", 
                           progress=f"{progress:.1f}%",
                           elapsed=f"{elapsed:.1f}s",
                           eta=f"{eta:.1f}s")
    
    def set_description(self, description: str):
        """Update progress bar description."""
        self.description = description
        if self.pbar:
            self.pbar.set_description(description)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current progress statistics."""
        elapsed = time.time() - self.start_time
        progress = (self.current / self.total) * 100 if self.total > 0 else 0
        eta = (elapsed / self.current) * (self.total - self.current) if self.current > 0 else 0
        rate = self.current / elapsed if elapsed > 0 else 0
        
        return {
            "current": self.current,
            "total": self.total,
            "progress": progress,
            "elapsed": elapsed,
            "eta": eta,
            "rate": rate
        }
