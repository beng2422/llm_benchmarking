"""
Unified evaluator for running multiple benchmarks.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from .base_benchmark import BaseBenchmark
from .registry import BenchmarkRegistry
from .parallel_executor import ParallelExecutor, ProgressTracker
from .logger import logger
from .config_manager import config


class BenchmarkEvaluator:
    """Unified evaluator for running benchmarks across multiple models."""
    
    def __init__(self, results_dir: Optional[str] = None):
        self.results_dir = results_dir or config.get('evaluation.results_dir', 'results')
        self.parallel_executor = ParallelExecutor()
        self.logger = logger
        os.makedirs(self.results_dir, exist_ok=True)
    
    def run_benchmark(
        self, 
        benchmark_name: str, 
        model_name: str, 
        model_func: callable,
        **model_kwargs
    ) -> Dict[str, Any]:
        """
        Run a single benchmark with a single model.
        
        Args:
            benchmark_name: Name of the benchmark to run
            model_name: Name of the model being evaluated
            model_func: Function that takes a prompt and returns a response
            **model_kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary containing results and metrics
        """
        start_time = time.time()
        
        try:
            # Get benchmark instance
            benchmark = BenchmarkRegistry.get_benchmark(benchmark_name)
            data = benchmark.get_data()
            
            # Log benchmark start
            self.logger.log_benchmark_start(benchmark_name, model_name, len(data))
            
            # Run evaluation
            results = benchmark.evaluate_model(model_name, model_func, **model_kwargs)
            
            # Calculate metrics
            metrics = benchmark.calculate_metrics(results)
            
            duration = time.time() - start_time
            
            # Prepare final results
            final_results = {
                "benchmark": benchmark_name,
                "model": model_name,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "num_examples": len(results),
                "results": results,
                "duration": duration
            }
            
            # Save results if configured
            if config.get('evaluation.save_detailed_results', True):
                self._save_results(final_results)
            
            # Log completion
            self.logger.log_benchmark_complete(benchmark_name, model_name, metrics, duration)
            
            return final_results
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.log_error(e, f"benchmark {benchmark_name}")
            return {
                "benchmark": benchmark_name,
                "model": model_name,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "duration": duration
            }
    
    def run_all_benchmarks(
        self, 
        model_name: str, 
        model_func: callable,
        benchmark_names: Optional[List[str]] = None,
        **model_kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all benchmarks (or specified ones) with a single model.
        
        Args:
            model_name: Name of the model being evaluated
            model_func: Function that takes a prompt and returns a response
            benchmark_names: List of benchmark names to run (None for all)
            **model_kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary mapping benchmark names to results
        """
        if benchmark_names is None:
            # Get enabled benchmarks from config
            benchmark_names = config.get_enabled_benchmarks()
        
        # Filter out disabled benchmarks
        enabled_benchmarks = [name for name in benchmark_names if config.is_benchmark_enabled(name)]
        
        if not enabled_benchmarks:
            self.logger.warning("No enabled benchmarks found")
            return {}
        
        self.logger.info(f"Running {len(enabled_benchmarks)} benchmarks", 
                        model=model_name, benchmarks=enabled_benchmarks)
        
        # Check if parallel execution is enabled
        if config.get('evaluation.parallel_benchmarks', True) and len(enabled_benchmarks) > 1:
            return self._run_benchmarks_parallel(enabled_benchmarks, model_name, model_func, **model_kwargs)
        else:
            return self._run_benchmarks_sequential(enabled_benchmarks, model_name, model_func, **model_kwargs)
    
    def _run_benchmarks_parallel(self, benchmark_names: List[str], model_name: str, 
                                model_func: callable, **model_kwargs) -> Dict[str, Dict[str, Any]]:
        """Run benchmarks in parallel."""
        # Prepare benchmark configurations
        benchmark_configs = []
        for benchmark_name in benchmark_names:
            benchmark_class = BenchmarkRegistry._benchmarks[benchmark_name]
            config_kwargs = config.get_benchmark_config(benchmark_name)
            benchmark_configs.append({
                'name': benchmark_name,
                'class': benchmark_class,
                'model_name': model_name,
                'kwargs': config_kwargs
            })
        
        # Run in parallel
        return self.parallel_executor.run_benchmarks_parallel(
            benchmark_configs, model_func, **model_kwargs
        )
    
    def _run_benchmarks_sequential(self, benchmark_names: List[str], model_name: str, 
                                  model_func: callable, **model_kwargs) -> Dict[str, Dict[str, Any]]:
        """Run benchmarks sequentially."""
        all_results = {}
        
        with ProgressTracker(len(benchmark_names), "Running benchmarks") as progress:
            for benchmark_name in benchmark_names:
                try:
                    results = self.run_benchmark(benchmark_name, model_name, model_func, **model_kwargs)
                    all_results[benchmark_name] = results
                    progress.update(1, benchmark=benchmark_name)
                except Exception as e:
                    self.logger.log_error(e, f"benchmark {benchmark_name}")
                    all_results[benchmark_name] = {"error": str(e)}
                    progress.update(1, benchmark=benchmark_name)
        
        return all_results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to file."""
        benchmark_name = results["benchmark"]
        model_name = results["model"]
        timestamp = results["timestamp"].replace(":", "-")
        
        filename = f"{benchmark_name}_{model_name}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def generate_leaderboard(self, output_file: str = "results/leaderboard.md"):
        """Generate a markdown leaderboard from all results."""
        leaderboard_data = self._collect_leaderboard_data()
        
        with open(output_file, 'w') as f:
            f.write("# LLM Benchmarking Leaderboard\n\n")
            f.write("This leaderboard tracks results across all benchmarks.\n\n")
            
            for benchmark_name, models in leaderboard_data.items():
                f.write(f"## {benchmark_name}\n\n")
                f.write("| Model | Accuracy | F1 Score | Examples |\n")
                f.write("|-------|----------|----------|----------|\n")
                
                for model_name, metrics in models.items():
                    acc = metrics.get('accuracy', 0.0)
                    f1 = metrics.get('f1_score', 0.0)
                    examples = metrics.get('num_examples', 0)
                    f.write(f"| {model_name} | {acc:.3f} | {f1:.3f} | {examples} |\n")
                
                f.write("\n")
    
    def _collect_leaderboard_data(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Collect all results for leaderboard generation."""
        leaderboard = {}
        
        for filename in os.listdir(self.results_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.results_dir, filename), 'r') as f:
                        data = json.load(f)
                    
                    benchmark_name = data.get('benchmark', 'unknown')
                    model_name = data.get('model', 'unknown')
                    metrics = data.get('metrics', {})
                    
                    if benchmark_name not in leaderboard:
                        leaderboard[benchmark_name] = {}
                    
                    leaderboard[benchmark_name][model_name] = metrics
                    
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
        
        return leaderboard
