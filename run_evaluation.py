#!/usr/bin/env python3
"""
Enhanced evaluation script for LLM Benchmarking Suite.

This script provides a unified interface for running multiple benchmarks
across different models with comprehensive logging, parallel processing,
and advanced analysis capabilities.
"""

import argparse
import os
import sys
import time
from typing import List, Optional
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from framework.evaluator import BenchmarkEvaluator
from framework.registry import BenchmarkRegistry
from framework.config_manager import config
from framework.logger import logger
from framework.results_analyzer import ResultsAnalyzer
from models.openai_adapter import OpenAIAdapter
from benchmarks.empathy.empathy_benchmark import EmpathyBenchmark
from benchmarks.commonsense.commonsense_benchmark import CommonsenseBenchmark
from benchmarks.math_reasoning.math_reasoning_benchmark import MathReasoningBenchmark
from benchmarks.code_generation.code_generation_benchmark import CodeGenerationBenchmark


def register_benchmarks():
    """Register all available benchmarks."""
    BenchmarkRegistry.register("empathy", EmpathyBenchmark)
    BenchmarkRegistry.register("commonsense", CommonsenseBenchmark)
    BenchmarkRegistry.register("math_reasoning", MathReasoningBenchmark)
    BenchmarkRegistry.register("code_generation", CodeGenerationBenchmark)


def create_model_adapter(model_name: str, **kwargs):
    """Create a model adapter based on the model name."""
    if model_name.startswith("gpt-") or model_name.startswith("o1-"):
        return OpenAIAdapter(model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Run LLM benchmarks with enhanced features")
    parser.add_argument("--benchmarks", nargs="+", default=None, 
                       help="Benchmarks to run (default: all enabled)")
    parser.add_argument("--model", default=None, 
                       help="Model to evaluate (default: from config)")
    parser.add_argument("--results-dir", default=None, 
                       help="Directory to save results (default: from config)")
    parser.add_argument("--generate-leaderboard", action="store_true",
                       help="Generate leaderboard after evaluation")
    parser.add_argument("--generate-analysis", action="store_true",
                       help="Generate comprehensive analysis report")
    parser.add_argument("--list-benchmarks", action="store_true",
                       help="List available benchmarks and exit")
    parser.add_argument("--list-models", action="store_true",
                       help="List available models and exit")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed output")
    parser.add_argument("--config", default="config.yaml",
                       help="Configuration file path")
    parser.add_argument("--parallel", action="store_true", default=None,
                       help="Force parallel execution")
    parser.add_argument("--no-parallel", action="store_true",
                       help="Disable parallel execution")
    parser.add_argument("--max-examples", type=int, default=None,
                       help="Maximum examples per benchmark")
    parser.add_argument("--dashboard", action="store_true",
                       help="Start web dashboard instead of running evaluation")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        global config
        config = ConfigManager(args.config)
        logger.log_config_loaded(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    # Start dashboard if requested
    if args.dashboard:
        start_dashboard()
        return
    
    # Register benchmarks
    register_benchmarks()
    
    # List benchmarks if requested
    if args.list_benchmarks:
        print("Available benchmarks:")
        for benchmark_name in BenchmarkRegistry.list_benchmarks():
            enabled = config.is_benchmark_enabled(benchmark_name)
            status = "✓" if enabled else "✗"
            print(f"  {status} {benchmark_name}")
        return
    
    # List models if requested
    if args.list_models:
        print("Available models:")
        openai_models = config.get('models.openai.models', {})
        for model_name in openai_models.keys():
            print(f"  - {model_name}")
        return
    
    # Get configuration values
    model_name = args.model or config.get('models.openai.default_model', 'gpt-4o-mini')
    results_dir = args.results_dir or config.get('evaluation.results_dir', 'results')
    
    # Update configuration based on arguments
    if args.parallel is not None:
        config.update_config('evaluation.parallel_benchmarks', args.parallel)
    elif args.no_parallel:
        config.update_config('evaluation.parallel_benchmarks', False)
    
    if args.max_examples:
        for benchmark_name in config.get_enabled_benchmarks():
            config.update_config(f'benchmarks.{benchmark_name}.max_examples', args.max_examples)
    
    # Create evaluator
    evaluator = BenchmarkEvaluator(results_dir)
    
    # Create model adapter
    try:
        model_adapter = create_model_adapter(model_name)
        logger.info(f"Using model: {model_name}")
        if args.verbose:
            model_info = model_adapter.get_model_info()
            print(f"Model info: {model_info}")
    except Exception as e:
        logger.error(f"Error creating model adapter: {e}")
        print(f"Error creating model adapter: {e}")
        return
    
    # Run evaluation
    start_time = time.time()
    
    try:
        logger.info("Starting evaluation", 
                   model=model_name, 
                   benchmarks=args.benchmarks or "all enabled")
        
        results = evaluator.run_all_benchmarks(
            model_name=model_name,
            model_func=model_adapter.generate,
            benchmark_names=args.benchmarks
        )
        
        total_duration = time.time() - start_time
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Model: {model_name}")
        print(f"Total Duration: {total_duration:.2f} seconds")
        print(f"Benchmarks Run: {len(results)}")
        print()
        
        successful_benchmarks = 0
        total_accuracy = 0
        
        for benchmark_name, benchmark_results in results.items():
            if "error" in benchmark_results:
                print(f"❌ {benchmark_name}: {benchmark_results['error']}")
            else:
                successful_benchmarks += 1
                metrics = benchmark_results.get("metrics", {})
                accuracy = metrics.get('accuracy', 0.0)
                total_accuracy += accuracy
                
                print(f"✅ {benchmark_name}:")
                print(f"   Accuracy:  {accuracy:.3f}")
                print(f"   F1 Score:  {metrics.get('f1_score', 0.0):.3f}")
                print(f"   Examples:  {metrics.get('num_examples', 0)}")
                print(f"   Duration:  {benchmark_results.get('duration', 0):.2f}s")
        
        if successful_benchmarks > 0:
            avg_accuracy = total_accuracy / successful_benchmarks
            print(f"\nAverage Accuracy: {avg_accuracy:.3f}")
        
        # Generate leaderboard if requested
        if args.generate_leaderboard or config.get('evaluation.generate_leaderboard', True):
            print(f"\nGenerating leaderboard...")
            evaluator.generate_leaderboard()
            print("Leaderboard saved to results/leaderboard.md")
        
        # Generate analysis if requested
        if args.generate_analysis:
            print(f"\nGenerating comprehensive analysis...")
            analyzer = ResultsAnalyzer(results_dir)
            report_path = analyzer.generate_report()
            print(f"Analysis report saved to {report_path}")
        
        logger.info("Evaluation completed successfully", 
                   duration=total_duration, 
                   successful_benchmarks=successful_benchmarks)
        
    except Exception as e:
        duration = time.time() - start_time
        logger.log_error(e, "evaluation")
        print(f"Evaluation failed after {duration:.2f} seconds: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


def start_dashboard():
    """Start the web dashboard."""
    try:
        from dashboard.app import app, init_evaluator
        
        # Initialize evaluator
        init_evaluator()
        
        # Get dashboard configuration
        dashboard_config = config.get('dashboard', {})
        host = dashboard_config.get('host', 'localhost')
        port = dashboard_config.get('port', 8080)
        debug = dashboard_config.get('debug', False)
        
        print(f"Starting dashboard on http://{host}:{port}")
        print("Press Ctrl+C to stop the dashboard")
        
        app.run(host=host, port=port, debug=debug)
        
    except ImportError:
        print("Dashboard dependencies not installed. Install Flask to use the dashboard.")
        print("Run: pip install flask")
    except Exception as e:
        print(f"Error starting dashboard: {e}")


if __name__ == "__main__":
    main()
