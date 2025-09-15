#!/usr/bin/env python3
"""
Main evaluation script for LLM Benchmarking Suite.

This script provides a unified interface for running multiple benchmarks
across different models.
"""

import argparse
import os
import sys
from typing import List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from framework.evaluator import BenchmarkEvaluator
from framework.registry import BenchmarkRegistry
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
    parser = argparse.ArgumentParser(description="Run LLM benchmarks")
    parser.add_argument("--benchmarks", nargs="+", default=None, 
                       help="Benchmarks to run (default: all)")
    parser.add_argument("--model", default="gpt-4o-mini", 
                       help="Model to evaluate")
    parser.add_argument("--results-dir", default="results", 
                       help="Directory to save results")
    parser.add_argument("--generate-leaderboard", action="store_true",
                       help="Generate leaderboard after evaluation")
    parser.add_argument("--list-benchmarks", action="store_true",
                       help="List available benchmarks and exit")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed output")
    
    args = parser.parse_args()
    
    # Register benchmarks
    register_benchmarks()
    
    # List benchmarks if requested
    if args.list_benchmarks:
        print("Available benchmarks:")
        for benchmark_name in BenchmarkRegistry.list_benchmarks():
            print(f"  - {benchmark_name}")
        return
    
    # Create evaluator
    evaluator = BenchmarkEvaluator(args.results_dir)
    
    # Create model adapter
    try:
        model_adapter = create_model_adapter(args.model)
        print(f"Using model: {args.model}")
        if args.verbose:
            print(f"Model info: {model_adapter.get_model_info()}")
    except Exception as e:
        print(f"Error creating model adapter: {e}")
        return
    
    # Run evaluation
    try:
        results = evaluator.run_all_benchmarks(
            model_name=args.model,
            model_func=model_adapter.generate,
            benchmark_names=args.benchmarks
        )
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        for benchmark_name, benchmark_results in results.items():
            if "error" in benchmark_results:
                print(f"\n❌ {benchmark_name}: {benchmark_results['error']}")
            else:
                metrics = benchmark_results.get("metrics", {})
                print(f"\n✅ {benchmark_name}:")
                print(f"   Accuracy:  {metrics.get('accuracy', 0.0):.3f}")
                print(f"   F1 Score:  {metrics.get('f1_score', 0.0):.3f}")
                print(f"   Examples:  {metrics.get('num_examples', 0)}")
        
        # Generate leaderboard if requested
        if args.generate_leaderboard:
            print(f"\nGenerating leaderboard...")
            evaluator.generate_leaderboard()
            print("Leaderboard saved to results/leaderboard.md")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
