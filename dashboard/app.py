"""
Web dashboard for the LLM Benchmarking Suite.
"""

import os
import json
import sys
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from framework.evaluator import BenchmarkEvaluator
from framework.registry import BenchmarkRegistry
from framework.config_manager import config
from framework.logger import logger
from models.openai_adapter import OpenAIAdapter


app = Flask(__name__)
app.secret_key = 'llm_benchmarking_secret_key'

# Global evaluator instance
evaluator = None


def init_evaluator():
    """Initialize the evaluator with registered benchmarks."""
    global evaluator
    if evaluator is None:
        # Register benchmarks
        from benchmarks.empathy.empathy_benchmark import EmpathyBenchmark
        from benchmarks.commonsense.commonsense_benchmark import CommonsenseBenchmark
        from benchmarks.math_reasoning.math_reasoning_benchmark import MathReasoningBenchmark
        from benchmarks.code_generation.code_generation_benchmark import CodeGenerationBenchmark
        
        BenchmarkRegistry.register("empathy", EmpathyBenchmark)
        BenchmarkRegistry.register("commonsense", CommonsenseBenchmark)
        BenchmarkRegistry.register("math_reasoning", MathReasoningBenchmark)
        BenchmarkRegistry.register("code_generation", CodeGenerationBenchmark)
        
        evaluator = BenchmarkEvaluator()
        logger.info("Dashboard initialized with evaluator")


@app.route('/')
def index():
    """Main dashboard page."""
    init_evaluator()
    
    # Get available benchmarks
    benchmarks = []
    for name in BenchmarkRegistry.list_benchmarks():
        if config.is_benchmark_enabled(name):
            benchmark_config = config.get_benchmark_config(name)
            benchmarks.append({
                'name': name,
                'description': benchmark_config.get('description', ''),
                'enabled': True
            })
    
    # Get recent results
    recent_results = get_recent_results()
    
    return render_template('index.html', 
                         benchmarks=benchmarks, 
                         recent_results=recent_results)


@app.route('/benchmarks')
def benchmarks():
    """Benchmarks management page."""
    init_evaluator()
    
    benchmarks = []
    for name in BenchmarkRegistry.list_benchmarks():
        config_data = config.get_benchmark_config(name)
        benchmarks.append({
            'name': name,
            'description': config_data.get('description', ''),
            'enabled': config.is_benchmark_enabled(name),
            'data_path': config_data.get('data_path', ''),
            'max_examples': config_data.get('max_examples', None)
        })
    
    return render_template('benchmarks.html', benchmarks=benchmarks)


@app.route('/run_evaluation', methods=['POST'])
def run_evaluation():
    """Run evaluation with selected parameters."""
    init_evaluator()
    
    try:
        data = request.get_json()
        model_name = data.get('model', 'gpt-4o-mini')
        benchmark_names = data.get('benchmarks', [])
        parallel = data.get('parallel', True)
        
        # Create model adapter
        model_adapter = OpenAIAdapter(model_name)
        
        # Update config for this run
        config.update_config('evaluation.parallel_benchmarks', parallel)
        
        # Run evaluation
        results = evaluator.run_all_benchmarks(
            model_name=model_name,
            model_func=model_adapter.generate,
            benchmark_names=benchmark_names if benchmark_names else None
        )
        
        # Generate leaderboard
        if config.get('evaluation.generate_leaderboard', True):
            evaluator.generate_leaderboard()
        
        return jsonify({
            'success': True,
            'results': results,
            'message': f'Evaluation completed for {len(results)} benchmarks'
        })
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/results')
def results():
    """Results visualization page."""
    init_evaluator()
    
    # Get all result files
    results_dir = config.get('evaluation.results_dir', 'results')
    result_files = []
    
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(results_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    result_files.append({
                        'filename': filename,
                        'benchmark': data.get('benchmark', 'unknown'),
                        'model': data.get('model', 'unknown'),
                        'timestamp': data.get('timestamp', ''),
                        'metrics': data.get('metrics', {}),
                        'num_examples': data.get('num_examples', 0),
                        'duration': data.get('duration', 0)
                    })
                except Exception as e:
                    logger.error(f"Error reading result file {filename}: {e}")
    
    # Sort by timestamp (newest first)
    result_files.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return render_template('results.html', results=result_files)


@app.route('/results/<filename>')
def result_detail(filename):
    """Detailed view of a specific result."""
    init_evaluator()
    
    results_dir = config.get('evaluation.results_dir', 'results')
    filepath = os.path.join(results_dir, filename)
    
    if not os.path.exists(filepath):
        return "Result file not found", 404
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return render_template('result_detail.html', result=data)
    except Exception as e:
        logger.error(f"Error reading result file {filename}: {e}")
        return f"Error reading result file: {e}", 500


@app.route('/leaderboard')
def leaderboard():
    """Leaderboard page."""
    init_evaluator()
    
    # Read leaderboard data
    leaderboard_file = os.path.join(config.get('evaluation.results_dir', 'results'), 'leaderboard.md')
    
    if os.path.exists(leaderboard_file):
        with open(leaderboard_file, 'r') as f:
            leaderboard_content = f.read()
    else:
        leaderboard_content = "No leaderboard data available. Run some evaluations first."
    
    return render_template('leaderboard.html', content=leaderboard_content)


@app.route('/config')
def config_page():
    """Configuration management page."""
    init_evaluator()
    
    # Get current configuration
    config_data = {
        'models': config.get('models', {}),
        'benchmarks': config.get('benchmarks', {}),
        'evaluation': config.get('evaluation', {}),
        'logging': config.get('logging', {})
    }
    
    return render_template('config.html', config=config_data)


@app.route('/api/benchmarks')
def api_benchmarks():
    """API endpoint for benchmark information."""
    init_evaluator()
    
    benchmarks = []
    for name in BenchmarkRegistry.list_benchmarks():
        if config.is_benchmark_enabled(name):
            benchmark_config = config.get_benchmark_config(name)
            benchmarks.append({
                'name': name,
                'description': benchmark_config.get('description', ''),
                'enabled': True,
                'data_path': benchmark_config.get('data_path', ''),
                'max_examples': config_data.get('max_examples', None)
            })
    
    return jsonify(benchmarks)


@app.route('/api/results')
def api_results():
    """API endpoint for results data."""
    init_evaluator()
    
    results_dir = config.get('evaluation.results_dir', 'results')
    results = []
    
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(results_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    results.append({
                        'filename': filename,
                        'benchmark': data.get('benchmark', 'unknown'),
                        'model': data.get('model', 'unknown'),
                        'timestamp': data.get('timestamp', ''),
                        'metrics': data.get('metrics', {}),
                        'num_examples': data.get('num_examples', 0),
                        'duration': data.get('duration', 0)
                    })
                except Exception as e:
                    logger.error(f"Error reading result file {filename}: {e}")
    
    return jsonify(results)


def get_recent_results(limit=5):
    """Get recent evaluation results."""
    results_dir = config.get('evaluation.results_dir', 'results')
    recent_results = []
    
    if os.path.exists(results_dir):
        files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
        
        for filename in files[:limit]:
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                recent_results.append({
                    'filename': filename,
                    'benchmark': data.get('benchmark', 'unknown'),
                    'model': data.get('model', 'unknown'),
                    'timestamp': data.get('timestamp', ''),
                    'accuracy': data.get('metrics', {}).get('accuracy', 0),
                    'duration': data.get('duration', 0)
                })
            except Exception as e:
                logger.error(f"Error reading result file {filename}: {e}")
    
    return recent_results


if __name__ == '__main__':
    # Initialize evaluator
    init_evaluator()
    
    # Get dashboard configuration
    dashboard_config = config.get('dashboard', {})
    host = dashboard_config.get('host', 'localhost')
    port = dashboard_config.get('port', 8080)
    debug = dashboard_config.get('debug', False)
    
    logger.info(f"Starting dashboard on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
