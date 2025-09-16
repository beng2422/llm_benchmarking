"""
Advanced results analysis and visualization for the LLM Benchmarking Suite.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from scipy import stats
from .logger import logger
from .config_manager import config


class ResultsAnalyzer:
    """Advanced analysis and visualization of benchmark results."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.logger = logger
        self.output_dir = Path("analysis_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def analyze_all_results(self) -> Dict[str, Any]:
        """Perform comprehensive analysis of all results."""
        self.logger.info("Starting comprehensive results analysis")
        
        # Load all results
        all_results = self._load_all_results()
        
        if not all_results:
            self.logger.warning("No results found for analysis")
            return {}
        
        # Convert to DataFrame for easier analysis
        df = self._results_to_dataframe(all_results)
        
        # Perform various analyses
        analysis = {
            'overview': self._analyze_overview(df),
            'model_comparison': self._analyze_model_comparison(df),
            'benchmark_comparison': self._analyze_benchmark_comparison(df),
            'performance_trends': self._analyze_performance_trends(df),
            'statistical_tests': self._perform_statistical_tests(df),
            'correlation_analysis': self._analyze_correlations(df)
        }
        
        # Generate visualizations
        self._generate_visualizations(df, analysis)
        
        # Save analysis results
        self._save_analysis_results(analysis)
        
        self.logger.info("Results analysis completed")
        return analysis
    
    def _load_all_results(self) -> List[Dict[str, Any]]:
        """Load all result files."""
        results = []
        
        for result_file in self.results_dir.glob("*.json"):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                # Skip error results
                if 'error' not in data:
                    results.append(data)
                    
            except Exception as e:
                self.logger.error(f"Error loading result file {result_file}: {e}")
        
        return results
    
    def _results_to_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        rows = []
        
        for result in results:
            metrics = result.get('metrics', {})
            row = {
                'benchmark': result.get('benchmark', 'unknown'),
                'model': result.get('model', 'unknown'),
                'timestamp': result.get('timestamp', ''),
                'duration': result.get('duration', 0),
                'num_examples': result.get('num_examples', 0),
                'accuracy': metrics.get('accuracy', 0),
                'f1_score': metrics.get('f1_score', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0)
            }
            
            # Add benchmark-specific metrics
            if result.get('benchmark') == 'code_generation':
                row.update({
                    'execution_success_rate': metrics.get('execution_success_rate', 0),
                    'avg_accuracy': metrics.get('avg_accuracy', 0)
                })
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _analyze_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall performance overview."""
        overview = {
            'total_evaluations': len(df),
            'unique_models': df['model'].nunique(),
            'unique_benchmarks': df['benchmark'].nunique(),
            'date_range': {
                'start': df['timestamp'].min() if not df.empty else None,
                'end': df['timestamp'].max() if not df.empty else None
            },
            'overall_metrics': {
                'mean_accuracy': df['accuracy'].mean(),
                'std_accuracy': df['accuracy'].std(),
                'mean_f1': df['f1_score'].mean(),
                'mean_duration': df['duration'].mean()
            }
        }
        
        return overview
    
    def _analyze_model_comparison(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compare performance across different models."""
        model_stats = df.groupby('model').agg({
            'accuracy': ['mean', 'std', 'count'],
            'f1_score': ['mean', 'std'],
            'duration': ['mean', 'std'],
            'num_examples': 'sum'
        }).round(4)
        
        # Flatten column names
        model_stats.columns = ['_'.join(col).strip() for col in model_stats.columns]
        model_stats = model_stats.reset_index()
        
        # Rank models by accuracy
        model_ranking = model_stats.sort_values('accuracy_mean', ascending=False)
        
        return {
            'model_statistics': model_stats.to_dict('records'),
            'model_ranking': model_ranking.to_dict('records'),
            'best_model': model_ranking.iloc[0]['model'] if not model_ranking.empty else None
        }
    
    def _analyze_benchmark_comparison(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compare performance across different benchmarks."""
        benchmark_stats = df.groupby('benchmark').agg({
            'accuracy': ['mean', 'std', 'count'],
            'f1_score': ['mean', 'std'],
            'duration': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        benchmark_stats.columns = ['_'.join(col).strip() for col in benchmark_stats.columns]
        benchmark_stats = benchmark_stats.reset_index()
        
        # Rank benchmarks by difficulty (lower accuracy = harder)
        benchmark_ranking = benchmark_stats.sort_values('accuracy_mean', ascending=True)
        
        return {
            'benchmark_statistics': benchmark_stats.to_dict('records'),
            'benchmark_ranking': benchmark_ranking.to_dict('records'),
            'hardest_benchmark': benchmark_ranking.iloc[0]['benchmark'] if not benchmark_ranking.empty else None
        }
    
    def _analyze_performance_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if df.empty or 'timestamp' not in df.columns:
            return {}
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['datetime'].dt.date
        
        # Group by date and model
        daily_performance = df.groupby(['date', 'model']).agg({
            'accuracy': 'mean',
            'f1_score': 'mean',
            'duration': 'mean'
        }).reset_index()
        
        return {
            'daily_performance': daily_performance.to_dict('records'),
            'trend_analysis': self._calculate_trends(daily_performance)
        }
    
    def _calculate_trends(self, daily_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance trends."""
        trends = {}
        
        for model in daily_df['model'].unique():
            model_data = daily_df[daily_df['model'] == model].sort_values('date')
            
            if len(model_data) > 1:
                # Calculate trend slope
                x = np.arange(len(model_data))
                y = model_data['accuracy'].values
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                trends[model] = {
                    'slope': slope,
                    'r_squared': r_value ** 2,
                    'p_value': p_value,
                    'trend_direction': 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable'
                }
        
        return trends
    
    def _perform_statistical_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        tests = {}
        
        # Compare models pairwise
        models = df['model'].unique()
        if len(models) > 1:
            model_comparisons = {}
            
            for i, model1 in enumerate(models):
                for model2 in models[i+1:]:
                    model1_data = df[df['model'] == model1]['accuracy']
                    model2_data = df[df['model'] == model2]['accuracy']
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(model1_data, model2_data)
                    
                    model_comparisons[f"{model1}_vs_{model2}"] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'effect_size': self._calculate_cohens_d(model1_data, model2_data)
                    }
            
            tests['model_comparisons'] = model_comparisons
        
        # Compare benchmarks
        benchmarks = df['benchmark'].unique()
        if len(benchmarks) > 1:
            benchmark_comparisons = {}
            
            for i, bench1 in enumerate(benchmarks):
                for bench2 in benchmarks[i+1:]:
                    bench1_data = df[df['benchmark'] == bench1]['accuracy']
                    bench2_data = df[df['benchmark'] == bench2]['accuracy']
                    
                    t_stat, p_value = stats.ttest_ind(bench1_data, bench2_data)
                    
                    benchmark_comparisons[f"{bench1}_vs_{bench2}"] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'effect_size': self._calculate_cohens_d(bench1_data, bench2_data)
                    }
            
            tests['benchmark_comparisons'] = benchmark_comparisons
        
        return tests
    
    def _calculate_cohens_d(self, group1: pd.Series, group2: pd.Series) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        s1, s2 = group1.std(), group2.std()
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        
        return (group1.mean() - group2.mean()) / pooled_std
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between different metrics."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': self._find_strong_correlations(correlation_matrix)
        }
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find strong correlations above threshold."""
        strong_corrs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_corrs.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return strong_corrs
    
    def _generate_visualizations(self, df: pd.DataFrame, analysis: Dict[str, Any]):
        """Generate comprehensive visualizations."""
        self.logger.info("Generating visualizations")
        
        # Model comparison chart
        self._plot_model_comparison(df)
        
        # Benchmark comparison chart
        self._plot_benchmark_comparison(df)
        
        # Performance trends
        self._plot_performance_trends(df)
        
        # Correlation heatmap
        self._plot_correlation_heatmap(df)
        
        # Statistical significance plot
        self._plot_statistical_significance(analysis.get('statistical_tests', {}))
    
    def _plot_model_comparison(self, df: pd.DataFrame):
        """Plot model comparison chart."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # Accuracy comparison
        sns.barplot(data=df, x='model', y='accuracy', ax=axes[0, 0])
        axes[0, 0].set_title('Accuracy by Model')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1 Score comparison
        sns.barplot(data=df, x='model', y='f1_score', ax=axes[0, 1])
        axes[0, 1].set_title('F1 Score by Model')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Duration comparison
        sns.barplot(data=df, x='model', y='duration', ax=axes[1, 0])
        axes[1, 0].set_title('Average Duration by Model')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Box plot for accuracy distribution
        sns.boxplot(data=df, x='model', y='accuracy', ax=axes[1, 1])
        axes[1, 1].set_title('Accuracy Distribution by Model')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_benchmark_comparison(self, df: pd.DataFrame):
        """Plot benchmark comparison chart."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Benchmark Difficulty Comparison', fontsize=16)
        
        # Accuracy by benchmark
        sns.barplot(data=df, x='benchmark', y='accuracy', ax=axes[0, 0])
        axes[0, 0].set_title('Accuracy by Benchmark')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1 Score by benchmark
        sns.barplot(data=df, x='benchmark', y='f1_score', ax=axes[0, 1])
        axes[0, 1].set_title('F1 Score by Benchmark')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Duration by benchmark
        sns.barplot(data=df, x='benchmark', y='duration', ax=axes[1, 0])
        axes[1, 0].set_title('Average Duration by Benchmark')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Heatmap of model vs benchmark performance
        pivot_table = df.pivot_table(values='accuracy', index='model', columns='benchmark', aggfunc='mean')
        sns.heatmap(pivot_table, annot=True, fmt='.3f', ax=axes[1, 1], cmap='YlOrRd')
        axes[1, 1].set_title('Model vs Benchmark Performance Heatmap')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'benchmark_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_trends(self, df: pd.DataFrame):
        """Plot performance trends over time."""
        if df.empty or 'timestamp' not in df.columns:
            return
        
        df['datetime'] = pd.to_datetime(df['timestamp'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model].sort_values('datetime')
            ax.plot(model_data['datetime'], model_data['accuracy'], 
                   marker='o', label=model, linewidth=2)
        
        ax.set_title('Performance Trends Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmap(self, df: pd.DataFrame):
        """Plot correlation heatmap."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return
        
        plt.figure(figsize=(10, 8))
        correlation_matrix = df[numeric_cols].corr()
        
        sns.heatmap(correlation_matrix, annot=True, fmt='.3f', 
                   cmap='coolwarm', center=0, square=True)
        plt.title('Correlation Matrix of Performance Metrics')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_significance(self, statistical_tests: Dict[str, Any]):
        """Plot statistical significance results."""
        if not statistical_tests:
            return
        
        # Create a summary plot of p-values
        comparisons = []
        p_values = []
        
        for test_type, tests in statistical_tests.items():
            for comparison, results in tests.items():
                comparisons.append(comparison)
                p_values.append(results['p_value'])
        
        if not comparisons:
            return
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(comparisons)), p_values)
        plt.axhline(y=0.05, color='r', linestyle='--', label='Significance threshold (p=0.05)')
        
        # Color bars based on significance
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            if p_val < 0.05:
                bar.set_color('red')
            else:
                bar.set_color('blue')
        
        plt.xlabel('Comparisons')
        plt.ylabel('P-value')
        plt.title('Statistical Significance Tests')
        plt.xticks(range(len(comparisons)), comparisons, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'statistical_significance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_analysis_results(self, analysis: Dict[str, Any]):
        """Save analysis results to file."""
        output_file = self.output_dir / f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        self.logger.info(f"Analysis results saved to {output_file}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive HTML report."""
        analysis = self.analyze_all_results()
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Benchmarking Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; margin-bottom: 40px; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f5f5f5; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .chart {{ text-align: center; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>LLM Benchmarking Analysis Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Overview</h2>
                <div class="metric">Total Evaluations: {analysis.get('overview', {}).get('total_evaluations', 0)}</div>
                <div class="metric">Unique Models: {analysis.get('overview', {}).get('unique_models', 0)}</div>
                <div class="metric">Unique Benchmarks: {analysis.get('overview', {}).get('unique_benchmarks', 0)}</div>
                <div class="metric">Mean Accuracy: {analysis.get('overview', {}).get('overall_metrics', {}).get('mean_accuracy', 0):.3f}</div>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <div class="chart">
                    <img src="model_comparison.png" alt="Model Comparison" style="max-width: 100%;">
                </div>
                <div class="chart">
                    <img src="benchmark_comparison.png" alt="Benchmark Comparison" style="max-width: 100%;">
                </div>
                <div class="chart">
                    <img src="performance_trends.png" alt="Performance Trends" style="max-width: 100%;">
                </div>
            </div>
        </body>
        </html>
        """
        
        report_file = self.output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Analysis report saved to {report_file}")
        return str(report_file)
