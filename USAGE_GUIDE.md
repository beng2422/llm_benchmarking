# LLM Benchmarking Suite - Usage Guide

This guide provides comprehensive instructions for using the enhanced LLM Benchmarking Suite.

## 📋 Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Configuration](#configuration)
3. [Command Line Usage](#command-line-usage)
4. [Web Dashboard](#web-dashboard)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Git (for cloning the repository)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/llm-benchmarking.git
   cd llm-benchmarking
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API key:**
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   ```

5. **Verify installation:**
   ```bash
   python run_evaluation.py --list-benchmarks
   ```

## ⚙️ Configuration

The framework uses YAML configuration files for all settings. The default configuration is in `config.yaml`.

### Key Configuration Sections

#### Models Configuration
```yaml
models:
  openai:
    default_model: "gpt-4o-mini"
    api_timeout: 30
    max_retries: 3
    retry_delay: 1.0
    models:
      gpt-4o:
        max_tokens: 100
        temperature: 0.0
      gpt-4o-mini:
        max_tokens: 50
        temperature: 0.0
```

#### Benchmarks Configuration
```yaml
benchmarks:
  empathy:
    enabled: true
    data_path: "benchmarks/empathy/data"
    max_examples: null  # null means all examples
    timeout: 60
```

#### Evaluation Configuration
```yaml
evaluation:
  parallel_benchmarks: true
  parallel_examples: false
  max_workers: 4
  save_detailed_results: true
  generate_leaderboard: true
  results_dir: "results"
```

#### Logging Configuration
```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/benchmarking.log"
  max_size: "10MB"
  backup_count: 5
```

### Custom Configuration

Create a custom configuration file:

```bash
cp config.yaml my_config.yaml
# Edit my_config.yaml with your settings
python run_evaluation.py --config my_config.yaml
```

## 💻 Command Line Usage

### Basic Commands

#### List Available Options
```bash
python run_evaluation.py --help
```

#### List Benchmarks and Models
```bash
python run_evaluation.py --list-benchmarks
python run_evaluation.py --list-models
```

#### Run All Benchmarks
```bash
python run_evaluation.py --model gpt-4o-mini
```

#### Run Specific Benchmarks
```bash
python run_evaluation.py --benchmarks empathy commonsense --model gpt-4o-mini
```

#### Run with Analysis
```bash
python run_evaluation.py --model gpt-4o-mini --generate-analysis
```

### Advanced Options

#### Parallel Execution
```bash
# Force parallel execution
python run_evaluation.py --parallel

# Disable parallel execution
python run_evaluation.py --no-parallel
```

#### Limit Examples
```bash
# Run only 50 examples per benchmark
python run_evaluation.py --max-examples 50
```

#### Verbose Output
```bash
# Show detailed output and model information
python run_evaluation.py --verbose
```

#### Custom Results Directory
```bash
python run_evaluation.py --results-dir my_results
```

### Complete Example

```bash
python run_evaluation.py \
  --model gpt-4o \
  --benchmarks empathy commonsense math_reasoning \
  --parallel \
  --max-examples 100 \
  --generate-analysis \
  --verbose
```

## 🌐 Web Dashboard

### Starting the Dashboard

```bash
python run_evaluation.py --dashboard
```

The dashboard will be available at `http://localhost:8080` by default.

### Dashboard Features

1. **Dashboard Overview**
   - Quick stats and recent results
   - Available benchmarks overview
   - Recent evaluation history

2. **Benchmark Management**
   - View all benchmarks
   - Enable/disable benchmarks
   - Configure benchmark settings

3. **Run Evaluations**
   - Interactive evaluation setup
   - Model selection
   - Benchmark selection
   - Real-time progress tracking

4. **Results Analysis**
   - View all results
   - Detailed result inspection
   - Performance metrics visualization

5. **Leaderboard**
   - Model performance comparison
   - Benchmark difficulty ranking
   - Historical performance trends

### Dashboard Configuration

Configure the dashboard in `config.yaml`:

```yaml
dashboard:
  enabled: true
  host: "localhost"
  port: 8080
  debug: false
```

## 🔧 Advanced Features

### Data Validation

The framework automatically validates benchmark data:

```python
from framework.data_validator import DataValidator

validator = DataValidator()
result = validator.validate_benchmark_data('empathy', data)
print(f"Valid: {result['valid']}")
print(f"Errors: {result['errors']}")
```

### Custom Analysis

Generate custom analysis reports:

```python
from framework.results_analyzer import ResultsAnalyzer

analyzer = ResultsAnalyzer("results")
analysis = analyzer.analyze_all_results()
report_path = analyzer.generate_report()
```

### Parallel Processing

Configure parallel processing:

```python
from framework.parallel_executor import ParallelExecutor

executor = ParallelExecutor(max_workers=8)
# Use executor for custom parallel tasks
```

### Custom Logging

Use structured logging:

```python
from framework.logger import logger

logger.info("Custom message", key1="value1", key2="value2")
logger.log_benchmark_start("empathy", "gpt-4o", 100)
```

### Data Management

Use the data manager for caching and validation:

```python
from framework.data_manager import DataManager

manager = DataManager()
data, metadata = manager.load_benchmark_data("empathy", "data/empathy.json")
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_framework.py

# Run with verbose output
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=framework
```

### Test Structure

- `tests/test_framework.py` - Core framework tests
- `tests/test_benchmarks.py` - Benchmark-specific tests
- `tests/test_models.py` - Model adapter tests
- `tests/test_integration.py` - End-to-end integration tests

## 📊 Results and Analysis

### Result Files

Results are saved in JSON format with the following structure:

```json
{
  "benchmark": "empathy",
  "model": "gpt-4o-mini",
  "timestamp": "2024-01-01T00:00:00",
  "metrics": {
    "accuracy": 0.85,
    "f1_score": 0.82,
    "precision": 0.80,
    "recall": 0.84
  },
  "num_examples": 100,
  "duration": 30.5,
  "results": [...]
}
```

### Analysis Output

The analysis generates:

1. **Visualizations** (PNG files)
   - Model comparison charts
   - Benchmark difficulty charts
   - Performance trend graphs
   - Correlation heatmaps

2. **Statistical Analysis** (JSON file)
   - Model performance statistics
   - Statistical significance tests
   - Correlation analysis
   - Trend analysis

3. **HTML Report**
   - Comprehensive analysis report
   - Interactive visualizations
   - Performance summaries

### Leaderboard

The leaderboard is automatically generated and shows:

- Model performance across benchmarks
- Benchmark difficulty rankings
- Historical performance trends
- Statistical significance indicators

## 🐛 Troubleshooting

### Common Issues

#### 1. API Key Not Found
```
Error: OpenAI API key not provided
```
**Solution:** Set your API key:
```bash
export OPENAI_API_KEY="your_key_here"
```

#### 2. Configuration File Not Found
```
Error: Configuration file not found
```
**Solution:** Ensure `config.yaml` exists or specify a custom config:
```bash
python run_evaluation.py --config my_config.yaml
```

#### 3. Import Errors
```
ModuleNotFoundError: No module named 'framework'
```
**Solution:** Ensure you're in the project root directory and dependencies are installed:
```bash
pip install -r requirements.txt
```

#### 4. Permission Errors
```
PermissionError: [Errno 13] Permission denied
```
**Solution:** Check file permissions and ensure you have write access to the results directory.

#### 5. Memory Issues
```
MemoryError: Unable to allocate array
```
**Solution:** Reduce the number of parallel workers or examples:
```bash
python run_evaluation.py --max-examples 50 --no-parallel
```

### Debug Mode

Enable debug logging for detailed troubleshooting:

1. Update `config.yaml`:
   ```yaml
   logging:
     level: "DEBUG"
   ```

2. Run with verbose output:
   ```bash
   python run_evaluation.py --verbose
   ```

### Log Files

Check log files for detailed error information:

- `logs/benchmarking.log` - Main application log
- Console output - Real-time status updates

### Performance Issues

If evaluations are running slowly:

1. **Enable parallel processing:**
   ```bash
   python run_evaluation.py --parallel
   ```

2. **Increase worker count** in `config.yaml`:
   ```yaml
   evaluation:
     max_workers: 8
   ```

3. **Reduce examples** for testing:
   ```bash
   python run_evaluation.py --max-examples 10
   ```

## 📞 Support

For additional help:

1. Check the [README.md](README.md) for general information
2. Review the [test files](tests/) for usage examples
3. Check the [configuration file](config.yaml) for all available options
4. Enable debug logging for detailed error information

## 🔄 Updates

To update the framework:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

Check the [CHANGELOG.md](CHANGELOG.md) for detailed update information.
