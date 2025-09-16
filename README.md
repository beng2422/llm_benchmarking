# LLM Benchmarking Suite

**LLM Benchmarking Suite** is a comprehensive, production-ready framework for evaluating large language models across multiple benchmarks. This project provides:

- **Multiple Benchmarks**: Empathy, Commonsense Reasoning, Math Reasoning, Code Generation, and more
- **Unified Evaluation Framework**: Consistent evaluation across all benchmarks with advanced analytics
- **Model Adapters**: Support for OpenAI with retry mechanisms and error handling
- **Parallel Processing**: Run multiple benchmarks and examples in parallel for faster evaluation
- **Web Dashboard**: Interactive web interface for running evaluations and visualizing results
- **Advanced Analytics**: Statistical analysis, trend tracking, and comprehensive reporting
- **Configuration Management**: YAML-based configuration for all settings
- **Comprehensive Logging**: Structured logging with multiple levels and file rotation
- **Data Validation**: Automatic data validation and caching for reliable results
- **Extensible Architecture**: Easy to add new benchmarks and models

---

## 📂 Project Structure

```
llm-benchmarking/
├── framework/                    # Core evaluation framework
│   ├── base_benchmark.py        # Abstract benchmark class
│   ├── evaluator.py             # Unified evaluator
│   └── registry.py              # Benchmark registry
├── benchmarks/                   # Individual benchmarks
│   ├── empathy/                 # Empathy response selection
│   │   ├── data/
│   │   └── empathy_benchmark.py
│   ├── commonsense/             # Commonsense reasoning
│   │   ├── data/
│   │   └── commonsense_benchmark.py
│   ├── math_reasoning/          # Mathematical reasoning
│   │   ├── data/
│   │   └── math_reasoning_benchmark.py
│   └── code_generation/         # Code generation
│       ├── data/
│       └── code_generation_benchmark.py
├── models/                      # Model adapters
│   ├── openai_adapter.py        # OpenAI API integration
│   └── base_adapter.py          # Base adapter interface
├── results/                     # Evaluation results
└── run_evaluation.py            # Main evaluation script
```

---

## 🚀 Quick Start

1. Clone the repo and install dependencies:
   ```bash
   git clone https://github.com/YOUR_USERNAME/llm-benchmarking.git
   cd llm-benchmarking
   pip install -r requirements.txt
   ```

2. Set your API key:
   ```bash
   export OPENAI_API_KEY="your_key_here"
   ```

3. Run all benchmarks with enhanced features:
   ```bash
   python run_evaluation.py --model gpt-4o-mini --generate-analysis
   ```

4. Run specific benchmarks in parallel:
   ```bash
   python run_evaluation.py --benchmarks empathy commonsense --model gpt-4o-mini --parallel
   ```

5. Start the web dashboard:
   ```bash
   python run_evaluation.py --dashboard
   ```

6. List available benchmarks and models:
   ```bash
   python run_evaluation.py --list-benchmarks
   python run_evaluation.py --list-models
   ```

7. Run with custom configuration:
   ```bash
   python run_evaluation.py --config custom_config.yaml --max-examples 50
   ```

---

## 📊 Available Benchmarks

### Empathy Benchmark
Evaluates whether LLMs can distinguish between empathetic and non-empathetic responses in therapy-like dialogues.

**Example:**
```json
{
  "patient": "I feel like I'm failing at everything lately.",
  "empathetic": "It sounds like you're feeling really overwhelmed, and that must be really tough to carry.",
  "non_empathetic": "You just need to work harder and stop worrying so much."
}
```

### Commonsense Benchmark
Tests basic commonsense reasoning abilities with multiple-choice questions.

**Example:**
```json
{
  "question": "If you drop a glass on a hard floor, what will most likely happen?",
  "options": {
    "A": "The glass will bounce back up",
    "B": "The glass will break",
    "C": "The glass will turn into water",
    "D": "The glass will float in the air"
  },
  "correct_answer": "B"
}
```

### Math Reasoning Benchmark
Evaluates mathematical problem-solving abilities across multiple domains including arithmetic, algebra, geometry, word problems, fractions, percentages, logic, sequences, probability, and exponents.

**Example:**
```json
{
  "question": "If a store sells 3 apples for $2.40, how much do 7 apples cost?",
  "answer": 5.60,
  "category": "arithmetic",
  "difficulty": "medium",
  "explanation": "First find price per apple: $2.40 ÷ 3 = $0.80. Then multiply by 7: $0.80 × 7 = $5.60"
}
```

### Code Generation Benchmark
Tests programming abilities across multiple languages (Python, JavaScript, Java) with problems covering algorithms, data structures, and problem-solving. Features automated code execution and validation.

**Example:**
```json
{
  "title": "Fibonacci Sequence",
  "description": "Write a function that returns the nth Fibonacci number.",
  "language": "python",
  "difficulty": "medium",
  "category": "algorithms",
  "function_signature": "def fibonacci(n):",
  "test_cases": [
    {"input": {"n": 5}, "expected_output": 5},
    {"input": {"n": 10}, "expected_output": 55}
  ]
}
```

---

## 🔧 Adding New Benchmarks

1. Create a new benchmark directory:
   ```bash
   mkdir benchmarks/your_benchmark
   ```

2. Implement the benchmark class:
   ```python
   from framework.base_benchmark import BaseBenchmark
   
   class YourBenchmark(BaseBenchmark):
       def load_data(self):
           # Load your dataset
           pass
       
       def evaluate_model(self, model_name, model_func, **kwargs):
           # Run evaluation
           pass
       
       def calculate_metrics(self, results):
           # Calculate metrics
           pass
   ```

3. Register the benchmark in `run_evaluation.py`

---

## 📈 Metrics

All benchmarks report:
- **Accuracy**: Percentage of correct predictions
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives that were predicted correctly

---

## 🛠️ Supported Models

- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-3.5-turbo
- **Planned**: Anthropic Claude, Google Gemini, Local models

---

## ✨ New Features

### 🎛️ Configuration Management
- **YAML-based configuration** for all settings
- **Model-specific parameters** and timeouts
- **Benchmark-specific settings** and limits
- **Runtime configuration updates**

### 📊 Advanced Analytics
- **Statistical significance testing** between models
- **Performance trend analysis** over time
- **Correlation analysis** between metrics
- **Comprehensive visualization** with matplotlib and plotly
- **HTML reports** with interactive charts

### 🚀 Performance & Scalability
- **Parallel benchmark execution** for faster evaluation
- **Parallel example processing** for large datasets
- **Progress tracking** with real-time updates
- **Data caching** to avoid redundant processing
- **Configurable worker limits**

### 🛡️ Reliability & Monitoring
- **Comprehensive logging** with structured output
- **Error handling** with retry mechanisms
- **Data validation** with schema checking
- **Graceful degradation** on failures
- **Performance monitoring** and metrics

### 🌐 Web Dashboard
- **Interactive web interface** for running evaluations
- **Real-time result visualization**
- **Benchmark management** and configuration
- **Results comparison** and analysis
- **Leaderboard generation**

### 🧪 Testing & Quality
- **Comprehensive test suite** with unit tests
- **Integration tests** for end-to-end validation
- **Data validation tests** for benchmark integrity
- **Mock testing** for external dependencies

## 📌 Roadmap

- [x] Configuration management system
- [x] Advanced analytics and visualization
- [x] Web dashboard interface
- [x] Parallel processing capabilities
- [x] Comprehensive logging and monitoring
- [x] Data validation and caching
- [x] Statistical analysis and reporting
- [ ] Support for more model providers (Anthropic, Google)
- [ ] Human evaluation integration
- [ ] Benchmark dataset expansion
- [ ] API endpoints for external integration
- [ ] Docker containerization
- [ ] CI/CD pipeline setup

---

## 🤝 Contributing

Contributions welcome! Submit PRs with:
- New benchmarks
- Model adapters
- Evaluation improvements
- Dataset additions

---
