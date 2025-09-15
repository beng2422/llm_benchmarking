# LLM Benchmarking Suite

**LLM Benchmarking Suite** is a comprehensive framework for evaluating large language models across multiple benchmarks. This project provides:

- **Multiple Benchmarks**: Empathy, Commonsense Reasoning, Math Reasoning, and more
- **Unified Evaluation Framework**: Consistent evaluation across all benchmarks
- **Model Adapters**: Support for OpenAI, Anthropic, and other providers
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
│   └── math_reasoning/          # Mathematical reasoning
│       ├── data/
│       └── math_reasoning_benchmark.py
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

3. Run all benchmarks:
   ```bash
   python run_evaluation.py --model gpt-4o-mini
   ```

4. Run specific benchmarks:
   ```bash
   python run_evaluation.py --benchmarks empathy commonsense math_reasoning --model gpt-4o-mini
   ```

5. List available benchmarks:
   ```bash
   python run_evaluation.py --list-benchmarks
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

## 📌 Roadmap

- [ ] Add more benchmarks (math, coding, safety)
- [ ] Support for more model providers
- [ ] Web interface for running evaluations
- [ ] Benchmark dataset expansion
- [ ] Human evaluation integration

---

## 🤝 Contributing

Contributions welcome! Submit PRs with:
- New benchmarks
- Model adapters
- Evaluation improvements
- Dataset additions

---
