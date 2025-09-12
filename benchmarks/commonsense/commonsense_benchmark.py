"""
Commonsense Benchmark implementation.

Evaluates whether LLMs can make basic commonsense reasoning judgments.
"""

import json
import os
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from framework.base_benchmark import BaseBenchmark


class CommonsenseBenchmark(BaseBenchmark):
    """Benchmark for evaluating commonsense reasoning."""
    
    def __init__(self, data_path: str = "benchmarks/commonsense/data"):
        super().__init__(
            name="commonsense",
            description="Evaluates basic commonsense reasoning abilities",
            data_path=data_path
        )
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load commonsense benchmark data."""
        test_path = os.path.join(self.data_path, "test.json")
        
        with open(test_path, 'r') as f:
            test_data = json.load(f)
        
        return test_data
    
    def evaluate_model(self, model_name: str, model_func: callable, **kwargs) -> List[Dict[str, Any]]:
        """
        Run model evaluation on commonsense benchmark.
        
        Args:
            model_name: Name of the model
            model_func: Function that takes a prompt and returns a response
            **kwargs: Additional model parameters
            
        Returns:
            List of evaluation results
        """
        data = self.get_data()
        results = []
        
        for example in data:
            question = example["question"]
            options = example["options"]
            correct_answer = example["correct_answer"]
            
            # Create prompt for model
            prompt = f"""Question: {question}

Options:
A: {options['A']}
B: {options['B']}
C: {options['C']}
D: {options['D']}

Answer with A, B, C, or D only."""
            
            # Get model response
            try:
                response = model_func(prompt, **kwargs)
                response = response.strip().upper()
                
                # Extract answer (take first letter if multiple characters)
                predicted_answer = response[0] if response else "A"
                is_correct = predicted_answer == correct_answer
                
                results.append({
                    "id": example["id"],
                    "question": question,
                    "options": options,
                    "correct_answer": correct_answer,
                    "predicted_answer": predicted_answer,
                    "model_response": response,
                    "is_correct": is_correct,
                    "explanation": example.get("explanation", "")
                })
                
            except Exception as e:
                # Handle errors gracefully
                results.append({
                    "id": example["id"],
                    "question": question,
                    "options": options,
                    "correct_answer": correct_answer,
                    "predicted_answer": "A",  # Default to first option
                    "model_response": f"ERROR: {str(e)}",
                    "is_correct": False,
                    "error": str(e),
                    "explanation": example.get("explanation", "")
                })
        
        return results
    
    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate commonsense benchmark metrics.
        
        Args:
            results: Results from evaluate_model()
            
        Returns:
            Dictionary of metrics
        """
        if not results:
            return {"accuracy": 0.0, "f1_score": 0.0, "precision": 0.0, "recall": 0.0}
        
        # Extract ground truth and predictions
        y_true = [1 if r["is_correct"] else 0 for r in results]
        y_pred = [1] * len(results)  # All predictions are "correct" class
        
        # Calculate metrics
        accuracy = sum(y_true) / len(y_true)  # Simple accuracy
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "num_examples": len(results),
            "num_correct": sum(y_true)
        }
