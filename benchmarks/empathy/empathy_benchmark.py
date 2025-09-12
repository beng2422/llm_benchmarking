"""
Empathy Benchmark implementation.

Evaluates whether LLMs can distinguish between empathetic and non-empathetic responses
in therapy-like dialogues.
"""

import json
import os
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from framework.base_benchmark import BaseBenchmark


class EmpathyBenchmark(BaseBenchmark):
    """Benchmark for evaluating empathetic response selection."""
    
    def __init__(self, data_path: str = "benchmarks/empathy/data"):
        super().__init__(
            name="empathy",
            description="Evaluates empathetic response selection in therapy-like dialogues",
            data_path=data_path
        )
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load empathy benchmark data."""
        train_path = os.path.join(self.data_path, "train.json")
        test_path = os.path.join(self.data_path, "test.json")
        
        # Load test data for evaluation
        with open(test_path, 'r') as f:
            test_data = json.load(f)
        
        return test_data
    
    def evaluate_model(self, model_name: str, model_func: callable, **kwargs) -> List[Dict[str, Any]]:
        """
        Run model evaluation on empathy benchmark.
        
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
            patient = example["patient"]
            empathetic_response = example["empathetic"]
            non_empathetic_response = example["non_empathetic"]
            
            # Create prompt for model
            prompt = f"""The patient says: "{patient}"
Choose the more empathetic response:
A: {empathetic_response}
B: {non_empathetic_response}
Answer with A or B only."""
            
            # Get model response
            try:
                response = model_func(prompt, **kwargs)
                response = response.strip()
                
                # Determine prediction
                predicted_response = empathetic_response if "A" in response else non_empathetic_response
                is_correct = predicted_response == empathetic_response
                
                results.append({
                    "id": example["id"],
                    "patient": patient,
                    "empathetic_response": empathetic_response,
                    "non_empathetic_response": non_empathetic_response,
                    "predicted_response": predicted_response,
                    "model_response": response,
                    "is_correct": is_correct
                })
                
            except Exception as e:
                # Handle errors gracefully
                results.append({
                    "id": example["id"],
                    "patient": patient,
                    "empathetic_response": empathetic_response,
                    "non_empathetic_response": non_empathetic_response,
                    "predicted_response": non_empathetic_response,  # Default to wrong answer
                    "model_response": f"ERROR: {str(e)}",
                    "is_correct": False,
                    "error": str(e)
                })
        
        return results
    
    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate empathy benchmark metrics.
        
        Args:
            results: Results from evaluate_model()
            
        Returns:
            Dictionary of metrics
        """
        if not results:
            return {"accuracy": 0.0, "f1_score": 0.0, "precision": 0.0, "recall": 0.0}
        
        # Extract ground truth and predictions
        y_true = [1] * len(results)  # All should be empathetic (class 1)
        y_pred = [1 if r["is_correct"] else 0 for r in results]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "num_examples": len(results),
            "num_correct": sum(y_pred)
        }
