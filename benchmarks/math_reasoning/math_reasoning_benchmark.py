"""
Math Reasoning Benchmark implementation.

Evaluates mathematical problem-solving abilities across various domains including
arithmetic, algebra, geometry, word problems, fractions, percentages, logic,
sequences, probability, and exponents.
"""

import json
import os
import re
from typing import List, Dict, Any, Union
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from framework.base_benchmark import BaseBenchmark


class MathReasoningBenchmark(BaseBenchmark):
    """Benchmark for evaluating mathematical reasoning abilities."""
    
    def __init__(self, data_path: str = "benchmarks/math_reasoning/data"):
        super().__init__(
            name="math_reasoning",
            description="Evaluates mathematical problem-solving across multiple domains",
            data_path=data_path
        )
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load math reasoning benchmark data."""
        test_path = os.path.join(self.data_path, "test.json")
        
        with open(test_path, 'r') as f:
            test_data = json.load(f)
        
        return test_data
    
    def _extract_numerical_answer(self, response: str) -> Union[float, str, None]:
        """
        Extract numerical answer from model response.
        
        Args:
            response: Model's text response
            
        Returns:
            Extracted numerical value, text answer, or None if not found
        """
        response = response.strip()
        
        # Handle yes/no questions
        if any(word in response.lower() for word in ['yes', 'no']):
            return 'yes' if 'yes' in response.lower() else 'no'
        
        # Look for numerical patterns
        # Pattern 1: Direct number (including decimals and negatives)
        number_pattern = r'-?\d+\.?\d*'
        numbers = re.findall(number_pattern, response)
        
        if numbers:
            # Try to convert the last number found (often the final answer)
            try:
                return float(numbers[-1])
            except ValueError:
                pass
        
        # Pattern 2: Look for "answer is X" or "= X" patterns
        answer_patterns = [
            r'answer is\s*:?\s*([-+]?\d+\.?\d*)',
            r'=\s*([-+]?\d+\.?\d*)',
            r'result is\s*:?\s*([-+]?\d+\.?\d*)',
            r'equals\s*:?\s*([-+]?\d+\.?\d*)',
            r'is\s*([-+]?\d+\.?\d*)'
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        # If no numerical answer found, return the response as-is for text answers
        return response if response else None
    
    def _is_correct_answer(self, predicted: Union[float, str, None], correct: Union[float, str]) -> bool:
        """
        Check if the predicted answer is correct.
        
        Args:
            predicted: Model's predicted answer
            correct: Correct answer from dataset
            
        Returns:
            True if answer is correct, False otherwise
        """
        if predicted is None:
            return False
        
        # Handle text answers (yes/no)
        if isinstance(correct, str) and isinstance(predicted, str):
            return predicted.lower().strip() == correct.lower().strip()
        
        # Handle numerical answers
        if isinstance(correct, (int, float)) and isinstance(predicted, (int, float)):
            # Allow small floating point differences
            return abs(predicted - correct) < 1e-6
        
        # Mixed types - try to convert predicted to correct type
        try:
            if isinstance(correct, str):
                predicted_str = str(predicted).lower().strip()
                return predicted_str == correct.lower().strip()
            elif isinstance(correct, (int, float)):
                predicted_num = float(predicted)
                return abs(predicted_num - correct) < 1e-6
        except (ValueError, TypeError):
            pass
        
        return False
    
    def evaluate_model(self, model_name: str, model_func: callable, **kwargs) -> List[Dict[str, Any]]:
        """
        Run model evaluation on math reasoning benchmark.
        
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
            correct_answer = example["answer"]
            category = example["category"]
            difficulty = example["difficulty"]
            explanation = example.get("explanation", "")
            
            # Create prompt for model
            prompt = f"""Solve this math problem step by step:

{question}

Please provide your answer clearly. If it's a numerical answer, just give the number. If it's a yes/no question, answer with yes or no."""
            
            # Get model response
            try:
                response = model_func(prompt, **kwargs)
                response = response.strip()
                
                # Extract answer from response
                predicted_answer = self._extract_numerical_answer(response)
                is_correct = self._is_correct_answer(predicted_answer, correct_answer)
                
                results.append({
                    "id": example["id"],
                    "question": question,
                    "correct_answer": correct_answer,
                    "predicted_answer": predicted_answer,
                    "model_response": response,
                    "is_correct": is_correct,
                    "category": category,
                    "difficulty": difficulty,
                    "explanation": explanation
                })
                
            except Exception as e:
                # Handle errors gracefully
                results.append({
                    "id": example["id"],
                    "question": question,
                    "correct_answer": correct_answer,
                    "predicted_answer": None,
                    "model_response": f"ERROR: {str(e)}",
                    "is_correct": False,
                    "category": category,
                    "difficulty": difficulty,
                    "error": str(e),
                    "explanation": explanation
                })
        
        return results
    
    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate math reasoning benchmark metrics.
        
        Args:
            results: Results from evaluate_model()
            
        Returns:
            Dictionary of metrics
        """
        if not results:
            return {"accuracy": 0.0, "f1_score": 0.0, "precision": 0.0, "recall": 0.0}
        
        # Overall metrics
        y_true = [1 if r["is_correct"] else 0 for r in results]
        y_pred = [1] * len(results)  # All predictions are "correct" class
        
        accuracy = sum(y_true) / len(y_true)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # Category-specific metrics
        categories = {}
        difficulties = {}
        
        for result in results:
            category = result["category"]
            difficulty = result["difficulty"]
            
            if category not in categories:
                categories[category] = {"correct": 0, "total": 0}
            if difficulty not in difficulties:
                difficulties[difficulty] = {"correct": 0, "total": 0}
            
            categories[category]["total"] += 1
            difficulties[difficulty]["total"] += 1
            
            if result["is_correct"]:
                categories[category]["correct"] += 1
                difficulties[difficulty]["correct"] += 1
        
        # Calculate category accuracies
        category_accuracies = {}
        for cat, stats in categories.items():
            category_accuracies[f"{cat}_accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        
        # Calculate difficulty accuracies
        difficulty_accuracies = {}
        for diff, stats in difficulties.items():
            difficulty_accuracies[f"{diff}_accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "num_examples": len(results),
            "num_correct": sum(y_true),
            **category_accuracies,
            **difficulty_accuracies
        }
    
    def get_detailed_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get detailed analysis of results by category and difficulty.
        
        Args:
            results: Results from evaluate_model()
            
        Returns:
            Dictionary with detailed analysis
        """
        analysis = {
            "overall": {
                "total": len(results),
                "correct": sum(1 for r in results if r["is_correct"]),
                "accuracy": sum(1 for r in results if r["is_correct"]) / len(results) if results else 0
            },
            "by_category": {},
            "by_difficulty": {}
        }
        
        # Analyze by category
        categories = {}
        for result in results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {"correct": 0, "total": 0, "examples": []}
            
            categories[cat]["total"] += 1
            if result["is_correct"]:
                categories[cat]["correct"] += 1
            categories[cat]["examples"].append({
                "id": result["id"],
                "question": result["question"],
                "correct": result["is_correct"],
                "predicted": result["predicted_answer"],
                "expected": result["correct_answer"]
            })
        
        for cat, stats in categories.items():
            analysis["by_category"][cat] = {
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
                "total": stats["total"],
                "correct": stats["correct"]
            }
        
        # Analyze by difficulty
        difficulties = {}
        for result in results:
            diff = result["difficulty"]
            if diff not in difficulties:
                difficulties[diff] = {"correct": 0, "total": 0}
            
            difficulties[diff]["total"] += 1
            if result["is_correct"]:
                difficulties[diff]["correct"] += 1
        
        for diff, stats in difficulties.items():
            analysis["by_difficulty"][diff] = {
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
                "total": stats["total"],
                "correct": stats["correct"]
            }
        
        return analysis
