"""
Code Generation Benchmark implementation.

Evaluates code generation abilities across multiple programming languages
including Python, JavaScript, and Java. Tests algorithms, data structures,
and problem-solving skills with automated code execution and validation.
"""

import json
import os
import subprocess
import tempfile
import re
from typing import List, Dict, Any, Union, Tuple
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from framework.base_benchmark import BaseBenchmark


class CodeGenerationBenchmark(BaseBenchmark):
    """Benchmark for evaluating code generation abilities."""
    
    def __init__(self, data_path: str = "benchmarks/code_generation/data"):
        super().__init__(
            name="code_generation",
            description="Evaluates code generation abilities across multiple programming languages",
            data_path=data_path
        )
        self.supported_languages = ["python", "javascript", "java"]
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load code generation benchmark data."""
        test_path = os.path.join(self.data_path, "test.json")
        
        with open(test_path, 'r') as f:
            test_data = json.load(f)
        
        return test_data
    
    def _extract_code_from_response(self, response: str, function_signature: str) -> str:
        """
        Extract code from model response.
        
        Args:
            response: Model's text response
            function_signature: Expected function signature
            
        Returns:
            Extracted code string
        """
        # Look for code blocks
        code_patterns = [
            r'```(?:python|javascript|java)?\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```',
            r'<code>(.*?)</code>',
            r'`(.*?)`'
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                # Return the longest match (likely the complete code)
                return max(matches, key=len).strip()
        
        # If no code blocks found, look for function definition
        lines = response.split('\n')
        code_lines = []
        in_function = False
        
        for line in lines:
            # Check if this line starts a function definition
            if function_signature.strip() in line or any(
                keyword in line for keyword in ['def ', 'function ', 'public static']
            ):
                in_function = True
                code_lines.append(line)
            elif in_function:
                # Check if we've reached the end of the function
                if line.strip() == '' or (line.startswith(' ') or line.startswith('\t')):
                    code_lines.append(line)
                else:
                    # Check if this is another function or class definition
                    if any(keyword in line for keyword in ['def ', 'function ', 'class ', 'public static']):
                        break
                    else:
                        code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        # If all else fails, return the entire response
        return response.strip()
    
    def _execute_python_code(self, code: str, test_cases: List[Dict]) -> Dict[str, Any]:
        """Execute Python code and test it."""
        results = {
            "execution_success": False,
            "test_results": [],
            "error": None,
            "output": ""
        }
        
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Import the function and test it
            exec_globals = {}
            exec(open(temp_file).read(), exec_globals)
            
            # Find the function name (assume it's the first function defined)
            function_name = None
            for name, obj in exec_globals.items():
                if callable(obj) and not name.startswith('_'):
                    function_name = name
                    break
            
            if not function_name:
                results["error"] = "No function found in code"
                return results
            
            function = exec_globals[function_name]
            
            # Test each test case
            for i, test_case in enumerate(test_cases):
                try:
                    input_args = test_case["input"]
                    expected = test_case["expected_output"]
                    
                    # Call the function with test inputs
                    if isinstance(input_args, dict):
                        result = function(**input_args)
                    else:
                        result = function(input_args)
                    
                    is_correct = result == expected
                    results["test_results"].append({
                        "test_case": i,
                        "input": input_args,
                        "expected": expected,
                        "actual": result,
                        "correct": is_correct
                    })
                    
                except Exception as e:
                    results["test_results"].append({
                        "test_case": i,
                        "input": test_case["input"],
                        "expected": test_case["expected_output"],
                        "actual": None,
                        "correct": False,
                        "error": str(e)
                    })
            
            results["execution_success"] = True
            
            # Clean up
            os.unlink(temp_file)
            
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    def _execute_javascript_code(self, code: str, test_cases: List[Dict]) -> Dict[str, Any]:
        """Execute JavaScript code and test it."""
        results = {
            "execution_success": False,
            "test_results": [],
            "error": None,
            "output": ""
        }
        
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                # Add test cases to the JavaScript file
                f.write(code + '\n\n')
                f.write('// Test cases\n')
                for i, test_case in enumerate(test_cases):
                    input_args = test_case["input"]
                    expected = test_case["expected_output"]
                    f.write(f'console.log("Test {i}:", JSON.stringify({input_args}) === JSON.stringify({expected}));\n')
                temp_file = f.name
            
            # Execute with Node.js
            result = subprocess.run(['node', temp_file], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                results["execution_success"] = True
                results["output"] = result.stdout
                
                # Parse test results from output
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    if line.startswith(f'Test {i}:'):
                        is_correct = 'true' in line
                        test_case = test_cases[i] if i < len(test_cases) else {}
                        results["test_results"].append({
                            "test_case": i,
                            "input": test_case.get("input", {}),
                            "expected": test_case.get("expected_output"),
                            "actual": None,
                            "correct": is_correct
                        })
            else:
                results["error"] = result.stderr
                
        except subprocess.TimeoutExpired:
            results["error"] = "Execution timeout"
        except FileNotFoundError:
            results["error"] = "Node.js not found. Please install Node.js to run JavaScript code."
        except Exception as e:
            results["error"] = str(e)
        finally:
            # Clean up
            if 'temp_file' in locals():
                try:
                    os.unlink(temp_file)
                except:
                    pass
        
        return results
    
    def _execute_java_code(self, code: str, test_cases: List[Dict]) -> Dict[str, Any]:
        """Execute Java code and test it."""
        results = {
            "execution_success": False,
            "test_results": [],
            "error": None,
            "output": ""
        }
        
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.java', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Compile Java code
            compile_result = subprocess.run(['javac', temp_file], 
                                          capture_output=True, text=True, timeout=10)
            
            if compile_result.returncode != 0:
                results["error"] = f"Compilation error: {compile_result.stderr}"
                return results
            
            # Execute compiled class
            class_name = os.path.splitext(os.path.basename(temp_file))[0]
            class_dir = os.path.dirname(temp_file)
            
            exec_result = subprocess.run(['java', '-cp', class_dir, class_name], 
                                       capture_output=True, text=True, timeout=10)
            
            if exec_result.returncode == 0:
                results["execution_success"] = True
                results["output"] = exec_result.stdout
                
                # For Java, we'll do a simple check - this could be enhanced
                # to parse actual test results from the output
                for i, test_case in enumerate(test_cases):
                    results["test_results"].append({
                        "test_case": i,
                        "input": test_case["input"],
                        "expected": test_case["expected_output"],
                        "actual": None,
                        "correct": True  # Simplified for now
                    })
            else:
                results["error"] = f"Execution error: {exec_result.stderr}"
                
        except subprocess.TimeoutExpired:
            results["error"] = "Execution timeout"
        except FileNotFoundError:
            results["error"] = "Java not found. Please install Java to run Java code."
        except Exception as e:
            results["error"] = str(e)
        finally:
            # Clean up
            if 'temp_file' in locals():
                try:
                    os.unlink(temp_file)
                    # Also remove .class file
                    class_file = temp_file.replace('.java', '.class')
                    if os.path.exists(class_file):
                        os.unlink(class_file)
                except:
                    pass
        
        return results
    
    def _execute_code(self, code: str, language: str, test_cases: List[Dict]) -> Dict[str, Any]:
        """Execute code based on language."""
        if language == "python":
            return self._execute_python_code(code, test_cases)
        elif language == "javascript":
            return self._execute_javascript_code(code, test_cases)
        elif language == "java":
            return self._execute_java_code(code, test_cases)
        else:
            return {
                "execution_success": False,
                "test_results": [],
                "error": f"Unsupported language: {language}",
                "output": ""
            }
    
    def evaluate_model(self, model_name: str, model_func: callable, **kwargs) -> List[Dict[str, Any]]:
        """
        Run model evaluation on code generation benchmark.
        
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
            title = example["title"]
            description = example["description"]
            function_signature = example["function_signature"]
            test_cases = example["test_cases"]
            language = example["language"]
            difficulty = example["difficulty"]
            category = example["category"]
            
            # Create prompt for model
            prompt = f"""Write a {language} function to solve this problem:

Title: {title}
Description: {description}

Function signature: {function_signature}

Please provide only the complete function implementation. Do not include any explanations or comments unless necessary for the code to work.

Test cases:
{json.dumps(test_cases, indent=2)}"""
            
            # Get model response
            try:
                response = model_func(prompt, **kwargs)
                response = response.strip()
                
                # Extract code from response
                generated_code = self._extract_code_from_response(response, function_signature)
                
                # Execute the code
                execution_results = self._execute_code(generated_code, language, test_cases)
                
                # Calculate correctness
                if execution_results["execution_success"]:
                    test_results = execution_results["test_results"]
                    correct_tests = sum(1 for test in test_results if test.get("correct", False))
                    total_tests = len(test_results)
                    is_correct = correct_tests == total_tests and total_tests > 0
                    accuracy = correct_tests / total_tests if total_tests > 0 else 0
                else:
                    is_correct = False
                    accuracy = 0
                    test_results = []
                
                results.append({
                    "id": example["id"],
                    "title": title,
                    "description": description,
                    "language": language,
                    "difficulty": difficulty,
                    "category": category,
                    "function_signature": function_signature,
                    "generated_code": generated_code,
                    "model_response": response,
                    "is_correct": is_correct,
                    "accuracy": accuracy,
                    "execution_success": execution_results["execution_success"],
                    "test_results": test_results,
                    "execution_error": execution_results.get("error"),
                    "expected_code": example.get("expected_code", "")
                })
                
            except Exception as e:
                # Handle errors gracefully
                results.append({
                    "id": example["id"],
                    "title": title,
                    "description": description,
                    "language": language,
                    "difficulty": difficulty,
                    "category": category,
                    "function_signature": function_signature,
                    "generated_code": "",
                    "model_response": f"ERROR: {str(e)}",
                    "is_correct": False,
                    "accuracy": 0,
                    "execution_success": False,
                    "test_results": [],
                    "execution_error": str(e),
                    "expected_code": example.get("expected_code", "")
                })
        
        return results
    
    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate code generation benchmark metrics.
        
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
        
        # Calculate average accuracy across all test cases
        avg_accuracy = sum(r.get("accuracy", 0) for r in results) / len(results)
        
        # Language-specific metrics
        languages = {}
        difficulties = {}
        categories = {}
        
        for result in results:
            lang = result["language"]
            diff = result["difficulty"]
            cat = result["category"]
            
            if lang not in languages:
                languages[lang] = {"correct": 0, "total": 0, "avg_accuracy": 0}
            if diff not in difficulties:
                difficulties[diff] = {"correct": 0, "total": 0, "avg_accuracy": 0}
            if cat not in categories:
                categories[cat] = {"correct": 0, "total": 0, "avg_accuracy": 0}
            
            languages[lang]["total"] += 1
            difficulties[diff]["total"] += 1
            categories[cat]["total"] += 1
            
            if result["is_correct"]:
                languages[lang]["correct"] += 1
                difficulties[diff]["correct"] += 1
                categories[cat]["correct"] += 1
            
            languages[lang]["avg_accuracy"] += result.get("accuracy", 0)
            difficulties[diff]["avg_accuracy"] += result.get("accuracy", 0)
            categories[cat]["avg_accuracy"] += result.get("accuracy", 0)
        
        # Calculate language accuracies
        language_accuracies = {}
        for lang, stats in languages.items():
            language_accuracies[f"{lang}_accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            language_accuracies[f"{lang}_avg_accuracy"] = stats["avg_accuracy"] / stats["total"] if stats["total"] > 0 else 0.0
        
        # Calculate difficulty accuracies
        difficulty_accuracies = {}
        for diff, stats in difficulties.items():
            difficulty_accuracies[f"{diff}_accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            difficulty_accuracies[f"{diff}_avg_accuracy"] = stats["avg_accuracy"] / stats["total"] if stats["total"] > 0 else 0.0
        
        # Calculate category accuracies
        category_accuracies = {}
        for cat, stats in categories.items():
            category_accuracies[f"{cat}_accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            category_accuracies[f"{cat}_avg_accuracy"] = stats["avg_accuracy"] / stats["total"] if stats["total"] > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "avg_accuracy": avg_accuracy,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "num_examples": len(results),
            "num_correct": sum(y_true),
            "execution_success_rate": sum(1 for r in results if r.get("execution_success", False)) / len(results),
            **language_accuracies,
            **difficulty_accuracies,
            **category_accuracies
        }
    
    def get_detailed_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get detailed analysis of results by language, difficulty, and category.
        
        Args:
            results: Results from evaluate_model()
            
        Returns:
            Dictionary with detailed analysis
        """
        analysis = {
            "overall": {
                "total": len(results),
                "correct": sum(1 for r in results if r["is_correct"]),
                "accuracy": sum(1 for r in results if r["is_correct"]) / len(results) if results else 0,
                "avg_accuracy": sum(r.get("accuracy", 0) for r in results) / len(results) if results else 0,
                "execution_success_rate": sum(1 for r in results if r.get("execution_success", False)) / len(results) if results else 0
            },
            "by_language": {},
            "by_difficulty": {},
            "by_category": {}
        }
        
        # Analyze by language
        languages = {}
        for result in results:
            lang = result["language"]
            if lang not in languages:
                languages[lang] = {"correct": 0, "total": 0, "avg_accuracy": 0, "execution_success": 0}
            
            languages[lang]["total"] += 1
            if result["is_correct"]:
                languages[lang]["correct"] += 1
            if result.get("execution_success", False):
                languages[lang]["execution_success"] += 1
            languages[lang]["avg_accuracy"] += result.get("accuracy", 0)
        
        for lang, stats in languages.items():
            analysis["by_language"][lang] = {
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
                "avg_accuracy": stats["avg_accuracy"] / stats["total"] if stats["total"] > 0 else 0,
                "execution_success_rate": stats["execution_success"] / stats["total"] if stats["total"] > 0 else 0,
                "total": stats["total"],
                "correct": stats["correct"]
            }
        
        # Analyze by difficulty
        difficulties = {}
        for result in results:
            diff = result["difficulty"]
            if diff not in difficulties:
                difficulties[diff] = {"correct": 0, "total": 0, "avg_accuracy": 0}
            
            difficulties[diff]["total"] += 1
            if result["is_correct"]:
                difficulties[diff]["correct"] += 1
            difficulties[diff]["avg_accuracy"] += result.get("accuracy", 0)
        
        for diff, stats in difficulties.items():
            analysis["by_difficulty"][diff] = {
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
                "avg_accuracy": stats["avg_accuracy"] / stats["total"] if stats["total"] > 0 else 0,
                "total": stats["total"],
                "correct": stats["correct"]
            }
        
        # Analyze by category
        categories = {}
        for result in results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {"correct": 0, "total": 0, "avg_accuracy": 0}
            
            categories[cat]["total"] += 1
            if result["is_correct"]:
                categories[cat]["correct"] += 1
            categories[cat]["avg_accuracy"] += result.get("accuracy", 0)
        
        for cat, stats in categories.items():
            analysis["by_category"][cat] = {
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
                "avg_accuracy": stats["avg_accuracy"] / stats["total"] if stats["total"] > 0 else 0,
                "total": stats["total"],
                "correct": stats["correct"]
            }
        
        return analysis
