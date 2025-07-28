"""
LiveCodeBench Evaluation Pipeline for HRM

This module provides evaluation capabilities for HRM models on LiveCodeBench tasks.
It implements pass@k metrics, code execution sandboxing, and the 4 evaluation scenarios.
"""

import os
import sys
import json
import subprocess
import tempfile
import traceback
import multiprocessing as mp
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import time

import numpy as np
import torch
from tqdm import tqdm

# LiveCodeBench imports
sys.path.append('/Users/micahoates/Developer/x/HRM-AstralDrift/LiveCodeBench')
from lcb_runner.benchmarks import (
    CodeGenerationProblem,
    TestOutputPredictionProblem, 
    CodeExecutionProblem
)
from lcb_runner.evaluation.pass_k_utils import estimate_pass_at_k, compute_metrics_from_results
from lcb_runner.evaluation.testing_util import run_test
from lcb_runner.utils.scenarios import Scenario

# HRM imports
from models.code_generation.hrm_code_model import HRMCodeGenerationModel
from models.code_generation.input_processor import CodeGenerationInput, CodeGenerationTask, ProgrammingLanguage
from models.code_generation.output_generator import CodeOutputGenerator

@dataclass
class EvaluationResult:
    """Result of evaluating a single problem"""
    problem_id: str
    scenario: Scenario
    generated_codes: List[str]
    test_results: List[bool]
    pass_at_1: float
    pass_at_5: float
    execution_time: float
    error_messages: List[str]
    metadata: Dict[str, Any]

@dataclass 
class EvaluationConfig:
    """Configuration for LiveCodeBench evaluation"""
    # Model configuration
    model_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generation configuration  
    num_samples: int = 5  # For pass@k evaluation
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    
    # Evaluation configuration
    timeout_seconds: int = 10  # Per test case
    max_workers: int = 4
    scenarios: List[Scenario] = None  # If None, evaluate all
    
    # Contamination filtering
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Output configuration
    save_detailed_results: bool = True
    output_dir: str = "evaluation_results"

class CodeExecutionSandbox:
    """Secure code execution environment"""
    
    def __init__(self, timeout_seconds: int = 10):
        self.timeout = timeout_seconds
        
    def execute_code(self, code: str, test_input: str = "", expected_output: str = "") -> Tuple[bool, str, str]:
        """
        Execute code safely and return (success, output, error_message)
        """
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Write code to temporary file
                f.write(code)
                f.flush()
                temp_file = f.name
            
            try:
                # Execute with timeout
                if test_input:
                    # Provide input via stdin
                    process = subprocess.Popen(
                        ['python', temp_file],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=self.timeout
                    )
                    stdout, stderr = process.communicate(input=test_input)
                else:
                    # No input needed
                    process = subprocess.Popen(
                        ['python', temp_file],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=self.timeout
                    )
                    stdout, stderr = process.communicate()
                
                # Check if execution was successful
                if process.returncode == 0:
                    # Clean output for comparison
                    actual_output = stdout.strip()
                    expected_clean = expected_output.strip() if expected_output else ""
                    
                    # Compare outputs
                    if expected_output:
                        success = actual_output == expected_clean
                        if not success:
                            error_msg = f"Output mismatch. Expected: {expected_clean}, Got: {actual_output}"
                        else:
                            error_msg = ""
                    else:
                        success = True
                        error_msg = ""
                    
                    return success, actual_output, error_msg
                else:
                    return False, stdout, stderr
                    
            except subprocess.TimeoutExpired:
                return False, "", f"Execution timed out after {self.timeout} seconds"
            except Exception as e:
                return False, "", f"Execution error: {str(e)}"
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
        except Exception as e:
            return False, "", f"Setup error: {str(e)}"
    
    def execute_function_test(self, code: str, function_name: str, test_input: Any, expected_output: Any) -> Tuple[bool, Any, str]:
        """
        Execute a specific function with test input
        """
        try:
            # Create test code
            test_code = f"""
{code}

import json
try:
    # Parse test input
    test_input = json.loads('''{json.dumps(test_input)}''')
    if isinstance(test_input, list):
        result = {function_name}(*test_input)
    else:
        result = {function_name}(test_input)
    print(json.dumps(result))
except Exception as e:
    print(f"ERROR: {{e}}")
    exit(1)
"""
            
            success, output, error = self.execute_code(test_code)
            
            if success and not error:
                try:
                    actual_result = json.loads(output)
                    success = actual_result == expected_output
                    if not success:
                        error = f"Result mismatch. Expected: {expected_output}, Got: {actual_result}"
                    return success, actual_result, error
                except json.JSONDecodeError:
                    return False, output, "Failed to parse function output as JSON"
            else:
                return False, None, error or output
                
        except Exception as e:
            return False, None, f"Function test error: {str(e)}"

class LiveCodeBenchEvaluator:
    """Main evaluator for LiveCodeBench problems"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.sandbox = CodeExecutionSandbox(config.timeout_seconds)
        
        # Load HRM model
        print(f"Loading HRM model from {config.model_path}")
        self.model = HRMCodeGenerationModel.load_pretrained(config.model_path)
        self.model.to(config.device)
        self.model.eval()
        
        # Initialize output generator
        self.output_generator = CodeOutputGenerator(
            vocab_size=self.model.config.vocab_size,
            max_length=config.max_new_tokens
        )
        
        # Ensure output directory exists
        os.makedirs(config.output_dir, exist_ok=True)
        
    def evaluate_code_generation(self, problems: List[CodeGenerationProblem]) -> Dict[str, EvaluationResult]:
        """Evaluate code generation problems"""
        print(f"Evaluating {len(problems)} code generation problems...")
        
        results = {}
        for problem in tqdm(problems, desc="Code Generation"):
            result = self._evaluate_single_generation(problem)
            results[problem.question_id] = result
            
        return results
    
    def evaluate_self_repair(self, problems: List[CodeGenerationProblem]) -> Dict[str, EvaluationResult]:
        """Evaluate self-repair by creating broken code and asking model to fix it"""
        print(f"Evaluating {len(problems)} self-repair problems...")
        
        results = {}
        for problem in tqdm(problems, desc="Self Repair"):
            result = self._evaluate_single_repair(problem)
            results[f"repair_{problem.question_id}"] = result
            
        return results
    
    def evaluate_test_prediction(self, problems: List[TestOutputPredictionProblem]) -> Dict[str, EvaluationResult]:
        """Evaluate test output prediction problems"""
        print(f"Evaluating {len(problems)} test prediction problems...")
        
        results = {}
        for problem in tqdm(problems, desc="Test Prediction"):
            result = self._evaluate_single_test_prediction(problem)
            results[f"test_{problem.question_id}_{problem.test_id}"] = result
            
        return results
    
    def evaluate_code_execution(self, problems: List[CodeExecutionProblem]) -> Dict[str, EvaluationResult]:
        """Evaluate code execution problems"""
        print(f"Evaluating {len(problems)} code execution problems...")
        
        results = {}
        for problem in tqdm(problems, desc="Code Execution"):
            result = self._evaluate_single_execution(problem)
            results[problem.id] = result
            
        return results
    
    def _evaluate_single_generation(self, problem: CodeGenerationProblem) -> EvaluationResult:
        """Evaluate a single code generation problem"""
        start_time = time.time()
        
        # Create input for the model
        input_text = self._format_generation_input(problem)
        code_input = CodeGenerationInput(
            problem_description=input_text,
            language=ProgrammingLanguage.PYTHON,
            task_type=CodeGenerationTask.GENERATION
        )
        
        # Generate multiple code samples
        generated_codes = []
        error_messages = []
        
        for i in range(self.config.num_samples):
            try:
                with torch.no_grad():
                    code = self.model.generate(
                        code_input,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p
                    )
                generated_codes.append(code)
            except Exception as e:
                error_messages.append(f"Generation {i}: {str(e)}")
                generated_codes.append("")  # Empty code for failed generation
        
        # Test each generated code
        test_results = []
        for code in generated_codes:
            if not code.strip():
                test_results.append(False)
                continue
                
            # Test against all test cases
            code_passes = True
            for test_case in problem.public_test_cases + problem.private_test_cases:
                success = self._test_code_against_case(code, test_case, problem.metadata)
                if not success:
                    code_passes = False
                    break
            
            test_results.append(code_passes)
        
        # Calculate pass@k metrics
        num_correct = sum(test_results)
        pass_at_1 = estimate_pass_at_k([self.config.num_samples], [num_correct], 1)[0]
        pass_at_5 = estimate_pass_at_k([self.config.num_samples], [num_correct], 5)[0] if self.config.num_samples >= 5 else 0.0
        
        execution_time = time.time() - start_time
        
        return EvaluationResult(
            problem_id=problem.question_id,
            scenario=Scenario.codegeneration,
            generated_codes=generated_codes,
            test_results=test_results,
            pass_at_1=pass_at_1,
            pass_at_5=pass_at_5,
            execution_time=execution_time,
            error_messages=error_messages,
            metadata={
                "platform": problem.platform.value,
                "difficulty": problem.difficulty.value,
                "num_test_cases": len(problem.public_test_cases) + len(problem.private_test_cases),
                "has_starter_code": bool(problem.starter_code)
            }
        )
    
    def _evaluate_single_repair(self, problem: CodeGenerationProblem) -> EvaluationResult:
        """Evaluate a single self-repair problem"""
        start_time = time.time()
        
        # Create broken code
        broken_code, error_msg = self._create_broken_code(problem)
        
        # Create repair input
        input_text = self._format_repair_input(problem, broken_code, error_msg)
        code_input = CodeGenerationInput(
            problem_description=input_text,
            language=ProgrammingLanguage.PYTHON,
            task_type=CodeGenerationTask.REPAIR,
            existing_code=broken_code,
            error_message=error_msg
        )
        
        # Generate repaired code
        generated_codes = []
        error_messages = []
        
        for i in range(self.config.num_samples):
            try:
                with torch.no_grad():
                    code = self.model.generate(
                        code_input,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p
                    )
                generated_codes.append(code)
            except Exception as e:
                error_messages.append(f"Repair {i}: {str(e)}")
                generated_codes.append("")
        
        # Test repaired code
        test_results = []
        for code in generated_codes:
            if not code.strip():
                test_results.append(False)
                continue
                
            # First check if code runs without syntax errors
            success, _, error = self.sandbox.execute_code(code)
            if not success:
                test_results.append(False)
                continue
            
            # Then test against problem test cases
            code_passes = True
            for test_case in problem.public_test_cases + problem.private_test_cases:
                success = self._test_code_against_case(code, test_case, problem.metadata)
                if not success:
                    code_passes = False
                    break
            
            test_results.append(code_passes)
        
        # Calculate metrics
        num_correct = sum(test_results)
        pass_at_1 = estimate_pass_at_k([self.config.num_samples], [num_correct], 1)[0]
        pass_at_5 = estimate_pass_at_k([self.config.num_samples], [num_correct], 5)[0] if self.config.num_samples >= 5 else 0.0
        
        execution_time = time.time() - start_time
        
        return EvaluationResult(
            problem_id=f"repair_{problem.question_id}",
            scenario=Scenario.selfrepair,
            generated_codes=generated_codes,
            test_results=test_results,
            pass_at_1=pass_at_1,
            pass_at_5=pass_at_5,
            execution_time=execution_time,
            error_messages=error_messages,
            metadata={
                "original_problem": problem.question_id,
                "error_type": error_msg.split(":")[0] if ":" in error_msg else "SyntaxError",
                "broken_code": broken_code
            }
        )
    
    def _evaluate_single_test_prediction(self, problem: TestOutputPredictionProblem) -> EvaluationResult:
        """Evaluate a single test output prediction problem"""
        start_time = time.time()
        
        # Create input for prediction
        input_text = self._format_test_prediction_input(problem)
        code_input = CodeGenerationInput(
            problem_description=input_text,
            language=ProgrammingLanguage.PYTHON,
            task_type=CodeGenerationTask.TEST_PREDICTION
        )
        
        # Generate predictions
        generated_outputs = []
        error_messages = []
        
        for i in range(self.config.num_samples):
            try:
                with torch.no_grad():
                    prediction = self.model.generate(
                        code_input,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p
                    )
                generated_outputs.append(prediction.strip())
            except Exception as e:
                error_messages.append(f"Prediction {i}: {str(e)}")
                generated_outputs.append("")
        
        # Check predictions against expected output
        expected_output = problem.test[0].output
        test_results = [pred == expected_output for pred in generated_outputs]
        
        # Calculate metrics
        num_correct = sum(test_results)
        pass_at_1 = estimate_pass_at_k([self.config.num_samples], [num_correct], 1)[0]
        pass_at_5 = estimate_pass_at_k([self.config.num_samples], [num_correct], 5)[0] if self.config.num_samples >= 5 else 0.0
        
        execution_time = time.time() - start_time
        
        return EvaluationResult(
            problem_id=f"test_{problem.question_id}_{problem.test_id}",
            scenario=Scenario.testoutputprediction,
            generated_codes=generated_outputs,  # Using for outputs
            test_results=test_results,
            pass_at_1=pass_at_1,
            pass_at_5=pass_at_5,
            execution_time=execution_time,
            error_messages=error_messages,
            metadata={
                "function_name": problem.function_name,
                "test_input": problem.test[0].input,
                "expected_output": expected_output
            }
        )
    
    def _evaluate_single_execution(self, problem: CodeExecutionProblem) -> EvaluationResult:
        """Evaluate a single code execution problem"""
        start_time = time.time()
        
        # Create input for execution prediction
        input_text = self._format_execution_input(problem)
        code_input = CodeGenerationInput(
            problem_description=input_text,
            language=ProgrammingLanguage.PYTHON,
            task_type=CodeGenerationTask.GENERATION  # Execution is like generation
        )
        
        # Generate execution predictions
        generated_outputs = []
        error_messages = []
        
        for i in range(self.config.num_samples):
            try:
                with torch.no_grad():
                    prediction = self.model.generate(
                        code_input,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p
                    )
                generated_outputs.append(prediction.strip())
            except Exception as e:
                error_messages.append(f"Execution {i}: {str(e)}")
                generated_outputs.append("")
        
        # Check predictions against expected output
        expected_output = problem.output.strip()
        test_results = [pred == expected_output for pred in generated_outputs]
        
        # Calculate metrics
        num_correct = sum(test_results)
        pass_at_1 = estimate_pass_at_k([self.config.num_samples], [num_correct], 1)[0]
        pass_at_5 = estimate_pass_at_k([self.config.num_samples], [num_correct], 5)[0] if self.config.num_samples >= 5 else 0.0
        
        execution_time = time.time() - start_time
        
        return EvaluationResult(
            problem_id=problem.id,
            scenario=Scenario.codeexecution,
            generated_codes=generated_outputs,
            test_results=test_results,
            pass_at_1=pass_at_1,
            pass_at_5=pass_at_5,
            execution_time=execution_time,
            error_messages=error_messages,
            metadata={
                "function_name": problem.function_name,
                "input": problem.input,
                "expected_output": expected_output,
                "num_steps": problem.numsteps
            }
        )
    
    def _test_code_against_case(self, code: str, test_case, problem_metadata: Dict) -> bool:
        """Test generated code against a specific test case"""
        try:
            func_name = problem_metadata.get("func_name", "")
            
            if func_name and hasattr(test_case, 'testtype') and test_case.testtype.value == "functional":
                # Functional test
                try:
                    test_input = json.loads(test_case.input)
                    expected_output = json.loads(test_case.output)
                except:
                    # Fallback to string comparison
                    test_input = test_case.input
                    expected_output = test_case.output
                
                success, _, _ = self.sandbox.execute_function_test(
                    code, func_name, test_input, expected_output
                )
                return success
            else:
                # STDIN test
                success, actual_output, _ = self.sandbox.execute_code(
                    code, test_case.input, test_case.output
                )
                return success
                
        except Exception as e:
            return False
    
    def _format_generation_input(self, problem: CodeGenerationProblem) -> str:
        """Format code generation input"""
        parts = [f"Problem: {problem.question_content}"]
        
        if problem.starter_code:
            parts.append(f"Starter Code:\n{problem.starter_code}")
        
        if problem.public_test_cases:
            test_examples = []
            for i, test in enumerate(problem.public_test_cases[:3]):
                test_examples.append(f"Example {i+1}:\nInput: {test.input}\nOutput: {test.output}")
            parts.append("Test Cases:\n" + "\n\n".join(test_examples))
        
        return "\n\n".join(parts)
    
    def _format_repair_input(self, problem: CodeGenerationProblem, broken_code: str, error_msg: str) -> str:
        """Format self-repair input"""
        parts = [
            f"Problem: {problem.question_content}",
            f"Broken Code:\n{broken_code}",
            f"Error Message: {error_msg}",
            "Fix the code to solve the problem correctly."
        ]
        return "\n\n".join(parts)
    
    def _format_test_prediction_input(self, problem: TestOutputPredictionProblem) -> str:
        """Format test prediction input"""
        parts = [
            f"Problem: {problem.question_content}",
            f"Starter Code:\n{problem.starter_code}",
            f"Function: {problem.function_name}",
            f"Test Input: {problem.test[0].input}",
            "Predict the output:"
        ]
        return "\n\n".join(parts)
    
    def _format_execution_input(self, problem: CodeExecutionProblem) -> str:
        """Format code execution input"""
        parts = [
            f"Code:\n{problem.code}",
            f"Input: {problem.input}",
            "Execute the code with the given input and provide the output:"
        ]
        return "\n\n".join(parts)
    
    def _create_broken_code(self, problem: CodeGenerationProblem) -> Tuple[str, str]:
        """Create broken code for self-repair evaluation"""
        import random
        
        if not problem.starter_code:
            func_name = problem.metadata.get("func_name", "solution")
            broken_variants = [
                (f"def {func_name}(:\n    pass", "SyntaxError: invalid syntax"),
                (f"def {func_name}():\n    return", "SyntaxError: invalid syntax"),
                (f"def {func_name}():\n    x = 1 +\n    return x", "SyntaxError: invalid syntax"),
            ]
            return random.choice(broken_variants)
        
        code = problem.starter_code
        error_types = [
            (lambda c: c.replace("(", "", 1), "SyntaxError: invalid syntax"),
            (lambda c: c.replace(":", "", 1), "SyntaxError: invalid syntax"),
            (lambda c: c.replace("    ", "  ", 1), "IndentationError: unindent does not match any outer indentation level"),
        ]
        
        transformer, error_msg = random.choice(error_types)
        try:
            broken_code = transformer(code)
            return broken_code, error_msg
        except:
            return code.replace(":", "", 1), "SyntaxError: invalid syntax"
    
    def compute_aggregate_metrics(self, all_results: Dict[str, Dict[str, EvaluationResult]]) -> Dict[str, Dict[str, float]]:
        """Compute aggregate metrics across all scenarios"""
        aggregate = {}
        
        for scenario_name, results in all_results.items():
            if not results:
                continue
                
            # Collect pass@k values
            pass_at_1_values = [r.pass_at_1 for r in results.values()]
            pass_at_5_values = [r.pass_at_5 for r in results.values()]
            execution_times = [r.execution_time for r in results.values()]
            
            aggregate[scenario_name] = {
                "pass@1_mean": np.mean(pass_at_1_values),
                "pass@1_std": np.std(pass_at_1_values),
                "pass@5_mean": np.mean(pass_at_5_values),
                "pass@5_std": np.std(pass_at_5_values),
                "avg_execution_time": np.mean(execution_times),
                "num_problems": len(results),
                "total_samples": len(results) * self.config.num_samples
            }
        
        # Overall metrics
        all_pass_1 = []
        all_pass_5 = []
        for results in all_results.values():
            all_pass_1.extend([r.pass_at_1 for r in results.values()])
            all_pass_5.extend([r.pass_at_5 for r in results.values()])
        
        if all_pass_1:
            aggregate["overall"] = {
                "pass@1_mean": np.mean(all_pass_1),
                "pass@1_std": np.std(all_pass_1),
                "pass@5_mean": np.mean(all_pass_5),
                "pass@5_std": np.std(all_pass_5),
                "total_problems": len(all_pass_1)
            }
        
        return aggregate
    
    def save_results(self, all_results: Dict[str, Dict[str, EvaluationResult]], 
                    aggregate_metrics: Dict[str, Dict[str, float]]):
        """Save evaluation results to files"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save aggregate metrics
        metrics_file = os.path.join(self.config.output_dir, f"metrics_{timestamp}.json")
        with open(metrics_file, 'w') as f:
            json.dump(aggregate_metrics, f, indent=2)
        
        print(f"Aggregate metrics saved to {metrics_file}")
        
        # Save detailed results if requested
        if self.config.save_detailed_results:
            for scenario_name, results in all_results.items():
                scenario_file = os.path.join(self.config.output_dir, f"{scenario_name}_{timestamp}.json")
                
                # Convert results to serializable format
                serializable_results = {}
                for problem_id, result in results.items():
                    serializable_results[problem_id] = {
                        "problem_id": result.problem_id,
                        "scenario": result.scenario.value,
                        "generated_codes": result.generated_codes,
                        "test_results": result.test_results,
                        "pass_at_1": result.pass_at_1,
                        "pass_at_5": result.pass_at_5,
                        "execution_time": result.execution_time,
                        "error_messages": result.error_messages,
                        "metadata": result.metadata
                    }
                
                with open(scenario_file, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
                
                print(f"Detailed {scenario_name} results saved to {scenario_file}")

def run_evaluation(config: EvaluationConfig) -> Dict[str, Dict[str, float]]:
    """Main function to run LiveCodeBench evaluation"""
    evaluator = LiveCodeBenchEvaluator(config)
    all_results = {}
    
    # Determine which scenarios to evaluate
    scenarios_to_eval = config.scenarios or [
        Scenario.codegeneration,
        Scenario.selfrepair, 
        Scenario.testoutputprediction,
        Scenario.codeexecution
    ]
    
    # Load datasets and evaluate each scenario
    for scenario in scenarios_to_eval:
        print(f"\n=== Evaluating {scenario.value} ===")
        
        try:
            if scenario == Scenario.codegeneration:
                from lcb_runner.benchmarks import load_code_generation_dataset
                problems = load_code_generation_dataset(
                    start_date=config.start_date,
                    end_date=config.end_date
                )
                results = evaluator.evaluate_code_generation(problems)
                
            elif scenario == Scenario.selfrepair:
                from lcb_runner.benchmarks import load_code_generation_dataset
                problems = load_code_generation_dataset(
                    start_date=config.start_date,
                    end_date=config.end_date
                )
                # Limit for self-repair to avoid too many problems
                problems = problems[:min(50, len(problems))]
                results = evaluator.evaluate_self_repair(problems)
                
            elif scenario == Scenario.testoutputprediction:
                from lcb_runner.benchmarks import load_test_prediction_dataset
                problems = load_test_prediction_dataset()
                results = evaluator.evaluate_test_prediction(problems)
                
            elif scenario == Scenario.codeexecution:
                from lcb_runner.benchmarks import load_code_execution_dataset
                problems = load_code_execution_dataset()
                results = evaluator.evaluate_code_execution(problems)
            
            all_results[scenario.value] = results
            print(f"Completed {scenario.value}: {len(results)} problems evaluated")
            
        except Exception as e:
            print(f"Error evaluating {scenario.value}: {str(e)}")
            traceback.print_exc()
            all_results[scenario.value] = {}
    
    # Compute aggregate metrics
    aggregate_metrics = evaluator.compute_aggregate_metrics(all_results)
    
    # Save results
    evaluator.save_results(all_results, aggregate_metrics)
    
    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    for scenario_name, metrics in aggregate_metrics.items():
        if scenario_name == "overall":
            print(f"\n{scenario_name.upper()}:")
        else:
            print(f"\n{scenario_name}:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")
    
    return aggregate_metrics

if __name__ == "__main__":
    # Example usage
    config = EvaluationConfig(
        model_path="checkpoints/hrm_livecodebench_latest.pt",
        num_samples=5,
        scenarios=[Scenario.codegeneration, Scenario.selfrepair],
        start_date="2023-07-01",
        end_date="2024-06-01",
        output_dir="evaluation_results/livecodebench"
    )
    
    aggregate_metrics = run_evaluation(config)
    print("Evaluation completed!")