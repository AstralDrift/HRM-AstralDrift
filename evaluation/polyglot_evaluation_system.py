"""
Polyglot Evaluation System for HRM Code Generation

This module implements comprehensive evaluation for multi-language code generation,
including test execution, diff quality assessment, and cross-language performance analysis.
"""

import os
import json
import subprocess
import tempfile
import shutil
import logging
import time
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import difflib
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import signal

# Import our modules
from dataset.polyglot_benchmark_extractor import PolyglotProblem, ProblemFile
from dataset.diff_based_training_generator import DiffTrainingExample
from dataset.cross_language_mapper import TransferLearningExample

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationResult(Enum):
    """Possible evaluation results"""
    PASS = "pass"
    FAIL = "fail"
    COMPILE_ERROR = "compile_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    SYNTAX_ERROR = "syntax_error"

@dataclass
class TestExecutionResult:
    """Result of executing tests for a single implementation"""
    language: str
    problem_id: str
    result: EvaluationResult
    passed_tests: int = 0
    total_tests: int = 0
    execution_time: float = 0.0
    error_message: str = ""
    stdout: str = ""
    stderr: str = ""
    compile_output: str = ""

@dataclass
class DiffQualityResult:
    """Result of evaluating diff quality"""
    problem_id: str
    language: str
    edit_type: str
    
    # Quality metrics
    edit_distance: int
    lines_changed: int
    semantic_correctness: float  # 0.0 to 1.0
    syntax_correctness: float    # 0.0 to 1.0
    
    # Diff analysis
    diff_text: str
    minimal_diff: bool
    context_preserved: bool
    
    # Overall score
    quality_score: float

@dataclass
class CrossLanguageResult:
    """Result of cross-language performance analysis"""
    problem_id: str
    source_language: str
    target_language: str
    transfer_success: bool
    semantic_preservation: float
    performance_comparison: Dict[str, float]
    error_analysis: List[str]

class LanguageTestRunner:
    """Handles test execution for different programming languages"""
    
    def __init__(self, timeout: int = 30):
        """
        Initialize test runner
        
        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout
        self.test_configs = {
            "python": {
                "build_command": None,
                "test_command": "python -m pytest {test_file} -v",
                "file_extension": ".py",
                "test_pattern": "*_test.py",
                "requires_main": False
            },
            "javascript": {
                "build_command": "npm install",
                "test_command": "npm test",
                "file_extension": ".js",
                "test_pattern": "*.spec.js",
                "requires_main": False
            },
            "java": {
                "build_command": "./gradlew build",
                "test_command": "./gradlew test",
                "file_extension": ".java",
                "test_pattern": "*Test.java",
                "requires_main": False
            },
            "cpp": {
                "build_command": "cmake . && make",
                "test_command": "./test_runner",
                "file_extension": ".cpp",
                "test_pattern": "*_test.cpp",
                "requires_main": True
            },
            "go": {
                "build_command": "go mod tidy",
                "test_command": "go test -v",
                "file_extension": ".go",
                "test_pattern": "*_test.go",
                "requires_main": False
            },
            "rust": {
                "build_command": "cargo build",
                "test_command": "cargo test",
                "file_extension": ".rs",
                "test_pattern": "tests/*.rs",
                "requires_main": False
            }
        }
    
    def execute_tests(self, problem: PolyglotProblem, language: str, 
                     generated_code: str) -> TestExecutionResult:
        """Execute tests for generated code"""
        logger.debug(f"Executing tests for {problem.problem_id} in {language}")
        
        if language not in self.test_configs:
            return TestExecutionResult(
                language=language,
                problem_id=problem.problem_id,
                result=EvaluationResult.FAIL,
                error_message=f"Unsupported language: {language}"
            )
        
        config = self.test_configs[language]
        
        # Create temporary directory for test execution
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            try:
                # Set up test environment
                self._setup_test_environment(problem, language, generated_code, temp_path)
                
                # Build if necessary
                if config["build_command"]:
                    build_result = self._run_command(config["build_command"], temp_path)
                    if build_result.returncode != 0:
                        return TestExecutionResult(
                            language=language,
                            problem_id=problem.problem_id,
                            result=EvaluationResult.COMPILE_ERROR,
                            error_message="Build failed",
                            stderr=build_result.stderr,
                            compile_output=build_result.stdout
                        )
                
                # Run tests
                start_time = time.time()
                test_result = self._run_command(config["test_command"], temp_path)
                execution_time = time.time() - start_time
                
                # Analyze test results
                return self._analyze_test_output(
                    test_result, language, problem.problem_id, execution_time
                )
                
            except TimeoutError:
                return TestExecutionResult(
                    language=language,
                    problem_id=problem.problem_id,
                    result=EvaluationResult.TIMEOUT,
                    execution_time=self.timeout,
                    error_message=f"Test execution timed out after {self.timeout} seconds"
                )
            except Exception as e:
                return TestExecutionResult(
                    language=language,
                    problem_id=problem.problem_id,
                    result=EvaluationResult.RUNTIME_ERROR,
                    error_message=f"Unexpected error: {str(e)}"
                )
    
    def _setup_test_environment(self, problem: PolyglotProblem, language: str,
                               generated_code: str, temp_path: Path):
        """Set up the test environment in temporary directory"""
        if language not in problem.files_by_language:
            raise ValueError(f"No files found for language {language}")
        
        files = problem.files_by_language[language]
        
        # Copy all necessary files
        for file in files:
            file_path = temp_path / file.filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if file.file_type == "implementation":
                # Use generated code instead of original
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(generated_code)
            else:
                # Copy other files as-is (tests, build files, etc.)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(file.content)
        
        # Make gradlew executable for Java projects
        if language == "java":
            gradlew_path = temp_path / "gradlew"
            if gradlew_path.exists():
                os.chmod(gradlew_path, 0o755)
    
    def _run_command(self, command: str, cwd: Path) -> subprocess.CompletedProcess:
        """Run a command with timeout"""
        try:
            # Handle command with placeholders
            if "{test_file}" in command:
                # Find test files
                test_files = list(cwd.glob("*test*"))
                if test_files:
                    command = command.format(test_file=test_files[0].name)
                else:
                    command = command.replace(" {test_file}", "")
            
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            return result
            
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Command timed out: {command}")
    
    def _analyze_test_output(self, result: subprocess.CompletedProcess,
                            language: str, problem_id: str, 
                            execution_time: float) -> TestExecutionResult:
        """Analyze test command output to determine results"""
        
        stdout = result.stdout
        stderr = result.stderr
        
        # Language-specific test result parsing
        if language == "python":
            return self._parse_python_tests(result, problem_id, execution_time)
        elif language == "javascript":
            return self._parse_javascript_tests(result, problem_id, execution_time)
        elif language == "java":
            return self._parse_java_tests(result, problem_id, execution_time)
        elif language == "cpp":
            return self._parse_cpp_tests(result, problem_id, execution_time)
        elif language == "go":
            return self._parse_go_tests(result, problem_id, execution_time)
        elif language == "rust":
            return self._parse_rust_tests(result, problem_id, execution_time)
        else:
            # Generic parsing
            return self._parse_generic_tests(result, language, problem_id, execution_time)
    
    def _parse_python_tests(self, result: subprocess.CompletedProcess,
                           problem_id: str, execution_time: float) -> TestExecutionResult:
        """Parse pytest output"""
        stdout = result.stdout
        stderr = result.stderr
        
        # Check for syntax errors
        if "SyntaxError" in stderr:
            return TestExecutionResult(
                language="python",
                problem_id=problem_id,
                result=EvaluationResult.SYNTAX_ERROR,
                execution_time=execution_time,
                error_message="Syntax error in generated code",
                stderr=stderr
            )
        
        # Parse test results
        passed_match = re.search(r'(\d+) passed', stdout)
        failed_match = re.search(r'(\d+) failed', stdout)
        
        passed_tests = int(passed_match.group(1)) if passed_match else 0
        failed_tests = int(failed_match.group(1)) if failed_match else 0
        total_tests = passed_tests + failed_tests
        
        if result.returncode == 0 and passed_tests > 0:
            eval_result = EvaluationResult.PASS
        elif failed_tests > 0:
            eval_result = EvaluationResult.FAIL
        else:
            eval_result = EvaluationResult.RUNTIME_ERROR
        
        return TestExecutionResult(
            language="python",
            problem_id=problem_id,
            result=eval_result,
            passed_tests=passed_tests,
            total_tests=total_tests,
            execution_time=execution_time,
            stdout=stdout,
            stderr=stderr
        )
    
    def _parse_javascript_tests(self, result: subprocess.CompletedProcess,
                               problem_id: str, execution_time: float) -> TestExecutionResult:
        """Parse JavaScript test output (Jest/Mocha)"""
        stdout = result.stdout
        
        # Look for test summary
        passed_match = re.search(r'(\d+) passing', stdout)
        failed_match = re.search(r'(\d+) failing', stdout)
        
        passed_tests = int(passed_match.group(1)) if passed_match else 0
        failed_tests = int(failed_match.group(1)) if failed_match else 0
        total_tests = passed_tests + failed_tests
        
        eval_result = EvaluationResult.PASS if result.returncode == 0 else EvaluationResult.FAIL
        
        return TestExecutionResult(
            language="javascript",
            problem_id=problem_id,
            result=eval_result,
            passed_tests=passed_tests,
            total_tests=total_tests,
            execution_time=execution_time,
            stdout=stdout,
            stderr=result.stderr
        )
    
    def _parse_java_tests(self, result: subprocess.CompletedProcess,
                         problem_id: str, execution_time: float) -> TestExecutionResult:
        """Parse Gradle test output"""
        stdout = result.stdout
        
        # Parse Gradle test results
        test_match = re.search(r'(\d+) tests completed, (\d+) failed', stdout)
        if test_match:
            total_tests = int(test_match.group(1))
            failed_tests = int(test_match.group(2))
            passed_tests = total_tests - failed_tests
        else:
            passed_tests = total_tests = 0
        
        eval_result = EvaluationResult.PASS if result.returncode == 0 else EvaluationResult.FAIL
        
        return TestExecutionResult(
            language="java",
            problem_id=problem_id,
            result=eval_result,
            passed_tests=passed_tests,
            total_tests=total_tests,
            execution_time=execution_time,
            stdout=stdout,
            stderr=result.stderr
        )
    
    def _parse_go_tests(self, result: subprocess.CompletedProcess,
                       problem_id: str, execution_time: float) -> TestExecutionResult:
        """Parse Go test output"""
        stdout = result.stdout
        
        # Count PASS and FAIL lines
        passed_tests = len(re.findall(r'--- PASS:', stdout))
        failed_tests = len(re.findall(r'--- FAIL:', stdout))
        total_tests = passed_tests + failed_tests
        
        eval_result = EvaluationResult.PASS if result.returncode == 0 else EvaluationResult.FAIL
        
        return TestExecutionResult(
            language="go",
            problem_id=problem_id,
            result=eval_result,
            passed_tests=passed_tests,
            total_tests=total_tests,
            execution_time=execution_time,
            stdout=stdout,
            stderr=result.stderr
        )
    
    def _parse_rust_tests(self, result: subprocess.CompletedProcess,
                         problem_id: str, execution_time: float) -> TestExecutionResult:
        """Parse Cargo test output"""
        stdout = result.stdout
        
        # Parse test results
        test_result_match = re.search(r'test result: (\w+)\. (\d+) passed; (\d+) failed', stdout)
        if test_result_match:
            passed_tests = int(test_result_match.group(2))
            failed_tests = int(test_result_match.group(3))
            total_tests = passed_tests + failed_tests
        else:
            passed_tests = total_tests = 0
        
        eval_result = EvaluationResult.PASS if result.returncode == 0 else EvaluationResult.FAIL
        
        return TestExecutionResult(
            language="rust",
            problem_id=problem_id,
            result=eval_result,
            passed_tests=passed_tests,
            total_tests=total_tests,
            execution_time=execution_time,
            stdout=stdout,
            stderr=result.stderr
        )
    
    def _parse_cpp_tests(self, result: subprocess.CompletedProcess,
                        problem_id: str, execution_time: float) -> TestExecutionResult:
        """Parse C++ test output (Catch2)"""
        stdout = result.stdout
        
        # Parse Catch2 output
        passed_match = re.search(r'All tests passed \((\d+) assertion', stdout)
        failed_match = re.search(r'(\d+) test case.* failed', stdout)
        
        if passed_match:
            passed_tests = 1  # At least one test passed
            failed_tests = 0
        elif failed_match:
            passed_tests = 0
            failed_tests = int(failed_match.group(1))
        else:
            passed_tests = failed_tests = 0
        
        total_tests = passed_tests + failed_tests
        eval_result = EvaluationResult.PASS if result.returncode == 0 else EvaluationResult.FAIL
        
        return TestExecutionResult(
            language="cpp",
            problem_id=problem_id,
            result=eval_result,
            passed_tests=passed_tests,
            total_tests=total_tests,
            execution_time=execution_time,
            stdout=stdout,
            stderr=result.stderr
        )
    
    def _parse_generic_tests(self, result: subprocess.CompletedProcess,
                            language: str, problem_id: str,
                            execution_time: float) -> TestExecutionResult:
        """Generic test result parsing"""
        eval_result = EvaluationResult.PASS if result.returncode == 0 else EvaluationResult.FAIL
        
        return TestExecutionResult(
            language=language,
            problem_id=problem_id,
            result=eval_result,
            execution_time=execution_time,
            stdout=result.stdout,
            stderr=result.stderr
        )

class DiffQualityEvaluator:
    """Evaluates the quality of diff-based edits"""
    
    def __init__(self):
        self.syntax_checkers = {
            "python": self._check_python_syntax,
            "javascript": self._check_javascript_syntax,
            "java": self._check_java_syntax,
            "cpp": self._check_cpp_syntax,
            "go": self._check_go_syntax,
            "rust": self._check_rust_syntax
        }
    
    def evaluate_diff_quality(self, original_code: str, modified_code: str,
                             language: str, problem_id: str,
                             edit_type: str) -> DiffQualityResult:
        """Evaluate the quality of a diff-based edit"""
        
        # Generate diff
        diff_lines = list(difflib.unified_diff(
            original_code.splitlines(keepends=True),
            modified_code.splitlines(keepends=True),
            fromfile="original",
            tofile="modified",
            n=3
        ))
        diff_text = ''.join(diff_lines)
        
        # Calculate metrics
        edit_distance = self._calculate_edit_distance(original_code, modified_code)
        lines_changed = self._count_changed_lines(diff_lines)
        syntax_correctness = self._check_syntax_correctness(modified_code, language)
        semantic_correctness = self._estimate_semantic_correctness(
            original_code, modified_code, language
        )
        
        # Analyze diff properties
        minimal_diff = self._is_minimal_diff(original_code, modified_code)
        context_preserved = self._is_context_preserved(original_code, modified_code)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            edit_distance, lines_changed, syntax_correctness,
            semantic_correctness, minimal_diff, context_preserved
        )
        
        return DiffQualityResult(
            problem_id=problem_id,
            language=language,
            edit_type=edit_type,
            edit_distance=edit_distance,
            lines_changed=lines_changed,
            semantic_correctness=semantic_correctness,
            syntax_correctness=syntax_correctness,
            diff_text=diff_text,
            minimal_diff=minimal_diff,
            context_preserved=context_preserved,
            quality_score=quality_score
        )
    
    def _calculate_edit_distance(self, text1: str, text2: str) -> int:
        """Calculate Levenshtein distance between two texts"""
        if len(text1) < len(text2):
            return self._calculate_edit_distance(text2, text1)
        
        if len(text2) == 0:
            return len(text1)
        
        previous_row = list(range(len(text2) + 1))
        for i, c1 in enumerate(text1):
            current_row = [i + 1]
            for j, c2 in enumerate(text2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _count_changed_lines(self, diff_lines: List[str]) -> int:
        """Count the number of changed lines in a diff"""
        changed_lines = 0
        for line in diff_lines:
            if line.startswith('+') or line.startswith('-'):
                if not line.startswith('+++') and not line.startswith('---'):
                    changed_lines += 1
        return changed_lines
    
    def _check_syntax_correctness(self, code: str, language: str) -> float:
        """Check syntax correctness of code"""
        if language in self.syntax_checkers:
            return self.syntax_checkers[language](code)
        return 1.0  # Assume correct if no checker available
    
    def _check_python_syntax(self, code: str) -> float:
        """Check Python syntax"""
        try:
            compile(code, '<string>', 'exec')
            return 1.0
        except SyntaxError:
            return 0.0
        except Exception:
            return 0.5  # Other errors might be runtime issues
    
    def _check_javascript_syntax(self, code: str) -> float:
        """Check JavaScript syntax (simplified)"""
        # Basic syntax checks
        if code.count('(') != code.count(')'):
            return 0.0
        if code.count('{') != code.count('}'):
            return 0.0
        if code.count('[') != code.count(']'):
            return 0.0
        return 1.0
    
    def _check_java_syntax(self, code: str) -> float:
        """Check Java syntax (simplified)"""
        # Basic syntax checks
        if code.count('(') != code.count(')'):
            return 0.0
        if code.count('{') != code.count('}'):
            return 0.0
        # Check for basic structure
        if 'class' not in code and 'interface' not in code:
            return 0.5
        return 1.0
    
    def _check_cpp_syntax(self, code: str) -> float:
        """Check C++ syntax (simplified)"""
        return self._check_java_syntax(code)  # Similar bracket checking
    
    def _check_go_syntax(self, code: str) -> float:
        """Check Go syntax (simplified)"""
        if code.count('(') != code.count(')'):
            return 0.0
        if code.count('{') != code.count('}'):
            return 0.0
        return 1.0
    
    def _check_rust_syntax(self, code: str) -> float:
        """Check Rust syntax (simplified)"""
        return self._check_go_syntax(code)  # Similar checks
    
    def _estimate_semantic_correctness(self, original: str, modified: str, language: str) -> float:
        """Estimate semantic correctness (simplified heuristic)"""
        # This is a simplified approach - in practice you'd want more sophisticated analysis
        
        # Check if major structural elements are preserved
        structure_preservation = 0.0
        
        # Function preservation
        orig_functions = re.findall(r'def\s+\w+|function\s+\w+|func\s+\w+|fn\s+\w+', original)
        mod_functions = re.findall(r'def\s+\w+|function\s+\w+|func\s+\w+|fn\s+\w+', modified)
        
        if orig_functions:
            common_functions = set(orig_functions) & set(mod_functions)
            structure_preservation += len(common_functions) / len(orig_functions) * 0.5
        else:
            structure_preservation += 0.5
        
        # Variable preservation (simplified)
        orig_vars = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', original))
        mod_vars = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', modified))
        
        if orig_vars:
            common_vars = orig_vars & mod_vars
            structure_preservation += len(common_vars) / len(orig_vars) * 0.5
        else:
            structure_preservation += 0.5
        
        return min(1.0, structure_preservation)
    
    def _is_minimal_diff(self, original: str, modified: str) -> bool:
        """Check if the diff is minimal (no unnecessary changes)"""
        # Simple heuristic: if edit distance is reasonable compared to total length
        edit_dist = self._calculate_edit_distance(original, modified)
        total_length = max(len(original), len(modified))
        
        return edit_dist / total_length < 0.5 if total_length > 0 else True
    
    def _is_context_preserved(self, original: str, modified: str) -> bool:
        """Check if surrounding context is preserved"""
        orig_lines = original.splitlines()
        mod_lines = modified.splitlines()
        
        # Check if first and last lines are similar (simple heuristic)
        if not orig_lines or not mod_lines:
            return False
        
        first_similar = orig_lines[0].strip() == mod_lines[0].strip()
        last_similar = orig_lines[-1].strip() == mod_lines[-1].strip()
        
        return first_similar and last_similar
    
    def _calculate_quality_score(self, edit_distance: int, lines_changed: int,
                                syntax_correctness: float, semantic_correctness: float,
                                minimal_diff: bool, context_preserved: bool) -> float:
        """Calculate overall quality score"""
        # Normalize edit distance (lower is better)
        edit_score = max(0, 1 - edit_distance / 1000)  # Assume 1000 as max reasonable distance
        
        # Normalize lines changed (fewer changes often better for diffs)
        lines_score = max(0, 1 - lines_changed / 100)  # Assume 100 as max reasonable changes
        
        # Boolean flags as scores
        minimal_score = 1.0 if minimal_diff else 0.5
        context_score = 1.0 if context_preserved else 0.5
        
        # Weighted average
        quality_score = (
            0.2 * edit_score +
            0.2 * lines_score +
            0.3 * syntax_correctness +
            0.2 * semantic_correctness +
            0.05 * minimal_score +
            0.05 * context_score
        )
        
        return min(1.0, quality_score)

class PolyglotEvaluationSystem:
    """
    Main evaluation system for Polyglot benchmark integration
    """
    
    def __init__(self, timeout: int = 30, max_workers: int = 4):
        """
        Initialize evaluation system
        
        Args:
            timeout: Test execution timeout in seconds
            max_workers: Maximum parallel workers for evaluation
        """
        self.timeout = timeout
        self.max_workers = max_workers
        
        self.test_runner = LanguageTestRunner(timeout)
        self.diff_evaluator = DiffQualityEvaluator()
        
        # Results storage
        self.test_results: List[TestExecutionResult] = []
        self.diff_results: List[DiffQualityResult] = []
        self.cross_language_results: List[CrossLanguageResult] = []
    
    def evaluate_generated_code(self, problems: Dict[str, PolyglotProblem],
                               generated_solutions: Dict[Tuple[str, str], str]) -> Dict:
        """
        Evaluate generated code solutions
        
        Args:
            problems: Dictionary of problem_id -> PolyglotProblem
            generated_solutions: Dictionary of (problem_id, language) -> generated_code
        
        Returns:
            Dictionary with evaluation results and statistics
        """
        logger.info(f"Evaluating {len(generated_solutions)} generated solutions...")
        
        self.test_results = []
        
        # Parallel evaluation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for (problem_id, language), code in generated_solutions.items():
                if problem_id in problems:
                    future = executor.submit(
                        self.test_runner.execute_tests,
                        problems[problem_id],
                        language,
                        code
                    )
                    futures.append(future)
            
            # Collect results
            for future in futures:
                try:
                    result = future.result(timeout=self.timeout + 10)
                    self.test_results.append(result)
                except Exception as e:
                    logger.error(f"Evaluation failed: {e}")
        
        return self._compile_test_statistics()
    
    def evaluate_diff_examples(self, diff_examples: List[DiffTrainingExample]) -> Dict:
        """
        Evaluate diff-based training examples
        
        Args:
            diff_examples: List of diff training examples to evaluate
        
        Returns:
            Dictionary with diff quality statistics
        """
        logger.info(f"Evaluating {len(diff_examples)} diff examples...")
        
        self.diff_results = []
        
        for example in diff_examples:
            for diff_op in example.diff_operations:
                result = self.diff_evaluator.evaluate_diff_quality(
                    example.original_code,
                    example.target_code,
                    example.language,
                    example.problem_description,
                    diff_op.edit_type.value
                )
                self.diff_results.append(result)
        
        return self._compile_diff_statistics()
    
    def evaluate_cross_language_transfer(self, transfer_examples: List[TransferLearningExample],
                                       problems: Dict[str, PolyglotProblem]) -> Dict:
        """
        Evaluate cross-language transfer examples
        
        Args:
            transfer_examples: List of transfer learning examples
            problems: Dictionary of problems for reference
        
        Returns:
            Dictionary with cross-language evaluation results
        """
        logger.info(f"Evaluating {len(transfer_examples)} cross-language transfers...")
        
        self.cross_language_results = []
        
        for example in transfer_examples:
            # Test both source and target implementations
            problem = problems.get(example.problem_id)
            if not problem:
                continue
            
            source_result = self.test_runner.execute_tests(
                problem, example.source_language, example.source_code
            )
            target_result = self.test_runner.execute_tests(
                problem, example.target_language, example.target_code
            )
            
            # Analyze transfer success
            transfer_success = (
                source_result.result == EvaluationResult.PASS and
                target_result.result == EvaluationResult.PASS
            )
            
            # Calculate semantic preservation
            semantic_preservation = self._calculate_semantic_preservation(
                example.source_code, example.target_code
            )
            
            # Performance comparison
            performance_comparison = {
                "source_time": source_result.execution_time,
                "target_time": target_result.execution_time,
                "time_ratio": (target_result.execution_time / max(source_result.execution_time, 0.001))
            }
            
            # Error analysis
            error_analysis = []
            if source_result.result != EvaluationResult.PASS:
                error_analysis.append(f"Source error: {source_result.error_message}")
            if target_result.result != EvaluationResult.PASS:
                error_analysis.append(f"Target error: {target_result.error_message}")
            
            cross_lang_result = CrossLanguageResult(
                problem_id=example.problem_id,
                source_language=example.source_language,
                target_language=example.target_language,
                transfer_success=transfer_success,
                semantic_preservation=semantic_preservation,
                performance_comparison=performance_comparison,
                error_analysis=error_analysis
            )
            
            self.cross_language_results.append(cross_lang_result)
        
        return self._compile_cross_language_statistics()
    
    def _calculate_semantic_preservation(self, source_code: str, target_code: str) -> float:
        """Calculate how well semantics are preserved in cross-language transfer"""
        # Simplified semantic preservation metric
        # In practice, you'd want more sophisticated analysis
        
        # Check structural similarity
        source_lines = len(source_code.splitlines())
        target_lines = len(target_code.splitlines())
        
        line_ratio = min(source_lines, target_lines) / max(source_lines, target_lines)
        
        # Check function count similarity
        source_funcs = len(re.findall(r'def\s+\w+|function\s+\w+|func\s+\w+|fn\s+\w+', source_code))
        target_funcs = len(re.findall(r'def\s+\w+|function\s+\w+|func\s+\w+|fn\s+\w+', target_code))
        
        func_ratio = min(source_funcs, target_funcs) / max(source_funcs, target_funcs) if max(source_funcs, target_funcs) > 0 else 1.0
        
        return (line_ratio + func_ratio) / 2
    
    def _compile_test_statistics(self) -> Dict:
        """Compile statistics from test execution results"""
        if not self.test_results:
            return {}
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.result == EvaluationResult.PASS)
        
        # Group by language
        by_language = {}
        for result in self.test_results:
            lang = result.language
            if lang not in by_language:
                by_language[lang] = {"total": 0, "passed": 0, "avg_time": 0.0}
            
            by_language[lang]["total"] += 1
            if result.result == EvaluationResult.PASS:
                by_language[lang]["passed"] += 1
            by_language[lang]["avg_time"] += result.execution_time
        
        # Calculate averages
        for lang_stats in by_language.values():
            lang_stats["success_rate"] = lang_stats["passed"] / lang_stats["total"]
            lang_stats["avg_time"] /= lang_stats["total"]
        
        # Group by result type
        result_counts = {}
        for result in self.test_results:
            result_type = result.result.value
            result_counts[result_type] = result_counts.get(result_type, 0) + 1
        
        return {
            "total_evaluations": total_tests,
            "overall_success_rate": passed_tests / total_tests,
            "by_language": by_language,
            "by_result_type": result_counts,
            "average_execution_time": sum(r.execution_time for r in self.test_results) / total_tests
        }
    
    def _compile_diff_statistics(self) -> Dict:
        """Compile statistics from diff quality results"""
        if not self.diff_results:
            return {}
        
        total_diffs = len(self.diff_results)
        
        # Overall quality metrics
        avg_quality = sum(r.quality_score for r in self.diff_results) / total_diffs
        avg_edit_distance = sum(r.edit_distance for r in self.diff_results) / total_diffs
        avg_syntax_correctness = sum(r.syntax_correctness for r in self.diff_results) / total_diffs
        
        # Group by language
        by_language = {}
        for result in self.diff_results:
            lang = result.language
            if lang not in by_language:
                by_language[lang] = {"count": 0, "avg_quality": 0.0, "avg_syntax": 0.0}
            
            by_language[lang]["count"] += 1
            by_language[lang]["avg_quality"] += result.quality_score
            by_language[lang]["avg_syntax"] += result.syntax_correctness
        
        for lang_stats in by_language.values():
            lang_stats["avg_quality"] /= lang_stats["count"]
            lang_stats["avg_syntax"] /= lang_stats["count"]
        
        # Group by edit type
        by_edit_type = {}
        for result in self.diff_results:
            edit_type = result.edit_type
            if edit_type not in by_edit_type:
                by_edit_type[edit_type] = {"count": 0, "avg_quality": 0.0}
            
            by_edit_type[edit_type]["count"] += 1
            by_edit_type[edit_type]["avg_quality"] += result.quality_score
        
        for edit_stats in by_edit_type.values():
            edit_stats["avg_quality"] /= edit_stats["count"]
        
        return {
            "total_diffs": total_diffs,
            "average_quality_score": avg_quality,
            "average_edit_distance": avg_edit_distance,
            "average_syntax_correctness": avg_syntax_correctness,
            "by_language": by_language,
            "by_edit_type": by_edit_type
        }
    
    def _compile_cross_language_statistics(self) -> Dict:
        """Compile statistics from cross-language evaluation results"""
        if not self.cross_language_results:
            return {}
        
        total_transfers = len(self.cross_language_results)
        successful_transfers = sum(1 for r in self.cross_language_results if r.transfer_success)
        
        # Group by language pair
        by_language_pair = {}
        for result in self.cross_language_results:
            pair = f"{result.source_language}->{result.target_language}"
            if pair not in by_language_pair:
                by_language_pair[pair] = {
                    "count": 0, "successful": 0, "avg_semantic_preservation": 0.0
                }
            
            by_language_pair[pair]["count"] += 1
            if result.transfer_success:
                by_language_pair[pair]["successful"] += 1
            by_language_pair[pair]["avg_semantic_preservation"] += result.semantic_preservation
        
        for pair_stats in by_language_pair.values():
            pair_stats["success_rate"] = pair_stats["successful"] / pair_stats["count"]
            pair_stats["avg_semantic_preservation"] /= pair_stats["count"]
        
        return {
            "total_transfers": total_transfers,
            "overall_success_rate": successful_transfers / total_transfers,
            "average_semantic_preservation": sum(r.semantic_preservation for r in self.cross_language_results) / total_transfers,
            "by_language_pair": by_language_pair
        }
    
    def export_detailed_results(self, output_path: str):
        """Export detailed evaluation results to JSON"""
        logger.info(f"Exporting detailed results to {output_path}")
        
        export_data = {
            "metadata": {
                "evaluation_timestamp": time.time(),
                "timeout_seconds": self.timeout,
                "max_workers": self.max_workers
            },
            "test_results": [asdict(r) for r in self.test_results],
            "diff_results": [asdict(r) for r in self.diff_results],
            "cross_language_results": [asdict(r) for r in self.cross_language_results],
            "statistics": {
                "test_statistics": self._compile_test_statistics(),
                "diff_statistics": self._compile_diff_statistics(),
                "cross_language_statistics": self._compile_cross_language_statistics()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info("Export complete")
    
    def get_success_rate_by_language(self) -> Dict[str, float]:
        """Get success rates by programming language"""
        stats = self._compile_test_statistics()
        return {lang: data["success_rate"] for lang, data in stats.get("by_language", {}).items()}
    
    def get_transfer_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get cross-language transfer success matrix"""
        matrix = {}
        
        for result in self.cross_language_results:
            source = result.source_language
            target = result.target_language
            
            if source not in matrix:
                matrix[source] = {}
            if target not in matrix[source]:
                matrix[source][target] = []
            
            matrix[source][target].append(1.0 if result.transfer_success else 0.0)
        
        # Calculate averages
        for source in matrix:
            for target in matrix[source]:
                values = matrix[source][target]
                matrix[source][target] = sum(values) / len(values) if values else 0.0
        
        return matrix

def main():
    """Main function for testing the evaluation system"""
    import argparse
    from dataset.polyglot_benchmark_extractor import PolyglotBenchmarkExtractor
    
    parser = argparse.ArgumentParser(description="Evaluate Polyglot benchmark solutions")
    parser.add_argument("--problems-json", required=True,
                       help="JSON file with extracted problems")
    parser.add_argument("--solutions-json", required=True,
                       help="JSON file with generated solutions")
    parser.add_argument("--output", default="evaluation_results.json",
                       help="Output results file")
    parser.add_argument("--timeout", type=int, default=30,
                       help="Test execution timeout in seconds")
    
    args = parser.parse_args()
    
    # Load problems
    extractor = PolyglotBenchmarkExtractor("")
    extractor.load_from_json(args.problems_json)
    
    # Load solutions (mock data for testing)
    with open(args.solutions_json, 'r') as f:
        solutions_data = json.load(f)
    
    # Initialize evaluation system
    evaluator = PolyglotEvaluationSystem(timeout=args.timeout)
    
    # Run evaluation
    results = evaluator.evaluate_generated_code(
        extractor.problems_cache,
        solutions_data["solutions"]
    )
    
    # Export results
    evaluator.export_detailed_results(args.output)
    
    # Print summary
    print(f"\nEvaluation Summary:")
    print(f"Total evaluations: {results['total_evaluations']}")
    print(f"Overall success rate: {results['overall_success_rate']:.2%}")
    print(f"Average execution time: {results['average_execution_time']:.3f}s")
    
    print(f"\nSuccess rates by language:")
    for lang, stats in results["by_language"].items():
        print(f"  {lang}: {stats['success_rate']:.2%} ({stats['passed']}/{stats['total']})")

if __name__ == "__main__":
    main()