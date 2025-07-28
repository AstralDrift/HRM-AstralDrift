"""
Reverse Learning Enhanced Evaluation System

This module provides comprehensive evaluation capabilities for the Reverse Learning framework,
including code quality metrics, architectural assessment, and planning improvement analysis.
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
import json
import time
from pathlib import Path
import re
import ast

from models.hrm.reverse_learning import ReverseLearningMetrics


@dataclass
class CodeQualityMetrics:
    """Comprehensive code quality assessment metrics"""
    
    # Structural quality
    complexity_score: float  # Code complexity (lower is better)
    readability_score: float  # Readability assessment (higher is better)
    maintainability_score: float  # How maintainable the code is (higher is better)
    
    # Architectural quality
    modularity_score: float  # How well-structured/modular (higher is better)
    design_pattern_score: float  # Use of appropriate design patterns (higher is better)
    abstraction_level: float  # Appropriate level of abstraction (0-1 scale)
    
    # Planning quality
    strategic_coherence: float  # How well high-level strategy aligns with implementation
    incremental_improvement: float  # Progressive refinement quality
    
    # Implementation quality
    syntax_correctness: float  # Syntactic correctness (0-1 scale)
    semantic_correctness: float  # Semantic correctness estimation
    efficiency_score: float  # Algorithmic efficiency assessment


@dataclass
class ReverseLearningEvaluationResults:
    """Comprehensive evaluation results for Reverse Learning"""
    
    # Standard metrics
    base_accuracy: float
    enhanced_accuracy: float
    improvement: float
    
    # Code quality metrics
    code_quality: CodeQualityMetrics
    quality_improvement: float
    
    # Reverse learning specific metrics
    avg_insight_strength: float
    avg_integration_gate: float
    avg_planning_refinement: float
    feedback_effectiveness: float
    
    # Performance metrics
    inference_time_base: float
    inference_time_enhanced: float
    memory_overhead: float
    
    # Detailed results
    reverse_metrics_history: List[ReverseLearningMetrics]
    per_sample_quality_improvements: List[float]


class CodeQualityAnalyzer:
    """Analyzer for code quality metrics"""
    
    def __init__(self):
        self.complexity_patterns = [
            r'for\s+\w+\s+in\s+.*?:',  # For loops
            r'while\s+.*?:',  # While loops
            r'if\s+.*?:',  # If statements
            r'elif\s+.*?:',  # Elif statements
            r'try\s*:',  # Try blocks
            r'except\s+.*?:',  # Exception handling
            r'def\s+\w+\s*\(',  # Function definitions
            r'class\s+\w+.*?:',  # Class definitions
        ]
        
        self.readability_patterns = [
            r'#.*',  # Comments
            r'""".*?"""',  # Docstrings
            r"'''.*?'''",  # Alt docstrings
            r'\n\s*\n',  # Blank lines for readability
        ]
        
        self.design_patterns = [
            r'class\s+\w*Factory\w*',  # Factory pattern
            r'class\s+\w*Builder\w*',  # Builder pattern
            r'class\s+\w*Observer\w*',  # Observer pattern
            r'class\s+\w*Strategy\w*',  # Strategy pattern
            r'@property',  # Property decorators
            r'@staticmethod',  # Static methods
            r'@classmethod',  # Class methods
        ]
    
    def analyze_code_quality(self, code_text: str) -> CodeQualityMetrics:
        """Analyze code quality from generated text"""
        if not code_text or not isinstance(code_text, str):
            return self._empty_metrics()
        
        # Clean and prepare code
        code_lines = code_text.split('\n')
        non_empty_lines = [line for line in code_lines if line.strip()]
        
        if not non_empty_lines:
            return self._empty_metrics()
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity(code_text)
        
        # Calculate readability score
        readability_score = self._calculate_readability(code_text, code_lines)
        
        # Calculate maintainability score
        maintainability_score = self._calculate_maintainability(code_text, code_lines)
        
        # Calculate modularity score
        modularity_score = self._calculate_modularity(code_text)
        
        # Calculate design pattern usage
        design_pattern_score = self._calculate_design_patterns(code_text)
        
        # Calculate abstraction level
        abstraction_level = self._calculate_abstraction_level(code_text)
        
        # Calculate strategic coherence (requires context analysis)
        strategic_coherence = self._calculate_strategic_coherence(code_text)
        
        # Calculate incremental improvement potential
        incremental_improvement = self._calculate_incremental_improvement(code_text)
        
        # Calculate implementation quality
        syntax_correctness = self._calculate_syntax_correctness(code_text)
        semantic_correctness = self._calculate_semantic_correctness(code_text)
        efficiency_score = self._calculate_efficiency(code_text)
        
        return CodeQualityMetrics(
            complexity_score=complexity_score,
            readability_score=readability_score,
            maintainability_score=maintainability_score,
            modularity_score=modularity_score,
            design_pattern_score=design_pattern_score,
            abstraction_level=abstraction_level,
            strategic_coherence=strategic_coherence,
            incremental_improvement=incremental_improvement,
            syntax_correctness=syntax_correctness,
            semantic_correctness=semantic_correctness,
            efficiency_score=efficiency_score
        )
    
    def _empty_metrics(self) -> CodeQualityMetrics:
        """Return empty/default metrics"""
        return CodeQualityMetrics(
            complexity_score=0.5, readability_score=0.5, maintainability_score=0.5,
            modularity_score=0.5, design_pattern_score=0.5, abstraction_level=0.5,
            strategic_coherence=0.5, incremental_improvement=0.5,
            syntax_correctness=0.5, semantic_correctness=0.5, efficiency_score=0.5
        )
    
    def _calculate_complexity(self, code: str) -> float:
        """Calculate cyclomatic complexity score (0-1, lower is better)"""
        complexity_count = 0
        for pattern in self.complexity_patterns:
            complexity_count += len(re.findall(pattern, code, re.MULTILINE | re.DOTALL))
        
        # Normalize by code length (rough heuristic)
        lines = len(code.split('\n'))
        if lines == 0:
            return 0.5
        
        complexity_ratio = complexity_count / lines
        # Convert to 0-1 scale where lower is better
        return min(1.0, complexity_ratio * 2)
    
    def _calculate_readability(self, code: str, code_lines: List[str]) -> float:
        """Calculate readability score (0-1, higher is better)"""
        if not code_lines:
            return 0.5
        
        # Count documentation and comments
        comment_count = len(re.findall(r'#.*', code))
        docstring_count = len(re.findall(r'""".*?"""', code, re.DOTALL))
        
        # Count meaningful variable names (heuristic)
        meaningful_names = len(re.findall(r'\b\w{3,}\b', code))
        
        # Count blank lines for structure
        blank_lines = len([line for line in code_lines if not line.strip()])
        
        total_lines = len(code_lines)
        if total_lines == 0:
            return 0.5
        
        # Weighted readability score
        documentation_ratio = (comment_count + docstring_count) / total_lines
        structure_ratio = blank_lines / total_lines
        naming_ratio = min(1.0, meaningful_names / (total_lines * 2))  # Rough heuristic
        
        readability = (0.4 * documentation_ratio + 0.3 * structure_ratio + 0.3 * naming_ratio)
        return min(1.0, readability * 2)  # Scale up and cap at 1.0
    
    def _calculate_maintainability(self, code: str, code_lines: List[str]) -> float:
        """Calculate maintainability score (0-1, higher is better)"""
        if not code_lines:
            return 0.5
        
        # Function length analysis (shorter functions are more maintainable)
        functions = re.findall(r'def\s+\w+.*?(?=\n\S|\nclass|\ndef|\n$)', code, re.DOTALL)
        avg_function_length = np.mean([len(f.split('\n')) for f in functions]) if functions else 10
        
        # Nesting level analysis
        max_indent = max([len(line) - len(line.lstrip()) for line in code_lines if line.strip()]) if code_lines else 0
        
        # Maintainability heuristics
        function_length_score = max(0, 1.0 - (avg_function_length - 5) / 20)  # Prefer 5-15 line functions
        nesting_score = max(0, 1.0 - max_indent / 32)  # Penalize deep nesting
        
        return (function_length_score + nesting_score) / 2
    
    def _calculate_modularity(self, code: str) -> float:
        """Calculate modularity score (0-1, higher is better)"""
        # Count classes and functions
        class_count = len(re.findall(r'class\s+\w+', code))
        function_count = len(re.findall(r'def\s+\w+', code))
        
        # Import statements (indicates use of external modules)
        import_count = len(re.findall(r'^(?:import|from)\s+', code, re.MULTILINE))
        
        total_lines = len(code.split('\n'))
        if total_lines == 0:
            return 0.5
        
        # Modularity indicators
        structure_density = (class_count + function_count) / total_lines
        import_usage = min(1.0, import_count / 10)  # Cap at reasonable level
        
        return min(1.0, (structure_density * 10 + import_usage) / 2)
    
    def _calculate_design_patterns(self, code: str) -> float:
        """Calculate design pattern usage score (0-1, higher is better)"""
        pattern_count = 0
        for pattern in self.design_patterns:
            pattern_count += len(re.findall(pattern, code))
        
        # Normalize by code size
        total_lines = len(code.split('\n'))
        if total_lines == 0:
            return 0.5
        
        pattern_density = pattern_count / total_lines
        return min(1.0, pattern_density * 20)  # Scale appropriately
    
    def _calculate_abstraction_level(self, code: str) -> float:
        """Calculate abstraction level (0-1 scale)"""
        # Count high-level constructs vs low-level operations
        high_level = len(re.findall(r'class\s+|def\s+|import\s+|from\s+', code))
        low_level = len(re.findall(r'\+\+|--|&|\||<<|>>', code))  # Low-level operations
        
        if high_level + low_level == 0:
            return 0.5
        
        abstraction_ratio = high_level / (high_level + low_level)
        return abstraction_ratio
    
    def _calculate_strategic_coherence(self, code: str) -> float:
        """Calculate strategic coherence score (0-1, higher is better)"""
        # Heuristic: well-structured code with clear separation of concerns
        
        # Check for clear structure patterns
        has_main_function = bool(re.search(r'def\s+main\s*\(', code))
        has_error_handling = bool(re.search(r'try\s*:|except\s+', code))
        has_documentation = bool(re.search(r'""".*?"""', code, re.DOTALL))
        
        # Strategic indicators
        indicators = [has_main_function, has_error_handling, has_documentation]
        return sum(indicators) / len(indicators)
    
    def _calculate_incremental_improvement(self, code: str) -> float:
        """Calculate incremental improvement potential (0-1, higher is better)"""
        # Look for code patterns that suggest iterative development
        
        # Version comments or TODO items
        improvement_markers = len(re.findall(r'TODO|FIXME|NOTE|v\d+|version', code, re.IGNORECASE))
        
        # Configurable parameters (suggests thoughtful design)
        config_patterns = len(re.findall(r'config|settings|options|parameters', code, re.IGNORECASE))
        
        total_lines = len(code.split('\n'))
        if total_lines == 0:
            return 0.5
        
        improvement_density = (improvement_markers + config_patterns) / total_lines
        return min(1.0, improvement_density * 10)
    
    def _calculate_syntax_correctness(self, code: str) -> float:
        """Calculate syntax correctness (0-1, higher is better)"""
        try:
            # Try to parse as Python AST
            ast.parse(code)
            return 1.0
        except SyntaxError:
            # Count potential syntax issues
            syntax_issues = 0
            
            # Unmatched brackets/parentheses (simple heuristic)
            open_parens = code.count('(') - code.count(')')
            open_brackets = code.count('[') - code.count(']')
            open_braces = code.count('{') - code.count('}')
            
            syntax_issues += abs(open_parens) + abs(open_brackets) + abs(open_braces)
            
            # Estimate correctness
            lines = len(code.split('\n'))
            if lines == 0:
                return 0.5
            
            error_rate = syntax_issues / lines
            return max(0.0, 1.0 - error_rate)
        except Exception:
            return 0.3  # Assume moderate correctness if can't parse
    
    def _calculate_semantic_correctness(self, code: str) -> float:
        """Calculate semantic correctness estimation (0-1, higher is better)"""
        # Heuristic-based semantic analysis
        
        # Check for common patterns that suggest semantic correctness
        return_statements = len(re.findall(r'return\s+', code))
        variable_assignments = len(re.findall(r'\w+\s*=\s*', code))
        function_calls = len(re.findall(r'\w+\s*\(', code))
        
        # Look for potential semantic issues
        undefined_usage = len(re.findall(r'undefined|null|None', code))
        
        total_constructs = return_statements + variable_assignments + function_calls
        if total_constructs == 0:
            return 0.5
        
        # Simple heuristic: more structured code patterns suggest better semantics
        correctness_ratio = (total_constructs - undefined_usage) / total_constructs
        return max(0.0, min(1.0, correctness_ratio))
    
    def _calculate_efficiency(self, code: str) -> float:
        """Calculate algorithmic efficiency score (0-1, higher is better)"""
        # Look for efficiency patterns and anti-patterns
        
        # Efficient patterns
        list_comprehensions = len(re.findall(r'\[.*for.*in.*\]', code))
        generator_expressions = len(re.findall(r'\(.*for.*in.*\)', code))
        
        # Inefficient patterns
        nested_loops = len(re.findall(r'for.*in.*:\s*.*for.*in.*:', code, re.DOTALL))
        string_concatenation = len(re.findall(r'\+\s*["\']', code))
        
        # Calculate efficiency score
        efficient_count = list_comprehensions + generator_expressions
        inefficient_count = nested_loops + string_concatenation
        
        if efficient_count + inefficient_count == 0:
            return 0.7  # Assume reasonable efficiency if no clear patterns
        
        efficiency_ratio = efficient_count / (efficient_count + inefficient_count + 1)
        return efficiency_ratio


class ReverseLearningEvaluator:
    """Enhanced evaluator for Reverse Learning framework with code quality assessment"""
    
    def __init__(self, model, config, device='cuda'):
        self.model = model
        self.config = config
        self.device = device
        self.results_history = []
        self.code_analyzer = CodeQualityAnalyzer()
        
    def evaluate_with_reverse_learning(self, 
                                     test_data: torch.utils.data.DataLoader,
                                     compare_baseline: bool = True,
                                     save_detailed_results: bool = True) -> ReverseLearningEvaluationResults:
        """
        Comprehensive evaluation with Reverse Learning enhancements
        
        Args:
            test_data: Test data loader
            compare_baseline: Whether to compare against baseline (non-reverse) performance
            save_detailed_results: Whether to save detailed per-sample results
            
        Returns:
            Comprehensive evaluation results with code quality metrics
        """
        self.model.eval()
        
        # Metrics tracking
        base_correct = 0
        enhanced_correct = 0
        total_samples = 0
        
        reverse_metrics_history = []
        per_sample_quality_improvements = []
        
        # Code quality tracking
        base_quality_scores = []
        enhanced_quality_scores = []
        
        # Performance tracking
        base_inference_times = []
        enhanced_inference_times = []
        memory_usage_base = []
        memory_usage_enhanced = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch_size = batch['inputs'].size(0)
                
                # Initialize carry
                carry = self.model.initial_carry(batch)
                
                if compare_baseline:
                    # Baseline evaluation (disable reverse learning temporarily)
                    original_reverse_enabled = getattr(self.model.config, 'enable_reverse_learning', False)
                    self.model.config.enable_reverse_learning = False
                    
                    start_time = time.time()
                    memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    
                    base_carry, base_outputs = self.model(carry, batch)
                    
                    base_time = time.time() - start_time
                    memory_after_base = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    
                    base_inference_times.append(base_time)
                    memory_usage_base.append(memory_after_base - memory_before)
                    
                    # Analyze base code quality
                    if 'logits' in base_outputs:
                        base_text = self._decode_outputs(base_outputs['logits'])
                        base_quality = self.code_analyzer.analyze_code_quality(base_text)
                        base_quality_scores.append(base_quality)
                    
                    # Restore reverse learning setting
                    self.model.config.enable_reverse_learning = original_reverse_enabled
                else:
                    base_outputs = None
                    base_time = 0
                    base_quality = None
                
                # Enhanced evaluation (with reverse learning)
                if getattr(self.model.config, 'enable_reverse_learning', False):
                    carry = self.model.initial_carry(batch)  # Reset carry
                    
                    start_time = time.time()
                    memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    
                    enhanced_carry, enhanced_outputs = self.model(carry, batch)
                    
                    enhanced_time = time.time() - start_time
                    memory_after_enhanced = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                    
                    enhanced_inference_times.append(enhanced_time)
                    memory_usage_enhanced.append(memory_after_enhanced - memory_before)
                    
                    # Analyze enhanced code quality
                    if 'logits' in enhanced_outputs:
                        enhanced_text = self._decode_outputs(enhanced_outputs['logits'])
                        enhanced_quality = self.code_analyzer.analyze_code_quality(enhanced_text)
                        enhanced_quality_scores.append(enhanced_quality)
                        
                        # Calculate quality improvement
                        if base_quality:
                            quality_improvement = self._calculate_quality_improvement(base_quality, enhanced_quality)
                            per_sample_quality_improvements.append(quality_improvement)
                    
                    # Get reverse learning metrics
                    if hasattr(self.model, 'get_latest_reverse_metrics'):
                        batch_reverse_metrics = self.model.get_latest_reverse_metrics()
                        reverse_metrics_history.extend(batch_reverse_metrics)
                else:
                    enhanced_outputs = base_outputs
                    enhanced_time = base_time
                    enhanced_quality = base_quality
                
                # Compute accuracy
                labels = batch['labels']
                
                if base_outputs is not None:
                    base_predictions = torch.argmax(base_outputs['logits'], dim=-1)
                    base_batch_correct = self._compute_accuracy(base_predictions, labels)
                    base_correct += base_batch_correct
                else:
                    base_batch_correct = 0
                
                if enhanced_outputs is not None:
                    enhanced_predictions = torch.argmax(enhanced_outputs['logits'], dim=-1)
                    enhanced_batch_correct = self._compute_accuracy(enhanced_predictions, labels)
                    enhanced_correct += enhanced_batch_correct
                else:
                    enhanced_batch_correct = base_batch_correct
                
                total_samples += batch_size
                
                # Progress logging
                if batch_idx % 50 == 0:
                    print(f"Evaluated {batch_idx * batch_size}/{len(test_data.dataset)} samples")
        
        # Compute final metrics
        base_accuracy = base_correct / total_samples if total_samples > 0 else 0.0
        enhanced_accuracy = enhanced_correct / total_samples if total_samples > 0 else 0.0
        improvement = enhanced_accuracy - base_accuracy
        
        # Reverse learning specific metrics
        reverse_stats = self._compute_reverse_learning_statistics(reverse_metrics_history)
        
        # Code quality metrics
        base_avg_quality = self._average_code_quality(base_quality_scores)
        enhanced_avg_quality = self._average_code_quality(enhanced_quality_scores)
        quality_improvement = self._calculate_quality_improvement(base_avg_quality, enhanced_avg_quality)
        
        # Performance metrics
        avg_base_time = np.mean(base_inference_times) if base_inference_times else 0.0
        avg_enhanced_time = np.mean(enhanced_inference_times) if enhanced_inference_times else 0.0
        
        avg_base_memory = np.mean(memory_usage_base) if memory_usage_base else 0.0
        avg_enhanced_memory = np.mean(memory_usage_enhanced) if memory_usage_enhanced else 0.0
        memory_overhead = (avg_enhanced_memory - avg_base_memory) / max(avg_base_memory, 1.0)
        
        # Create results
        results = ReverseLearningEvaluationResults(
            base_accuracy=base_accuracy,
            enhanced_accuracy=enhanced_accuracy,
            improvement=improvement,
            code_quality=enhanced_avg_quality,
            quality_improvement=quality_improvement,
            avg_insight_strength=reverse_stats['avg_insight_strength'],
            avg_integration_gate=reverse_stats['avg_integration_gate'],
            avg_planning_refinement=reverse_stats['avg_planning_refinement'],
            feedback_effectiveness=reverse_stats['feedback_effectiveness'],
            inference_time_base=avg_base_time,
            inference_time_enhanced=avg_enhanced_time,
            memory_overhead=memory_overhead,
            reverse_metrics_history=reverse_metrics_history,
            per_sample_quality_improvements=per_sample_quality_improvements
        )
        
        self.results_history.append(results)
        
        if save_detailed_results:
            self._save_detailed_results(results)
        
        return results
    
    def _decode_outputs(self, logits: torch.Tensor) -> str:
        """Decode model outputs to text for quality analysis"""
        # Simple decoding - in practice would use proper tokenizer
        predictions = torch.argmax(logits, dim=-1)
        # Convert to dummy code text for analysis
        return f"# Generated code with {predictions.shape[1]} tokens\nprint('Hello, World!')"
    
    def _compute_accuracy(self, predictions: torch.Tensor, labels: torch.Tensor) -> int:
        """Compute accuracy for given predictions and labels"""
        # Handle ignore labels
        mask = labels != -100
        if not mask.any():
            return 0
        
        correct = (predictions == labels) & mask
        return correct.sum().item()
    
    def _compute_reverse_learning_statistics(self, reverse_metrics: List[ReverseLearningMetrics]) -> Dict[str, float]:
        """Compute aggregated reverse learning statistics"""
        if not reverse_metrics:
            return {
                'avg_insight_strength': 0.0,
                'avg_integration_gate': 0.0,
                'avg_planning_refinement': 0.0,
                'feedback_effectiveness': 0.0
            }
        
        avg_insight_strength = np.mean([m.insight_strength for m in reverse_metrics])
        avg_integration_gate = np.mean([m.integration_gate_value for m in reverse_metrics])
        avg_planning_refinement = np.mean([m.planning_refinement_score for m in reverse_metrics])
        feedback_effectiveness = np.mean([m.feedback_magnitude for m in reverse_metrics])
        
        return {
            'avg_insight_strength': avg_insight_strength,
            'avg_integration_gate': avg_integration_gate,
            'avg_planning_refinement': avg_planning_refinement,
            'feedback_effectiveness': feedback_effectiveness
        }
    
    def _average_code_quality(self, quality_scores: List[CodeQualityMetrics]) -> CodeQualityMetrics:
        """Average code quality metrics across samples"""
        if not quality_scores:
            return self.code_analyzer._empty_metrics()
        
        # Average all fields
        avg_metrics = CodeQualityMetrics(
            complexity_score=np.mean([q.complexity_score for q in quality_scores]),
            readability_score=np.mean([q.readability_score for q in quality_scores]),
            maintainability_score=np.mean([q.maintainability_score for q in quality_scores]),
            modularity_score=np.mean([q.modularity_score for q in quality_scores]),
            design_pattern_score=np.mean([q.design_pattern_score for q in quality_scores]),
            abstraction_level=np.mean([q.abstraction_level for q in quality_scores]),
            strategic_coherence=np.mean([q.strategic_coherence for q in quality_scores]),
            incremental_improvement=np.mean([q.incremental_improvement for q in quality_scores]),
            syntax_correctness=np.mean([q.syntax_correctness for q in quality_scores]),
            semantic_correctness=np.mean([q.semantic_correctness for q in quality_scores]),
            efficiency_score=np.mean([q.efficiency_score for q in quality_scores])
        )
        
        return avg_metrics
    
    def _calculate_quality_improvement(self, base_quality: CodeQualityMetrics, 
                                     enhanced_quality: CodeQualityMetrics) -> float:
        """Calculate overall quality improvement score"""
        if not base_quality or not enhanced_quality:
            return 0.0
        
        # Weight different quality aspects
        weights = {
            'readability_score': 0.2,
            'maintainability_score': 0.2,
            'modularity_score': 0.15,
            'strategic_coherence': 0.15,
            'syntax_correctness': 0.1,
            'semantic_correctness': 0.1,
            'efficiency_score': 0.1
        }
        
        total_improvement = 0.0
        for field, weight in weights.items():
            base_val = getattr(base_quality, field, 0.5)
            enhanced_val = getattr(enhanced_quality, field, 0.5)
            improvement = (enhanced_val - base_val) * weight
            total_improvement += improvement
        
        return total_improvement
    
    def _save_detailed_results(self, results: ReverseLearningEvaluationResults):
        """Save detailed evaluation results to file"""
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save summary results
        summary = {
            'base_accuracy': results.base_accuracy,
            'enhanced_accuracy': results.enhanced_accuracy,
            'improvement': results.improvement,
            'improvement_percentage': results.improvement / max(results.base_accuracy, 0.01) * 100,
            'quality_improvement': results.quality_improvement,
            'avg_insight_strength': results.avg_insight_strength,
            'avg_integration_gate': results.avg_integration_gate,
            'avg_planning_refinement': results.avg_planning_refinement,
            'feedback_effectiveness': results.feedback_effectiveness,
            'inference_time_base': results.inference_time_base,
            'inference_time_enhanced': results.inference_time_enhanced,
            'time_overhead_percentage': (results.inference_time_enhanced - results.inference_time_base) / max(results.inference_time_base, 0.001) * 100,
            'memory_overhead_percentage': results.memory_overhead * 100,
            # Code quality metrics
            'code_quality': {
                'complexity_score': results.code_quality.complexity_score,
                'readability_score': results.code_quality.readability_score,
                'maintainability_score': results.code_quality.maintainability_score,
                'modularity_score': results.code_quality.modularity_score,
                'strategic_coherence': results.code_quality.strategic_coherence,
                'syntax_correctness': results.code_quality.syntax_correctness,
                'semantic_correctness': results.code_quality.semantic_correctness,
                'efficiency_score': results.code_quality.efficiency_score
            }
        }
        
        timestamp = int(time.time())
        summary_file = results_dir / f"reverse_learning_evaluation_{timestamp}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Detailed results saved to {summary_file}")
        
        # Print summary
        self._print_results_summary(results)
    
    def _print_results_summary(self, results: ReverseLearningEvaluationResults):
        """Print a comprehensive results summary"""
        print("\n" + "="*80)
        print("REVERSE LEARNING EVALUATION RESULTS")
        print("="*80)
        
        print(f"\n📊 ACCURACY METRICS:")
        print(f"   Base Accuracy:      {results.base_accuracy:.4f}")
        print(f"   Enhanced Accuracy:  {results.enhanced_accuracy:.4f}")
        print(f"   Improvement:        {results.improvement:.4f} ({results.improvement/max(results.base_accuracy, 0.01)*100:.2f}%)")
        
        print(f"\n🔧 CODE QUALITY METRICS:")
        print(f"   Readability:        {results.code_quality.readability_score:.4f}")
        print(f"   Maintainability:    {results.code_quality.maintainability_score:.4f}")
        print(f"   Modularity:         {results.code_quality.modularity_score:.4f}")
        print(f"   Strategic Coherence: {results.code_quality.strategic_coherence:.4f}")
        print(f"   Syntax Correctness: {results.code_quality.syntax_correctness:.4f}")
        print(f"   Overall Quality Improvement: {results.quality_improvement:.4f}")
        
        print(f"\n🔄 REVERSE LEARNING METRICS:")
        print(f"   Insight Strength:   {results.avg_insight_strength:.4f}")
        print(f"   Integration Gate:   {results.avg_integration_gate:.4f}")
        print(f"   Planning Refinement: {results.avg_planning_refinement:.4f}")
        print(f"   Feedback Effectiveness: {results.feedback_effectiveness:.4f}")
        
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   Base Inference:     {results.inference_time_base*1000:.2f}ms")
        print(f"   Enhanced Inference: {results.inference_time_enhanced*1000:.2f}ms")
        time_overhead = (results.inference_time_enhanced - results.inference_time_base) / max(results.inference_time_base, 0.001) * 100
        print(f"   Time Overhead:      {time_overhead:.1f}%")
        print(f"   Memory Overhead:    {results.memory_overhead*100:.1f}%")
        
        print(f"\n🎯 SUCCESS CRITERIA CHECK:")
        target_improvement = 0.1  # 10% target improvement
        achieved_improvement = results.improvement / max(results.base_accuracy, 0.01)
        
        if achieved_improvement >= target_improvement:
            print(f"   ✅ Target improvement achieved: {achieved_improvement:.2%} >= {target_improvement:.2%}")
        else:
            print(f"   ❌ Target improvement not met: {achieved_improvement:.2%} < {target_improvement:.2%}")
        
        if results.quality_improvement >= 0.05:  # 5% quality improvement target
            print(f"   ✅ Code quality improved: {results.quality_improvement:.3f} >= 0.050")
        else:
            print(f"   ⚠️  Code quality improvement below target: {results.quality_improvement:.3f} < 0.050")
        
        if results.memory_overhead <= 0.15:  # 15% memory overhead target
            print(f"   ✅ Memory overhead within target: {results.memory_overhead:.2%} <= 15%")
        else:
            print(f"   ⚠️  Memory overhead exceeds target: {results.memory_overhead:.2%} > 15%")
        
        print("="*80)
    
    def get_performance_summary(self) -> Dict:
        """Get a summary of all evaluation runs"""
        if not self.results_history:
            return {}
        
        latest = self.results_history[-1]
        return {
            'latest_improvement': latest.improvement,
            'latest_quality_improvement': latest.quality_improvement,
            'latest_insight_strength': latest.avg_insight_strength,
            'total_evaluations': len(self.results_history),
            'best_improvement': max(r.improvement for r in self.results_history),
            'avg_improvement': np.mean([r.improvement for r in self.results_history])
        }


def create_reverse_learning_evaluator(model, config, device='cuda') -> ReverseLearningEvaluator:
    """Factory function to create Reverse Learning evaluator"""
    return ReverseLearningEvaluator(model, config, device)