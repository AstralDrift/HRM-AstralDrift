"""
Diff-Based Training Data Generator for Polyglot Benchmark

This module generates high-quality diff-based training examples for code editing tasks,
following the Aider-style search-replace paradigm used in the Polyglot benchmark.
"""

import os
import re
import ast
import random
import logging
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
import difflib
import copy

from .polyglot_benchmark_extractor import PolyglotProblem, ProblemFile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiffEditType(Enum):
    """Types of diff edits that can be generated"""
    BUG_FIX = "bug_fix"                    # Fix syntax/logic errors
    FEATURE_ADD = "feature_add"            # Add new functionality
    REFACTOR = "refactor"                  # Improve code structure
    OPTIMIZE = "optimize"                  # Performance improvements
    STYLE_FIX = "style_fix"               # Code style improvements
    TEST_DRIVEN = "test_driven"           # Implement based on tests

@dataclass
class DiffEdit:
    """Represents a single diff-based edit operation"""
    edit_type: DiffEditType
    language: str
    problem_id: str
    
    # Search-replace operation
    search_text: str
    replace_text: str
    
    # Context information
    original_code: str
    modified_code: str
    context_before: str = ""
    context_after: str = ""
    
    # Metadata
    description: str = ""
    difficulty: str = "medium"  # easy, medium, hard
    edit_distance: int = 0
    line_changes: int = 0

@dataclass 
class DiffTrainingExample:
    """Complete training example for diff-based editing"""
    problem_description: str
    edit_instruction: str
    original_code: str
    target_code: str
    diff_operations: List[DiffEdit]
    language: str
    complexity_score: float
    test_cases: Optional[List[str]] = None

class CodeAnalyzer:
    """Analyzes code structure for intelligent diff generation"""
    
    def __init__(self):
        self.function_patterns = {
            "python": r'def\s+(\w+)\s*\([^)]*\):',
            "javascript": r'(?:function\s+(\w+)|const\s+(\w+)\s*=.*?=>)',
            "java": r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\([^)]*\)',
            "cpp": r'(?:\w+\s+)?(\w+)\s*\([^)]*\)\s*{',
            "go": r'func\s+(\w+)\s*\([^)]*\)',
            "rust": r'(?:pub\s+)?fn\s+(\w+)\s*\([^)]*\)'
        }
    
    def extract_functions(self, code: str, language: str) -> List[Dict]:
        """Extract function definitions with their locations"""
        functions = []
        if language not in self.function_patterns:
            return functions
        
        pattern = self.function_patterns[language]
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            match = re.search(pattern, line)
            if match:
                func_name = match.group(1)
                # Find function body (simplified)
                func_start = i
                func_end = self._find_function_end(lines, i, language)
                
                functions.append({
                    "name": func_name,
                    "start_line": func_start,
                    "end_line": func_end,
                    "content": '\n'.join(lines[func_start:func_end+1])
                })
        
        return functions
    
    def _find_function_end(self, lines: List[str], start_line: int, language: str) -> int:
        """Find the end line of a function (simplified heuristic)"""
        indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
        
        if language == "python":
            # Python: find next line with same or less indentation
            for i in range(start_line + 1, len(lines)):
                line = lines[i]
                if line.strip() and len(line) - len(line.lstrip()) <= indent_level:
                    return i - 1
        elif language in ["java", "cpp", "javascript", "go", "rust"]:
            # Brace-based languages: count braces
            brace_count = 0
            for i in range(start_line, len(lines)):
                line = lines[i]
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0 and i > start_line:
                    return i
        
        return len(lines) - 1
    
    def find_syntax_errors(self, code: str, language: str) -> List[Dict]:
        """Find potential syntax errors for bug fix generation"""
        errors = []
        
        if language == "python":
            try:
                ast.parse(code)
            except SyntaxError as e:
                errors.append({
                    "type": "syntax_error",
                    "line": e.lineno,
                    "message": str(e),
                    "text": e.text
                })
        
        # Language-specific common errors
        common_errors = {
            "python": [
                (r':\s*$', r'missing colon'),
                (r'def\s+\w+\([^)]*\s*$', r'missing closing parenthesis'),
                (r'print\s+[^(]', r'missing parentheses in print statement')
            ],
            "javascript": [
                (r'function\s+\w+\([^)]*\s*$', r'missing closing parenthesis'),
                (r'if\s*\([^)]*\s*$', r'missing closing parenthesis'),
                (r'}\s*$', r'missing semicolon')
            ],
            "java": [
                (r'class\s+\w+\s*{', r'missing extends or implements'),
                (r'public\s+\w+\s+\w+\([^)]*\)\s*$', r'missing method body'),
            ]
        }
        
        if language in common_errors:
            lines = code.split('\n')
            for i, line in enumerate(lines):
                for pattern, error_type in common_errors[language]:
                    if re.search(pattern, line):
                        errors.append({
                            "type": error_type,
                            "line": i + 1,
                            "text": line
                        })
        
        return errors

class DiffBasedTrainingGenerator:
    """
    Main class for generating diff-based training examples
    """
    
    def __init__(self, problems: Dict[str, PolyglotProblem]):
        """
        Initialize generator with extracted Polyglot problems
        
        Args:
            problems: Dictionary of problem_id -> PolyglotProblem
        """
        self.problems = problems
        self.code_analyzer = CodeAnalyzer()
        self.generated_examples = []
        
        # Templates for different edit types
        self.edit_templates = self._initialize_edit_templates()
        
        # Quality filters
        self.min_edit_distance = 5
        self.max_edit_distance = 200
        
    def _initialize_edit_templates(self) -> Dict[DiffEditType, Dict]:
        """Initialize templates for different types of edits"""
        return {
            DiffEditType.BUG_FIX: {
                "description_templates": [
                    "Fix the syntax error in the {function_name} function",
                    "Correct the missing {error_type} in line {line_number}",
                    "Fix the incorrect {language} syntax"
                ],
                "common_fixes": {
                    "python": [
                        ("def function_name(:", "def function_name():"),
                        ("if condition", "if condition:"),
                        ("print statement", "print(statement)")
                    ],
                    "javascript": [
                        ("function name(", "function name() {"),
                        ("if (condition", "if (condition) {"),
                        ("var x = ", "var x = value;")
                    ]
                }
            },
            DiffEditType.REFACTOR: {
                "description_templates": [
                    "Refactor the {function_name} function to use {pattern}",
                    "Convert this loop to use {language} best practices",
                    "Extract common logic into a separate function"
                ],
                "patterns": {
                    "python": ["list comprehension", "generator expression", "with statement"],
                    "javascript": ["arrow functions", "destructuring", "async/await"],
                    "java": ["streams", "optional", "lambda expressions"]
                }
            },
            DiffEditType.FEATURE_ADD: {
                "description_templates": [
                    "Add error handling to the {function_name} function",
                    "Implement the missing {feature} functionality",
                    "Add validation for input parameters"
                ]
            },
            DiffEditType.TEST_DRIVEN: {
                "description_templates": [
                    "Implement the function to pass the given tests",
                    "Complete the implementation based on test requirements",
                    "Fill in the missing logic to satisfy test cases"
                ]
            }
        }
    
    def generate_bug_fix_examples(self, problem: PolyglotProblem, 
                                 language: str, count: int = 5) -> List[DiffTrainingExample]:
        """Generate bug fix training examples"""
        examples = []
        
        if language not in problem.files_by_language:
            return examples
        
        implementation_files = [f for f in problem.files_by_language[language] 
                              if f.file_type == "implementation"]
        
        for impl_file in implementation_files:
            original_code = impl_file.content
            
            # Generate various types of bugs
            bug_variants = self._generate_bug_variants(original_code, language)
            
            for bug_code, fix_description in bug_variants[:count]:
                diff_edit = DiffEdit(
                    edit_type=DiffEditType.BUG_FIX,
                    language=language,
                    problem_id=problem.problem_id,
                    search_text=self._extract_buggy_section(bug_code, original_code),
                    replace_text=self._extract_fixed_section(original_code, bug_code),
                    original_code=bug_code,
                    modified_code=original_code,
                    description=fix_description
                )
                
                training_example = DiffTrainingExample(
                    problem_description=problem.description,
                    edit_instruction=fix_description,
                    original_code=bug_code,
                    target_code=original_code,
                    diff_operations=[diff_edit],
                    language=language,
                    complexity_score=problem.complexity_score
                )
                
                examples.append(training_example)
        
        return examples
    
    def _generate_bug_variants(self, code: str, language: str) -> List[Tuple[str, str]]:
        """Generate different buggy versions of correct code"""
        variants = []
        
        # Common bug patterns by language
        bug_patterns = {
            "python": [
                (r'(\s+)def\s+(\w+)\s*\([^)]*\):', r'\1def \2([^)]*', "Missing closing parenthesis in function definition"),
                (r'(\s+)if\s+([^:]+):', r'\1if \2', "Missing colon after if statement"),
                (r'print\s*\(([^)]+)\)', r'print \1', "Missing parentheses in print statement"),
                (r'(\s+)for\s+(\w+)\s+in\s+([^:]+):', r'\1for \2 in \3', "Missing colon in for loop"),
            ],
            "javascript": [
                (r'function\s+(\w+)\s*\([^)]*\)\s*{', r'function \1([^)]*', "Missing opening brace in function"),
                (r'if\s*\(([^)]+)\)\s*{', r'if (\1', "Missing opening brace in if statement"),
                (r'var\s+(\w+)\s*=\s*([^;]+);', r'var \1 = \2', "Missing semicolon"),
                (r'}\s*else\s*{', r'} else', "Missing opening brace in else"),
            ],
            "java": [
                (r'public\s+(\w+)\s+(\w+)\s*\([^)]*\)\s*{', r'public \1 \2([^)]*', "Missing opening brace in method"),
                (r'if\s*\(([^)]+)\)\s*{', r'if (\1', "Missing opening brace in if statement"),
                (r'}\s*catch\s*\(([^)]+)\)\s*{', r'} catch(\1', "Missing opening brace in catch block"),
            ]
        }
        
        if language in bug_patterns:
            for pattern, replacement, description in bug_patterns[language]:
                buggy_code = re.sub(pattern, replacement, code)
                if buggy_code != code:
                    variants.append((buggy_code, description))
        
        return variants
    
    def generate_refactor_examples(self, problem: PolyglotProblem, 
                                  language: str, count: int = 3) -> List[DiffTrainingExample]:
        """Generate refactoring training examples"""
        examples = []
        
        if language not in problem.files_by_language:
            return examples
        
        implementation_files = [f for f in problem.files_by_language[language] 
                              if f.file_type == "implementation"]
        
        for impl_file in implementation_files:
            original_code = impl_file.content
            
            # Generate refactoring opportunities
            refactor_ops = self._identify_refactor_opportunities(original_code, language)
            
            for refactor_op in refactor_ops[:count]:
                refactored_code = self._apply_refactoring(original_code, refactor_op, language)
                
                if refactored_code != original_code:
                    diff_edit = DiffEdit(
                        edit_type=DiffEditType.REFACTOR,
                        language=language,
                        problem_id=problem.problem_id,
                        search_text=refactor_op["original_pattern"],
                        replace_text=refactor_op["refactored_pattern"],
                        original_code=original_code,
                        modified_code=refactored_code,
                        description=refactor_op["description"]
                    )
                    
                    training_example = DiffTrainingExample(
                        problem_description=problem.description,
                        edit_instruction=refactor_op["description"],
                        original_code=original_code,
                        target_code=refactored_code,
                        diff_operations=[diff_edit],
                        language=language,
                        complexity_score=problem.complexity_score
                    )
                    
                    examples.append(training_example)
        
        return examples
    
    def _identify_refactor_opportunities(self, code: str, language: str) -> List[Dict]:
        """Identify code that can be refactored"""
        opportunities = []
        
        # Language-specific refactoring patterns
        refactor_patterns = {
            "python": [
                {
                    "name": "list_comprehension",
                    "pattern": r'(\s+)(\w+)\s*=\s*\[\]\s*\n(\s+)for\s+(\w+)\s+in\s+([^:]+):\s*\n(\s+)\2\.append\(([^)]+)\)',
                    "replacement": r'\1\2 = [\7 for \4 in \5]',
                    "description": "Convert loop to list comprehension"
                },
                {
                    "name": "with_statement", 
                    "pattern": r'(\s+)(\w+)\s*=\s*open\(([^)]+)\)\s*\n(.*?)\n(\s+)\2\.close\(\)',
                    "replacement": r'\1with open(\3) as \2:\n\4',
                    "description": "Use with statement for file handling"
                }
            ],
            "javascript": [
                {
                    "name": "arrow_function",
                    "pattern": r'function\s*\(([^)]*)\)\s*{\s*return\s+([^;]+);\s*}',
                    "replacement": r'(\1) => \2',
                    "description": "Convert to arrow function"
                },
                {
                    "name": "const_let",
                    "pattern": r'var\s+(\w+)',
                    "replacement": r'const \1',
                    "description": "Use const instead of var"
                }
            ]
        }
        
        if language in refactor_patterns:
            for pattern_info in refactor_patterns[language]:
                matches = re.finditer(pattern_info["pattern"], code, re.MULTILINE | re.DOTALL)
                for match in matches:
                    opportunities.append({
                        "type": pattern_info["name"],
                        "original_pattern": match.group(0),
                        "refactored_pattern": re.sub(pattern_info["pattern"], 
                                                   pattern_info["replacement"], 
                                                   match.group(0)),
                        "description": pattern_info["description"],
                        "start": match.start(),
                        "end": match.end()
                    })
        
        return opportunities
    
    def _apply_refactoring(self, code: str, refactor_op: Dict, language: str) -> str:
        """Apply a refactoring operation to code"""
        return code.replace(refactor_op["original_pattern"], 
                          refactor_op["refactored_pattern"])
    
    def generate_test_driven_examples(self, problem: PolyglotProblem, 
                                    language: str) -> List[DiffTrainingExample]:
        """Generate test-driven development examples"""
        examples = []
        
        if language not in problem.files_by_language:
            return examples
        
        # Find test files and implementation files
        test_files = [f for f in problem.files_by_language[language] 
                     if f.file_type == "test"]
        impl_files = [f for f in problem.files_by_language[language] 
                     if f.file_type == "implementation"]
        
        if not test_files or not impl_files:
            return examples
        
        for test_file in test_files:
            for impl_file in impl_files:
                # Create stub implementation
                stub_code = self._create_stub_implementation(impl_file.content, language)
                
                if stub_code != impl_file.content:
                    diff_edit = DiffEdit(
                        edit_type=DiffEditType.TEST_DRIVEN,
                        language=language,
                        problem_id=problem.problem_id,
                        search_text=stub_code,
                        replace_text=impl_file.content,
                        original_code=stub_code,
                        modified_code=impl_file.content,
                        description=f"Implement function to pass tests in {test_file.filename}"
                    )
                    
                    training_example = DiffTrainingExample(
                        problem_description=problem.description,
                        edit_instruction=f"Implement the function to pass the tests",
                        original_code=stub_code,
                        target_code=impl_file.content,
                        diff_operations=[diff_edit],
                        language=language,
                        complexity_score=problem.complexity_score,
                        test_cases=self._extract_test_cases(test_file.content, language)
                    )
                    
                    examples.append(training_example)
        
        return examples
    
    def _create_stub_implementation(self, complete_code: str, language: str) -> str:
        """Create a stub version of the implementation"""
        stub_code = complete_code
        
        if language == "python":
            # Replace function bodies with pass
            stub_code = re.sub(r'(def\s+\w+\s*\([^)]*\):\s*\n)([^d].*?)(?=\ndef|\nclass|\Z)', 
                             r'\1    pass\n', stub_code, flags=re.DOTALL)
        elif language == "javascript":
            # Replace function bodies with return null
            stub_code = re.sub(r'(function\s+\w+\s*\([^)]*\)\s*{\s*)([^}].*?)(\s*})', 
                             r'\1return null;\3', stub_code, flags=re.DOTALL)
        elif language == "java":
            # Replace method bodies with return default values
            stub_code = re.sub(r'(public\s+\w+\s+\w+\s*\([^)]*\)\s*{\s*)([^}].*?)(\s*})', 
                             r'\1return null;\3', stub_code, flags=re.DOTALL)
        
        return stub_code
    
    def _extract_test_cases(self, test_code: str, language: str) -> List[str]:
        """Extract test cases from test file"""
        test_cases = []
        
        if language == "python":
            # Extract assert statements
            assertions = re.findall(r'assert\s+[^,\n]+(?:,\s*[^,\n]+)?', test_code)
            test_cases.extend(assertions)
        elif language == "javascript":
            # Extract expect statements
            expects = re.findall(r'expect\([^)]+\)\.to[^;]+;', test_code)
            test_cases.extend(expects)
        elif language == "java":
            # Extract assertEquals statements
            asserts = re.findall(r'assert\w+\([^)]+\);', test_code)
            test_cases.extend(asserts)
        
        return test_cases[:10]  # Limit to first 10 test cases
    
    def _extract_buggy_section(self, buggy_code: str, original_code: str) -> str:
        """Extract the specific section that contains the bug"""
        # Use difflib to find differences
        diff = list(difflib.unified_diff(
            buggy_code.splitlines(keepends=True),
            original_code.splitlines(keepends=True),
            n=2  # Context lines
        ))
        
        # Extract the buggy section (lines starting with -)
        buggy_lines = []
        for line in diff:
            if line.startswith('-') and not line.startswith('---'):
                buggy_lines.append(line[1:])  # Remove the - prefix
        
        return ''.join(buggy_lines).strip()
    
    def _extract_fixed_section(self, original_code: str, buggy_code: str) -> str:
        """Extract the specific section that contains the fix"""
        # Use difflib to find differences
        diff = list(difflib.unified_diff(
            buggy_code.splitlines(keepends=True),
            original_code.splitlines(keepends=True),
            n=2  # Context lines
        ))
        
        # Extract the fixed section (lines starting with +)
        fixed_lines = []
        for line in diff:
            if line.startswith('+') and not line.startswith('+++'):
                fixed_lines.append(line[1:])  # Remove the + prefix
        
        return ''.join(fixed_lines).strip()
    
    def generate_all_training_examples(self, max_examples_per_problem: int = 10) -> List[DiffTrainingExample]:
        """Generate comprehensive training examples for all problems"""
        logger.info("Generating diff-based training examples...")
        
        all_examples = []
        
        for problem_id, problem in self.problems.items():
            logger.debug(f"Processing problem: {problem_id}")
            
            for language in problem.files_by_language.keys():
                # Generate different types of examples
                bug_fix_examples = self.generate_bug_fix_examples(problem, language, count=3)
                refactor_examples = self.generate_refactor_examples(problem, language, count=2)
                test_driven_examples = self.generate_test_driven_examples(problem, language)
                
                # Combine and limit examples per problem
                problem_examples = bug_fix_examples + refactor_examples + test_driven_examples
                
                # Select best examples based on quality metrics
                selected_examples = self._select_best_examples(
                    problem_examples, max_examples_per_problem
                )
                
                all_examples.extend(selected_examples)
        
        self.generated_examples = all_examples
        logger.info(f"Generated {len(all_examples)} total training examples")
        
        return all_examples
    
    def _select_best_examples(self, examples: List[DiffTrainingExample], 
                            max_count: int) -> List[DiffTrainingExample]:
        """Select the best examples based on quality metrics"""
        if len(examples) <= max_count:
            return examples
        
        # Score examples based on various factors
        scored_examples = []
        for example in examples:
            score = self._calculate_example_quality_score(example)
            scored_examples.append((score, example))
        
        # Sort by score and take top examples
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [example for _, example in scored_examples[:max_count]]
    
    def _calculate_example_quality_score(self, example: DiffTrainingExample) -> float:
        """Calculate quality score for a training example"""
        score = 0.0
        
        # Complexity factor (prefer medium complexity)
        complexity = example.complexity_score
        if 0.3 <= complexity <= 0.7:
            score += 1.0
        else:
            score += 0.5
        
        # Edit distance factor (prefer meaningful but not too large changes)
        for diff_op in example.diff_operations:
            edit_distance = len(diff_op.search_text) + len(diff_op.replace_text)
            if self.min_edit_distance <= edit_distance <= self.max_edit_distance:
                score += 1.0
            else:
                score += 0.3
        
        # Diversity bonus for different edit types
        edit_types = set(diff_op.edit_type for diff_op in example.diff_operations)
        score += len(edit_types) * 0.2
        
        # Test case bonus
        if example.test_cases:
            score += 0.5
        
        return score
    
    def export_training_data(self, output_path: str, format: str = "json"):
        """Export generated training examples"""
        logger.info(f"Exporting training data to {output_path}")
        
        export_data = {
            "metadata": {
                "total_examples": len(self.generated_examples),
                "languages": list(set(ex.language for ex in self.generated_examples)),
                "edit_types": list(set(
                    diff_op.edit_type.value 
                    for ex in self.generated_examples 
                    for diff_op in ex.diff_operations
                ))
            },
            "examples": []
        }
        
        for example in self.generated_examples:
            example_data = {
                "problem_description": example.problem_description,
                "edit_instruction": example.edit_instruction,
                "original_code": example.original_code,
                "target_code": example.target_code,
                "language": example.language,
                "complexity_score": example.complexity_score,
                "test_cases": example.test_cases,
                "diff_operations": [
                    {
                        "edit_type": diff_op.edit_type.value,
                        "search_text": diff_op.search_text,
                        "replace_text": diff_op.replace_text,
                        "description": diff_op.description
                    }
                    for diff_op in example.diff_operations
                ]
            }
            export_data["examples"].append(example_data)
        
        if format == "json":
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Export complete: {len(self.generated_examples)} examples")
    
    def get_statistics(self) -> Dict:
        """Get statistics about generated training examples"""
        if not self.generated_examples:
            return {}
        
        stats = {
            "total_examples": len(self.generated_examples),
            "by_language": {},
            "by_edit_type": {},
            "complexity_distribution": {"low": 0, "medium": 0, "high": 0},
            "average_edit_distance": 0
        }
        
        total_edit_distance = 0
        for example in self.generated_examples:
            # Language stats
            lang = example.language
            stats["by_language"][lang] = stats["by_language"].get(lang, 0) + 1
            
            # Edit type stats
            for diff_op in example.diff_operations:
                edit_type = diff_op.edit_type.value
                stats["by_edit_type"][edit_type] = stats["by_edit_type"].get(edit_type, 0) + 1
            
            # Complexity distribution
            complexity = example.complexity_score
            if complexity < 0.3:
                stats["complexity_distribution"]["low"] += 1
            elif complexity > 0.7:
                stats["complexity_distribution"]["high"] += 1
            else:
                stats["complexity_distribution"]["medium"] += 1
            
            # Edit distance
            for diff_op in example.diff_operations:
                total_edit_distance += len(diff_op.search_text) + len(diff_op.replace_text)
        
        stats["average_edit_distance"] = total_edit_distance / len(self.generated_examples)
        
        return stats

def main():
    """Main function for testing the generator"""
    import argparse
    from .polyglot_benchmark_extractor import PolyglotBenchmarkExtractor
    
    parser = argparse.ArgumentParser(description="Generate diff-based training data")
    parser.add_argument("--problems-json", required=True,
                       help="JSON file with extracted problems")
    parser.add_argument("--output", default="diff_training_data.json",
                       help="Output training data file")
    parser.add_argument("--max-examples", type=int, default=10,
                       help="Maximum examples per problem")
    
    args = parser.parse_args()
    
    # Load problems
    extractor = PolyglotBenchmarkExtractor("")
    extractor.load_from_json(args.problems_json)
    
    # Generate training data
    generator = DiffBasedTrainingGenerator(extractor.problems_cache)
    examples = generator.generate_all_training_examples(args.max_examples)
    
    # Export data
    generator.export_training_data(args.output)
    
    # Show statistics
    stats = generator.get_statistics()
    print(f"\nGenerated {stats['total_examples']} training examples:")
    print(f"Languages: {stats['by_language']}")
    print(f"Edit types: {stats['by_edit_type']}")
    print(f"Complexity: {stats['complexity_distribution']}")

if __name__ == "__main__":
    main()