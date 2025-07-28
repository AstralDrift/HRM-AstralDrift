"""
Code Output Generation System for HRM

This module implements sophisticated output generation for different code generation
tasks including direct generation, diff-based editing, test prediction, and tool commands.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from models.code_generation.input_processor import (
    CodeGenerationTask, 
    ProgrammingLanguage,
    CodeGenerationInput
)


class OutputFormat(Enum):
    """Output format types for different tasks"""
    DIRECT_GENERATION = "direct"        # Complete code generation
    DIFF_SEARCH_REPLACE = "diff"       # Search-replace operations  
    TEST_PREDICTION = "test"           # Test output prediction
    TOOL_COMMANDS = "tool"             # CLI tool sequences
    MULTI_FILE = "multi_file"          # Multiple file generation


@dataclass
class CodeOutput:
    """Structure for code generation outputs"""
    content: str
    format_type: OutputFormat
    language: ProgrammingLanguage
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class DiffOperation:
    """Structure for diff-based code editing"""
    search_pattern: str
    replace_content: str
    file_path: Optional[str] = None
    line_numbers: Optional[Tuple[int, int]] = None
    confidence: float = 1.0


@dataclass
class ToolCommand:
    """Structure for tool command generation"""
    command: str
    arguments: List[str]
    working_directory: Optional[str] = None
    environment_vars: Optional[Dict[str, str]] = None
    expected_output: Optional[str] = None
    timeout: Optional[int] = None


class CodePostProcessor:
    """Post-processing utilities for generated code"""
    
    def __init__(self):
        self.language_patterns = {
            ProgrammingLanguage.PYTHON: {
                'function_def': r'def\s+(\w+)\s*\(',
                'class_def': r'class\s+(\w+)\s*[:\(]',
                'import': r'(?:from\s+\w+\s+)?import\s+[\w\.,\s]+',
                'comment': r'#.*$'
            },
            ProgrammingLanguage.JAVASCRIPT: {
                'function_def': r'(?:function\s+(\w+)|(\w+)\s*=\s*(?:function|\([^)]*\)\s*=>))',
                'class_def': r'class\s+(\w+)',
                'import': r'(?:import|require)\s*[^;]+',
                'comment': r'//.*$|/\*[\s\S]*?\*/'
            },
            ProgrammingLanguage.JAVA: {
                'function_def': r'(?:public|private|protected)?\s*(?:static\s+)?[\w<>,\[\]\s]+\s+(\w+)\s*\(',
                'class_def': r'(?:public\s+)?class\s+(\w+)',
                'import': r'import\s+[\w\.]+\s*;',
                'comment': r'//.*$|/\*[\s\S]*?\*/'
            },
            ProgrammingLanguage.CPP: {
                'function_def': r'[\w\s:<>,\*&]+\s+(\w+)\s*\([^)]*\)\s*{',
                'class_def': r'class\s+(\w+)',
                'import': r'#include\s*[<"][^">]+[">]',
                'comment': r'//.*$|/\*[\s\S]*?\*/'
            },
            ProgrammingLanguage.GO: {
                'function_def': r'func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(',
                'class_def': r'type\s+(\w+)\s+struct',
                'import': r'import\s+[^)]+',
                'comment': r'//.*$|/\*[\s\S]*?\*/'
            },
            ProgrammingLanguage.RUST: {
                'function_def': r'fn\s+(\w+)\s*[<\(]',
                'class_def': r'(?:struct|enum|trait)\s+(\w+)',
                'import': r'use\s+[\w::]+\s*;',
                'comment': r'//.*$|/\*[\s\S]*?\*/'
            }
        }
    
    def extract_functions(self, code: str, language: ProgrammingLanguage) -> List[str]:
        """Extract function names from generated code"""
        if language not in self.language_patterns:
            return []
        
        pattern = self.language_patterns[language]['function_def']
        matches = re.findall(pattern, code, re.MULTILINE)
        
        # Handle different regex group structures
        if isinstance(matches[0], tuple) if matches else False:
            return [match[0] or match[1] for match in matches if match[0] or match[1]]
        return matches
    
    def validate_syntax(self, code: str, language: ProgrammingLanguage) -> Tuple[bool, Optional[str]]:
        """Basic syntax validation for generated code"""
        try:
            if language == ProgrammingLanguage.PYTHON:
                import ast
                ast.parse(code)
                return True, None
            
            # For other languages, do basic bracket matching
            brackets = {'(': ')', '[': ']', '{': '}'}
            stack = []
            
            for char in code:
                if char in brackets:
                    stack.append(brackets[char])
                elif char in brackets.values():
                    if not stack or stack.pop() != char:
                        return False, f"Mismatched bracket: {char}"
            
            if stack:
                return False, f"Unclosed brackets: {stack}"
            
            return True, None
            
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def format_code(self, code: str, language: ProgrammingLanguage) -> str:
        """Apply basic formatting to generated code"""
        lines = code.split('\n')
        formatted_lines = []
        indent_level = 0
        
        # Language-specific indentation
        indent_chars = {
            ProgrammingLanguage.PYTHON: '    ',  # 4 spaces
            ProgrammingLanguage.JAVASCRIPT: '  ', # 2 spaces
            ProgrammingLanguage.JAVA: '    ',     # 4 spaces
            ProgrammingLanguage.CPP: '    ',      # 4 spaces
            ProgrammingLanguage.GO: '\t',         # tab
            ProgrammingLanguage.RUST: '    '      # 4 spaces
        }
        
        indent_str = indent_chars.get(language, '    ')
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                formatted_lines.append('')
                continue
            
            # Decrease indent for closing brackets
            if stripped.startswith(('}', ')', ']')):
                indent_level = max(0, indent_level - 1)
            
            # Add indented line
            formatted_lines.append(indent_str * indent_level + stripped)
            
            # Increase indent for opening brackets and certain keywords
            if stripped.endswith(('{', '(', '[')):
                indent_level += 1
            elif language == ProgrammingLanguage.PYTHON and stripped.endswith(':'):
                indent_level += 1
        
        return '\n'.join(formatted_lines)


class DiffGenerator:
    """Generate diff-based search-replace operations for code editing"""
    
    def __init__(self):
        self.post_processor = CodePostProcessor()
    
    def generate_diff_operations(self, 
                                original_code: str,
                                target_description: str,
                                language: ProgrammingLanguage) -> List[DiffOperation]:
        """Generate search-replace operations based on target description"""
        
        operations = []
        
        # Simple pattern matching for common operations
        if "add function" in target_description.lower():
            # Find appropriate insertion point
            lines = original_code.split('\n')
            insert_point = len(lines)
            
            # Find last function or class for insertion
            for i, line in enumerate(reversed(lines)):
                if re.match(r'^\s*(def|function|class)', line):
                    insert_point = len(lines) - i
                    break
            
            operations.append(DiffOperation(
                search_pattern=lines[insert_point-1] if insert_point > 0 else "",
                replace_content=lines[insert_point-1] + "\n\n# New function placeholder",
                line_numbers=(insert_point, insert_point)
            ))
        
        elif "fix syntax" in target_description.lower():
            # Common syntax fixes
            if language == ProgrammingLanguage.PYTHON:
                # Fix common Python syntax issues
                fixes = [
                    (r'(\w+)\s*\(\s*([^)]+)\s*\{', r'\1(\2):'),  # function definition
                    (r'if\s*\(\s*([^)]+)\s*\)\s*\{', r'if \1:'),  # if statement
                ]
                
                for pattern, replacement in fixes:
                    matches = list(re.finditer(pattern, original_code))
                    for match in matches:
                        operations.append(DiffOperation(
                            search_pattern=match.group(0),
                            replace_content=re.sub(pattern, replacement, match.group(0))
                        ))
        
        return operations
    
    def apply_diff_operations(self, 
                             original_code: str,
                             operations: List[DiffOperation]) -> str:
        """Apply diff operations to original code"""
        modified_code = original_code
        
        # Sort operations by position (reverse order to maintain positions)
        operations = sorted(operations, key=lambda op: op.line_numbers[0] if op.line_numbers else 0, reverse=True)
        
        for operation in operations:
            if operation.search_pattern in modified_code:
                modified_code = modified_code.replace(
                    operation.search_pattern,
                    operation.replace_content,
                    1  # Replace only first occurrence
                )
        
        return modified_code


class ToolCommandGenerator:
    """Generate CLI tool commands and development workflows"""
    
    def __init__(self):
        self.command_templates = {
            # Git operations
            'git_init': 'git init',
            'git_add': 'git add {files}',
            'git_commit': 'git commit -m "{message}"',
            'git_push': 'git push {remote} {branch}',
            'git_pull': 'git pull {remote} {branch}',
            'git_status': 'git status',
            'git_log': 'git log --oneline -n {count}',
            
            # Package managers
            'npm_install': 'npm install {packages}',
            'npm_run': 'npm run {script}',
            'pip_install': 'pip install {packages}',
            'pip_requirements': 'pip install -r requirements.txt',
            'cargo_build': 'cargo build {flags}',
            'cargo_run': 'cargo run {args}',
            
            # Build systems
            'make_build': 'make {target}',
            'cmake_build': 'cmake --build {build_dir}',
            'gradle_build': './gradlew {task}',
            
            # File operations
            'create_file': 'touch {filename}',
            'create_dir': 'mkdir -p {dirname}',
            'copy_file': 'cp {source} {dest}',
            'move_file': 'mv {source} {dest}',
            'remove_file': 'rm {filename}',
            
            # Code execution
            'python_run': 'python {script} {args}',
            'node_run': 'node {script} {args}',
            'java_compile': 'javac {files}',
            'java_run': 'java {main_class} {args}',
            'gcc_compile': 'gcc {files} -o {output}',
            'rustc_compile': 'rustc {file} -o {output}',
        }
    
    def generate_workflow(self, 
                         task_description: str,
                         language: Optional[ProgrammingLanguage] = None,
                         project_context: Optional[Dict[str, Any]] = None) -> List[ToolCommand]:
        """Generate a sequence of tool commands for a development task"""
        
        commands = []
        
        # Parse task description for command generation
        if "create new project" in task_description.lower():
            commands.extend(self._generate_project_setup(language, project_context))
        
        elif "run tests" in task_description.lower():
            commands.extend(self._generate_test_commands(language, project_context))
        
        elif "build project" in task_description.lower():
            commands.extend(self._generate_build_commands(language, project_context))
        
        elif "deploy" in task_description.lower():
            commands.extend(self._generate_deployment_commands(language, project_context))
        
        elif "commit changes" in task_description.lower():
            commands.extend(self._generate_git_workflow())
        
        return commands
    
    def _generate_project_setup(self, 
                               language: Optional[ProgrammingLanguage],
                               context: Optional[Dict[str, Any]]) -> List[ToolCommand]:
        """Generate commands for setting up a new project"""
        commands = []
        
        project_name = context.get('project_name', 'my_project') if context else 'my_project'
        
        # Create project directory
        commands.append(ToolCommand(
            command='mkdir',
            arguments=['-p', project_name]
        ))
        
        # Initialize version control
        commands.append(ToolCommand(
            command='git',
            arguments=['init'],
            working_directory=project_name
        ))
        
        # Language-specific setup
        if language == ProgrammingLanguage.PYTHON:
            commands.extend([
                ToolCommand(command='python', arguments=['-m', 'venv', 'venv'], working_directory=project_name),
                ToolCommand(command='touch', arguments=['requirements.txt'], working_directory=project_name),
                ToolCommand(command='touch', arguments=['main.py'], working_directory=project_name)
            ])
        
        elif language == ProgrammingLanguage.JAVASCRIPT:
            commands.extend([
                ToolCommand(command='npm', arguments=['init', '-y'], working_directory=project_name),
                ToolCommand(command='touch', arguments=['index.js'], working_directory=project_name)
            ])
        
        elif language == ProgrammingLanguage.RUST:
            commands.append(ToolCommand(
                command='cargo',
                arguments=['init', project_name],
                working_directory='.'
            ))
        
        return commands
    
    def _generate_test_commands(self, 
                               language: Optional[ProgrammingLanguage],
                               context: Optional[Dict[str, Any]]) -> List[ToolCommand]:
        """Generate commands for running tests"""
        commands = []
        
        if language == ProgrammingLanguage.PYTHON:
            commands.append(ToolCommand(command='python', arguments=['-m', 'pytest']))
        
        elif language == ProgrammingLanguage.JAVASCRIPT:
            commands.append(ToolCommand(command='npm', arguments=['test']))
        
        elif language == ProgrammingLanguage.JAVA:
            commands.append(ToolCommand(command='mvn', arguments=['test']))
        
        elif language == ProgrammingLanguage.RUST:
            commands.append(ToolCommand(command='cargo', arguments=['test']))
        
        return commands
    
    def _generate_build_commands(self, 
                                language: Optional[ProgrammingLanguage],
                                context: Optional[Dict[str, Any]]) -> List[ToolCommand]:
        """Generate commands for building the project"""
        commands = []
        
        if language == ProgrammingLanguage.JAVASCRIPT:
            commands.append(ToolCommand(command='npm', arguments=['run', 'build']))
        
        elif language == ProgrammingLanguage.JAVA:
            commands.append(ToolCommand(command='mvn', arguments=['compile']))
        
        elif language == ProgrammingLanguage.RUST:
            commands.append(ToolCommand(command='cargo', arguments=['build', '--release']))
        
        elif language == ProgrammingLanguage.GO:
            commands.append(ToolCommand(command='go', arguments=['build']))
        
        return commands
    
    def _generate_deployment_commands(self, 
                                     language: Optional[ProgrammingLanguage],
                                     context: Optional[Dict[str, Any]]) -> List[ToolCommand]:
        """Generate commands for deployment"""
        commands = []
        
        # Build first
        commands.extend(self._generate_build_commands(language, context))
        
        # Run tests
        commands.extend(self._generate_test_commands(language, context))
        
        # Tag and push
        commands.extend([
            ToolCommand(command='git', arguments=['tag', '-a', 'v1.0.0', '-m', 'Release v1.0.0']),
            ToolCommand(command='git', arguments=['push', 'origin', 'main']),
            ToolCommand(command='git', arguments=['push', 'origin', '--tags'])
        ])
        
        return commands
    
    def _generate_git_workflow(self) -> List[ToolCommand]:
        """Generate standard git workflow commands"""
        return [
            ToolCommand(command='git', arguments=['add', '.']),
            ToolCommand(command='git', arguments=['status']),
            ToolCommand(command='git', arguments=['commit', '-m', 'Update code']),
            ToolCommand(command='git', arguments=['push', 'origin', 'main'])
        ]


class CodeOutputGenerator:
    """Main output generator that coordinates different output types"""
    
    def __init__(self, tokenizer_name: str = "codegen-350M-multi"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.post_processor = CodePostProcessor()
        self.diff_generator = DiffGenerator()
        self.tool_generator = ToolCommandGenerator()
    
    def decode_logits(self, 
                     logits: torch.Tensor,
                     task_type: CodeGenerationTask,
                     language: ProgrammingLanguage,
                     temperature: float = 0.8,
                     top_p: float = 0.9,
                     max_length: int = 512) -> str:
        """Decode logits to text using appropriate sampling strategy"""
        
        # Apply temperature
        logits = logits / temperature
        
        # Top-p sampling
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter back to original indices
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Decode token
        text = self.tokenizer.decode(next_token.squeeze(), skip_special_tokens=True)
        return text
    
    def generate_output(self, 
                       model_outputs: Dict[str, torch.Tensor],
                       input_data: CodeGenerationInput,
                       generation_config: Optional[Dict[str, Any]] = None) -> CodeOutput:
        """Generate final output based on model predictions and task type"""
        
        config = generation_config or {}
        task_type = input_data.task_type
        language = input_data.language
        
        if task_type == CodeGenerationTask.GENERATION:
            return self._generate_direct_code(model_outputs, input_data, config)
        
        elif task_type == CodeGenerationTask.REPAIR:
            return self._generate_code_repair(model_outputs, input_data, config)
        
        elif task_type == CodeGenerationTask.DIFF_EDIT:
            return self._generate_diff_edit(model_outputs, input_data, config)
        
        elif task_type == CodeGenerationTask.TEST_PREDICTION:
            return self._generate_test_prediction(model_outputs, input_data, config)
        
        elif task_type == CodeGenerationTask.TOOL_USE:
            return self._generate_tool_commands(model_outputs, input_data, config)
        
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def _generate_direct_code(self, 
                             outputs: Dict[str, torch.Tensor],
                             input_data: CodeGenerationInput,
                             config: Dict[str, Any]) -> CodeOutput:
        """Generate complete code from scratch"""
        
        logits = outputs.get('generation', outputs.get('shared'))
        generated_text = self.decode_logits(
            logits[0],  # First sequence in batch
            input_data.task_type,
            input_data.language,
            temperature=config.get('temperature', 0.8)
        )
        
        # Post-process generated code
        formatted_code = self.post_processor.format_code(generated_text, input_data.language)
        is_valid, error_msg = self.post_processor.validate_syntax(formatted_code, input_data.language)
        
        # Extract functions for metadata
        functions = self.post_processor.extract_functions(formatted_code, input_data.language)
        
        return CodeOutput(
            content=formatted_code,
            format_type=OutputFormat.DIRECT_GENERATION,
            language=input_data.language,
            confidence=0.8,  # Could be computed from logits
            metadata={
                'valid_syntax': is_valid,
                'syntax_error': error_msg,
                'extracted_functions': functions,
                'line_count': len(formatted_code.split('\n'))
            }
        )
    
    def _generate_code_repair(self, 
                             outputs: Dict[str, torch.Tensor],
                             input_data: CodeGenerationInput,
                             config: Dict[str, Any]) -> CodeOutput:
        """Generate repaired code based on error message"""
        
        # Use diff generation for repairs
        if 'diff_replace' in outputs:
            logits = outputs['diff_replace']
            replacement_text = self.decode_logits(logits[0], input_data.task_type, input_data.language)
            
            # Apply simple repair heuristics
            repaired_code = input_data.existing_code or ""
            if input_data.error_message and "SyntaxError" in input_data.error_message:
                # Try to fix common syntax errors
                repaired_code = self._apply_syntax_fixes(repaired_code, input_data.language)
            
            return CodeOutput(
                content=repaired_code,
                format_type=OutputFormat.DIRECT_GENERATION,
                language=input_data.language,
                confidence=0.7,
                metadata={
                    'repair_type': 'syntax_fix',
                    'original_error': input_data.error_message
                }
            )
        
        # Fallback to direct generation
        return self._generate_direct_code(outputs, input_data, config)
    
    def _generate_diff_edit(self, 
                           outputs: Dict[str, torch.Tensor],
                           input_data: CodeGenerationInput,
                           config: Dict[str, Any]) -> CodeOutput:
        """Generate diff-based search-replace operations"""
        
        operations = []
        
        if 'diff_search' in outputs and 'diff_replace' in outputs:
            search_logits = outputs['diff_search'][0]
            replace_logits = outputs['diff_replace'][0]
            
            search_text = self.decode_logits(search_logits, input_data.task_type, input_data.language)
            replace_text = self.decode_logits(replace_logits, input_data.task_type, input_data.language)
            
            operations.append(DiffOperation(
                search_pattern=search_text,
                replace_content=replace_text,
                confidence=0.8
            ))
        
        # Apply operations to original code
        modified_code = input_data.existing_code or ""
        if operations:
            modified_code = self.diff_generator.apply_diff_operations(modified_code, operations)
        
        return CodeOutput(
            content=modified_code,
            format_type=OutputFormat.DIFF_SEARCH_REPLACE,
            language=input_data.language,
            confidence=0.8,
            metadata={
                'diff_operations': [
                    {
                        'search': op.search_pattern,
                        'replace': op.replace_content,
                        'confidence': op.confidence
                    } for op in operations
                ],
                'num_changes': len(operations)
            }
        )
    
    def _generate_test_prediction(self, 
                                 outputs: Dict[str, torch.Tensor],
                                 input_data: CodeGenerationInput,
                                 config: Dict[str, Any]) -> CodeOutput:
        """Generate test output predictions"""
        
        logits = outputs.get('generation', outputs.get('shared'))
        predicted_output = self.decode_logits(logits[0], input_data.task_type, input_data.language)
        
        return CodeOutput(
            content=predicted_output,
            format_type=OutputFormat.TEST_PREDICTION,
            language=input_data.language,
            confidence=0.75,
            metadata={
                'prediction_type': 'test_output',
                'test_cases': input_data.test_cases or []
            }
        )
    
    def _generate_tool_commands(self, 
                               outputs: Dict[str, torch.Tensor],
                               input_data: CodeGenerationInput,
                               config: Dict[str, Any]) -> CodeOutput:
        """Generate CLI tool command sequences"""
        
        if 'tool_commands' in outputs:
            logits = outputs['tool_commands'][0]
            command_text = self.decode_logits(logits, input_data.task_type, input_data.language)
        else:
            # Fallback to workflow generation
            commands = self.tool_generator.generate_workflow(
                input_data.problem_description,
                input_data.language
            )
            command_text = '\n'.join([f"{cmd.command} {' '.join(cmd.arguments)}" for cmd in commands])
        
        return CodeOutput(
            content=command_text,
            format_type=OutputFormat.TOOL_COMMANDS,
            language=input_data.language,
            confidence=0.8,
            metadata={
                'command_count': len(command_text.split('\n')),
                'tool_constraints': input_data.tool_constraints or []
            }
        )
    
    def _apply_syntax_fixes(self, code: str, language: ProgrammingLanguage) -> str:
        """Apply common syntax fixes"""
        if language == ProgrammingLanguage.PYTHON:
            # Fix missing colons
            code = re.sub(r'(if\s+.*)\s*{', r'\1:', code)
            code = re.sub(r'(def\s+\w+\([^)]*\))\s*{', r'\1:', code)
            code = re.sub(r'(for\s+.*)\s*{', r'\1:', code)
            code = re.sub(r'(while\s+.*)\s*{', r'\1:', code)
            
            # Fix braces to indentation
            code = re.sub(r'{\s*', ':\n    ', code)
            code = re.sub(r'\s*}', '', code)
        
        return code


if __name__ == "__main__":
    # Test the output generation system
    generator = CodeOutputGenerator()
    
    # Create test input
    test_input = CodeGenerationInput(
        problem_description="Write a function to calculate factorial",
        language=ProgrammingLanguage.PYTHON,
        task_type=CodeGenerationTask.GENERATION
    )
    
    # Mock model outputs
    test_outputs = {
        'generation': torch.randn(1, 50, 40000),  # batch_size=1, seq_len=50, vocab_size=40000
        'shared': torch.randn(1, 50, 25000)
    }
    
    print("Testing Code Output Generator...")
    
    # Test direct generation
    output = generator.generate_output(test_outputs, test_input)
    print(f"Generated output type: {output.format_type}")
    print(f"Language: {output.language}")
    print(f"Confidence: {output.confidence}")
    print(f"Metadata: {output.metadata}")
    
    # Test diff generation
    diff_input = CodeGenerationInput(
        problem_description="Fix the syntax error",
        language=ProgrammingLanguage.PYTHON,
        task_type=CodeGenerationTask.DIFF_EDIT,
        existing_code="def factorial(n { return 1 if n <= 1 else n * factorial(n-1) }"
    )
    
    diff_outputs = {
        'diff_search': torch.randn(1, 20, 25000),
        'diff_replace': torch.randn(1, 20, 25000)
    }
    
    diff_output = generator.generate_output(diff_outputs, diff_input)
    print(f"\nDiff output type: {diff_output.format_type}")
    print(f"Number of changes: {diff_output.metadata.get('num_changes', 0)}")
    
    # Test tool command generation
    tool_input = CodeGenerationInput(
        problem_description="Create a new Python project and run tests",
        language=ProgrammingLanguage.PYTHON,
        task_type=CodeGenerationTask.TOOL_USE
    )
    
    tool_output = generator.generate_output(test_outputs, tool_input)
    print(f"\nTool command output type: {tool_output.format_type}")
    print(f"Command count: {tool_output.metadata.get('command_count', 0)}")
    
    print("\nCode output generation test completed successfully!")