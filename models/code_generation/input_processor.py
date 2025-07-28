"""
Code Generation Input Processing Module for HRM

This module handles the processing of code generation problems and converts them
into the format expected by the HRM architecture. It supports multi-language
code problems and various code generation scenarios.
"""

from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import ast

class CodeGenerationTask(Enum):
    """Types of code generation tasks supported"""
    GENERATION = "generation"          # Generate code from problem description
    REPAIR = "repair"                 # Fix broken code given error messages
    DIFF_EDIT = "diff_edit"          # Polyglot-style search-replace operations
    TEST_PREDICTION = "test_prediction" # Predict test outputs
    TOOL_USE = "tool_use"            # CLI and development tool usage

class ProgrammingLanguage(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"

@dataclass
class CodeGenerationInput:
    """Structure for code generation input data"""
    problem_description: str
    language: ProgrammingLanguage
    task_type: CodeGenerationTask
    existing_code: Optional[str] = None
    test_cases: Optional[List[str]] = None
    error_message: Optional[str] = None
    context_files: Optional[Dict[str, str]] = None  # filename -> content
    tool_constraints: Optional[List[str]] = None

@dataclass 
class ProcessedCodeInput:
    """Processed input ready for HRM"""
    input_tokens: torch.Tensor
    language_id: torch.Tensor
    task_id: torch.Tensor
    attention_mask: torch.Tensor
    puzzle_identifier: torch.Tensor
    metadata: Dict[str, Union[str, int, float]]

class MultiLanguageTokenizer:
    """
    Hierarchical tokenizer supporting 6 programming languages
    Based on the architecture from Phase 1a analysis
    """
    
    def __init__(self, vocab_size: int = 40000):
        self.vocab_size = vocab_size
        self.shared_vocab_size = 25000  # Common programming concepts
        self.lang_vocab_size = 2500     # Per-language specific tokens
        
        # Language detection patterns
        self.language_patterns = {
            ProgrammingLanguage.PYTHON: [r'def\s+\w+', r'import\s+\w+', r'__\w+__', r':\s*$'],
            ProgrammingLanguage.JAVASCRIPT: [r'function\s+\w+', r'const\s+\w+', r'=>', r'console\.log'],
            ProgrammingLanguage.JAVA: [r'public\s+class', r'public\s+static', r'System\.out'],
            ProgrammingLanguage.CPP: [r'#include\s*<', r'std::', r'cout\s*<<', r'int\s+main'],
            ProgrammingLanguage.GO: [r'func\s+\w+', r'package\s+\w+', r'fmt\.Print', r'go\s+\w+'],
            ProgrammingLanguage.RUST: [r'fn\s+\w+', r'let\s+mut', r'println!', r'impl\s+\w+']
        }
        
        # Initialize sub-tokenizers
        self._init_tokenizers()
        
    def _init_tokenizers(self):
        """Initialize shared and language-specific tokenizers"""
        # For now, use a simple approach with standard tokenizers
        # In production, this would use custom BPE vocabularies
        self.base_tokenizer = AutoTokenizer.from_pretrained("codegen-350M-multi", trust_remote_code=True)
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            
        # Language ID mapping
        self.language_to_id = {lang: i for i, lang in enumerate(ProgrammingLanguage)}
        self.id_to_language = {i: lang for lang, i in self.language_to_id.items()}
    
    def detect_language(self, code_text: str) -> ProgrammingLanguage:
        """Detect programming language from code text"""
        scores = {}
        
        for lang, patterns in self.language_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, code_text, re.MULTILINE))
                score += matches
            scores[lang] = score
        
        # Return language with highest score, default to Python
        if not scores or max(scores.values()) == 0:
            return ProgrammingLanguage.PYTHON
        return max(scores, key=scores.get)
    
    def tokenize(self, text: str, language: Optional[ProgrammingLanguage] = None) -> Dict[str, torch.Tensor]:
        """Tokenize text with language-aware processing"""
        if language is None:
            language = self.detect_language(text)
        
        # Basic tokenization using base tokenizer
        tokens = self.base_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "language_id": torch.tensor([self.language_to_id[language]]),
        }

class SyntaxAnalyzer:
    """Analyze code syntax and structure for enhanced processing"""
    
    def __init__(self):
        self.function_patterns = {
            ProgrammingLanguage.PYTHON: r'def\s+(\w+)\s*\(',
            ProgrammingLanguage.JAVASCRIPT: r'function\s+(\w+)\s*\(',
            ProgrammingLanguage.JAVA: r'public\s+\w+\s+(\w+)\s*\(',
            ProgrammingLanguage.CPP: r'\w+\s+(\w+)\s*\([^)]*\)\s*{',
            ProgrammingLanguage.GO: r'func\s+(\w+)\s*\(',
            ProgrammingLanguage.RUST: r'fn\s+(\w+)\s*\('
        }
    
    def extract_functions(self, code: str, language: ProgrammingLanguage) -> List[str]:
        """Extract function names from code"""
        if language not in self.function_patterns:
            return []
        
        pattern = self.function_patterns[language]
        return re.findall(pattern, code)
    
    def get_complexity_score(self, code: str) -> float:
        """Estimate code complexity for ACT cycle allocation"""
        # Simple heuristic based on code characteristics
        lines = len(code.split('\n'))
        loops = len(re.findall(r'\b(for|while)\b', code))
        conditions = len(re.findall(r'\b(if|switch|case)\b', code))
        functions = len(re.findall(r'\b(def|function|func|fn)\b', code))
        
        # Normalize to 0-1 range
        complexity = (lines / 100 + loops * 0.2 + conditions * 0.15 + functions * 0.1)
        return min(1.0, complexity)

class CodeGenerationInputProcessor(nn.Module):
    """
    Main input processor for code generation tasks
    Integrates with HRM's existing input processing pipeline
    """
    
    def __init__(self, 
                 vocab_size: int = 40000,
                 hidden_size: int = 512,
                 max_seq_len: int = 2048,
                 num_languages: int = 6):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.num_languages = num_languages
        
        # Initialize components
        self.tokenizer = MultiLanguageTokenizer(vocab_size)
        self.syntax_analyzer = SyntaxAnalyzer()
        
        # Task type embeddings
        self.task_embeddings = nn.Embedding(len(CodeGenerationTask), hidden_size)
        
        # Language embeddings
        self.language_embeddings = nn.Embedding(num_languages, hidden_size)
        
        # Complexity embeddings for ACT guidance
        self.complexity_embeddings = nn.Linear(1, hidden_size)
        
        # Special tokens
        self.special_tokens = {
            'problem_start': vocab_size - 10,
            'problem_end': vocab_size - 9,
            'code_start': vocab_size - 8,
            'code_end': vocab_size - 7,
            'test_start': vocab_size - 6,
            'test_end': vocab_size - 5,
            'error_start': vocab_size - 4,
            'error_end': vocab_size - 3,
            'tool_start': vocab_size - 2,
            'tool_end': vocab_size - 1,
        }
    
    def _create_structured_input(self, input_data: CodeGenerationInput) -> str:
        """Create structured input text with special tokens"""
        parts = []
        
        # Problem description
        parts.append(f"<problem>{input_data.problem_description}</problem>")
        
        # Existing code if available
        if input_data.existing_code:
            parts.append(f"<code>{input_data.existing_code}</code>")
        
        # Test cases
        if input_data.test_cases:
            test_text = "\n".join(input_data.test_cases)
            parts.append(f"<tests>{test_text}</tests>")
        
        # Error message for repair tasks
        if input_data.error_message:
            parts.append(f"<error>{input_data.error_message}</error>")
        
        # Tool constraints
        if input_data.tool_constraints:
            tool_text = "\n".join(input_data.tool_constraints)
            parts.append(f"<tools>{tool_text}</tools>")
        
        return "\n".join(parts)
    
    def process_input(self, input_data: CodeGenerationInput) -> ProcessedCodeInput:
        """
        Process code generation input into HRM-compatible format
        """
        # Create structured input text
        structured_text = self._create_structured_input(input_data)
        
        # Tokenize with language awareness
        tokenized = self.tokenizer.tokenize(structured_text, input_data.language)
        
        # Get complexity score for ACT guidance
        code_for_analysis = input_data.existing_code or input_data.problem_description
        complexity_score = self.syntax_analyzer.get_complexity_score(code_for_analysis)
        
        # Create puzzle identifier (combines language and task type)
        language_id = self.tokenizer.language_to_id[input_data.language]
        task_id = list(CodeGenerationTask).index(input_data.task_type)
        puzzle_identifier = language_id * len(CodeGenerationTask) + task_id
        
        # Prepare tensors
        input_tokens = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        
        # Pad or truncate to max_seq_len
        if input_tokens.size(0) > self.max_seq_len:
            input_tokens = input_tokens[:self.max_seq_len]
            attention_mask = attention_mask[:self.max_seq_len]
        else:
            pad_length = self.max_seq_len - input_tokens.size(0)
            input_tokens = torch.cat([input_tokens, torch.zeros(pad_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])
        
        return ProcessedCodeInput(
            input_tokens=input_tokens,
            language_id=torch.tensor([language_id]),
            task_id=torch.tensor([task_id]),
            attention_mask=attention_mask,
            puzzle_identifier=torch.tensor([puzzle_identifier]),
            metadata={
                "complexity_score": complexity_score,
                "language": input_data.language.value,
                "task_type": input_data.task_type.value,
                "has_existing_code": input_data.existing_code is not None,
                "has_tests": input_data.test_cases is not None,
                "num_functions": len(self.syntax_analyzer.extract_functions(
                    code_for_analysis, input_data.language))
            }
        )
    
    def batch_process(self, inputs: List[CodeGenerationInput]) -> Dict[str, torch.Tensor]:
        """Process a batch of inputs for training"""
        processed_inputs = [self.process_input(inp) for inp in inputs]
        
        # Stack into batch tensors
        batch = {
            "inputs": torch.stack([p.input_tokens for p in processed_inputs]),
            "puzzle_identifiers": torch.stack([p.puzzle_identifier for p in processed_inputs]),
            "language_ids": torch.stack([p.language_id for p in processed_inputs]),
            "task_ids": torch.stack([p.task_id for p in processed_inputs]),
            "attention_masks": torch.stack([p.attention_mask for p in processed_inputs]),
            "complexity_scores": torch.tensor([p.metadata["complexity_score"] for p in processed_inputs])
        }
        
        return batch
    
    def forward(self, input_data: Union[CodeGenerationInput, List[CodeGenerationInput]]) -> Dict[str, torch.Tensor]:
        """Forward pass for integration with HRM"""
        if isinstance(input_data, list):
            return self.batch_process(input_data)
        else:
            processed = self.process_input(input_data)
            return {
                "inputs": processed.input_tokens.unsqueeze(0),
                "puzzle_identifiers": processed.puzzle_identifier.unsqueeze(0),
                "language_ids": processed.language_id.unsqueeze(0),
                "task_ids": processed.task_id.unsqueeze(0),
                "attention_masks": processed.attention_mask.unsqueeze(0),
                "complexity_scores": torch.tensor([processed.metadata["complexity_score"]])
            }

def create_sample_inputs() -> List[CodeGenerationInput]:
    """Create sample inputs for testing"""
    return [
        CodeGenerationInput(
            problem_description="Write a function that finds the maximum element in an array",
            language=ProgrammingLanguage.PYTHON,
            task_type=CodeGenerationTask.GENERATION,
            test_cases=["assert max_element([1,2,3]) == 3", "assert max_element([5,1,9]) == 9"]
        ),
        CodeGenerationInput(
            problem_description="Fix the syntax error in this function",
            language=ProgrammingLanguage.JAVASCRIPT,
            task_type=CodeGenerationTask.REPAIR,
            existing_code="function add(a, b { return a + b; }",
            error_message="SyntaxError: missing ) after argument list"
        ),
        CodeGenerationInput(
            problem_description="Convert this Python function to use list comprehension",
            language=ProgrammingLanguage.PYTHON,
            task_type=CodeGenerationTask.DIFF_EDIT,
            existing_code="def squares(nums):\n    result = []\n    for n in nums:\n        result.append(n*n)\n    return result"
        )
    ]

if __name__ == "__main__":
    # Test the input processor
    processor = CodeGenerationInputProcessor()
    sample_inputs = create_sample_inputs()
    
    print("Testing Code Generation Input Processor...")
    
    # Test single input
    single_result = processor(sample_inputs[0])
    print(f"Single input shape: {single_result['inputs'].shape}")
    
    # Test batch processing
    batch_result = processor(sample_inputs)
    print(f"Batch input shape: {batch_result['inputs'].shape}")
    print(f"Languages: {batch_result['language_ids']}")
    print(f"Task types: {batch_result['task_ids']}")
    
    print("Input processor test completed successfully!")