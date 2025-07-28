"""
HRM Code Generation Module

This module contains the complete implementation of the enhanced HRM architecture
for code generation tasks, including multi-language support, GSPO integration,
and sophisticated output generation capabilities.

Components:
- input_processor: Multi-language input processing and tokenization
- embeddings: Code-specific embeddings with language adapters
- hrm_code_model: Enhanced HRM architecture for code generation
- output_generator: Multi-format output generation (direct, diff, tool commands)
- losses: GSPO-enhanced loss functions with code-specific objectives

Key Features:
- 6 programming languages (Python, JavaScript, Java, C++, Go, Rust)
- 5 task types (generation, repair, diff editing, test prediction, tool use)
- GSPO sequence-level optimization
- Complexity-aware adaptive computation
- Syntax-aware loss functions
- Constitutional AI safety integration ready
"""

from .input_processor import (
    CodeGenerationInput,
    CodeGenerationTask,
    ProgrammingLanguage,
    CodeGenerationInputProcessor,
    MultiLanguageTokenizer,
    SyntaxAnalyzer
)

from .embeddings import (
    MultiLanguageCodeEmbedding,
    CodeEmbeddingConfig,
    CodeSpecificSparseEmbedding,
    LanguageAdapterEmbedding,
    SyntaxAwarePositionalEmbedding,
    LanguageType,
    create_cross_language_alignment_loss
)

from .hrm_code_model import (
    CodeGenHRM,
    CodeGenHRMConfig,
    CodeGenHRM_Inner,
    CodeGenHierarchicalBlock,
    CodeGenReasoningModule,
    ComplexityAwareACTController,
    MultiLanguageOutputHead
)

from .output_generator import (
    CodeOutputGenerator,
    CodeOutput,
    OutputFormat,
    DiffOperation,
    ToolCommand,
    CodePostProcessor,
    DiffGenerator,
    ToolCommandGenerator
)

from .losses import (
    CodeGenACTLossHead,
    SyntaxAwareLoss,
    CompilationLoss,
    TestPassLoss,
    GSPOSequenceLoss,
    MultiTaskCodeLoss,
    CodeGenerationMetrics,
    create_code_generation_loss
)

__all__ = [
    # Core classes
    'CodeGenHRM',
    'CodeGenHRMConfig',
    'CodeGenerationInputProcessor',
    'CodeOutputGenerator',
    'CodeGenACTLossHead',
    
    # Input/Output
    'CodeGenerationInput',
    'CodeOutput',
    'DiffOperation',
    'ToolCommand',
    
    # Enums
    'CodeGenerationTask',
    'ProgrammingLanguage',
    'LanguageType',
    'OutputFormat',
    
    # Configurations
    'CodeEmbeddingConfig',
    
    # Loss functions
    'create_code_generation_loss',
    'GSPOSequenceLoss',
    
    # Utilities
    'create_cross_language_alignment_loss'
]

# Version info
__version__ = "1.0.0"
__author__ = "HRM Code Generation Team"
__description__ = "Enhanced Hierarchical Reasoning Model for Code Generation"