"""
Cross-Language Problem Mapper for Polyglot Benchmark

This module identifies equivalent problems across different languages and creates
language transfer learning examples to help HRM understand language-agnostic
algorithmic patterns.
"""

import re
import json
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from enum import Enum
import logging

from .polyglot_benchmark_extractor import PolyglotProblem, ProblemFile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimilarityMetric(Enum):
    """Types of similarity metrics for cross-language comparison"""
    STRUCTURAL = "structural"      # Code structure similarity
    SEMANTIC = "semantic"          # Algorithmic approach similarity  
    FUNCTIONAL = "functional"      # Input/output behavior similarity
    SYNTACTIC = "syntactic"        # Surface-level syntax similarity

@dataclass
class LanguagePair:
    """Represents a pair of languages for transfer learning"""
    source_lang: str
    target_lang: str
    similarity_score: float
    common_patterns: List[str]

@dataclass
class CrossLanguageMapping:
    """Mapping between equivalent implementations across languages"""
    problem_id: str
    languages: List[str]
    function_mappings: Dict[str, Dict[str, str]]  # lang -> {func_name -> signature}
    concept_mappings: Dict[str, Dict[str, str]]   # lang -> {concept -> implementation}
    difficulty_by_language: Dict[str, float]
    transfer_difficulty: Dict[Tuple[str, str], float]  # (source, target) -> difficulty

@dataclass 
class TransferLearningExample:
    """Training example for cross-language transfer"""
    problem_id: str
    source_language: str
    target_language: str
    source_code: str
    target_code: str
    shared_concepts: List[str]
    language_specific_patterns: Dict[str, List[str]]
    transfer_instructions: str
    difficulty_score: float

class CodeStructureAnalyzer:
    """Analyzes code structure for cross-language comparison"""
    
    def __init__(self):
        self.structure_patterns = {
            "loops": {
                "python": [r'for\s+\w+\s+in\s+', r'while\s+[^:]+:'],
                "javascript": [r'for\s*\([^)]+\)', r'while\s*\([^)]+\)'],
                "java": [r'for\s*\([^)]+\)', r'while\s*\([^)]+\)'],
                "cpp": [r'for\s*\([^)]+\)', r'while\s*\([^)]+\)'],
                "go": [r'for\s+[^{]+{', r'for\s+[^{]*{'],
                "rust": [r'for\s+\w+\s+in\s+', r'while\s+[^{]+{']
            },
            "conditionals": {
                "python": [r'if\s+[^:]+:', r'elif\s+[^:]+:', r'else:'],
                "javascript": [r'if\s*\([^)]+\)', r'else\s+if\s*\([^)]+\)', r'else'],
                "java": [r'if\s*\([^)]+\)', r'else\s+if\s*\([^)]+\)', r'else'],
                "cpp": [r'if\s*\([^)]+\)', r'else\s+if\s*\([^)]+\)', r'else'],
                "go": [r'if\s+[^{]+{', r'else\s+if\s+[^{]+{', r'else\s*{'],
                "rust": [r'if\s+[^{]+{', r'else\s+if\s+[^{]+{', r'else\s*{']
            },
            "functions": {
                "python": [r'def\s+\w+\s*\([^)]*\):'],
                "javascript": [r'function\s+\w+\s*\([^)]*\)', r'const\s+\w+\s*=.*?=>'],
                "java": [r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+\w+\s*\([^)]*\)'],
                "cpp": [r'\w+\s+\w+\s*\([^)]*\)\s*{'],
                "go": [r'func\s+\w+\s*\([^)]*\)'],
                "rust": [r'(?:pub\s+)?fn\s+\w+\s*\([^)]*\)']
            },
            "data_structures": {
                "python": [r'\[.*?\]', r'\{.*?\}', r'dict\s*\(', r'list\s*\('],
                "javascript": [r'\[.*?\]', r'\{.*?\}', r'new\s+Array', r'new\s+Object'],
                "java": [r'new\s+\w+\s*\[', r'new\s+\w+\s*\(', r'ArrayList', r'HashMap'],
                "cpp": [r'std::vector', r'std::map', r'std::set', r'\w+\s*\[.*?\]'],
                "go": [r'make\s*\(', r'\[\].*?{', r'map\s*\['],
                "rust": [r'Vec::', r'HashMap::', r'vec!', r'\[.*?\]']
            }
        }
    
    def extract_structural_features(self, code: str, language: str) -> Dict[str, int]:
        """Extract structural features from code"""
        features = {}
        
        for category, lang_patterns in self.structure_patterns.items():
            if language in lang_patterns:
                patterns = lang_patterns[language]
                count = 0
                for pattern in patterns:
                    matches = re.findall(pattern, code, re.MULTILINE | re.IGNORECASE)
                    count += len(matches)
                features[category] = count
        
        # Additional structural metrics
        features.update({
            "lines": len(code.split('\n')),
            "characters": len(code),
            "words": len(code.split()),
            "indentation_levels": self._count_indentation_levels(code, language)
        })
        
        return features
    
    def _count_indentation_levels(self, code: str, language: str) -> int:
        """Count maximum indentation levels in code"""
        max_level = 0
        
        if language == "python":
            # Python uses indentation for structure
            for line in code.split('\n'):
                if line.strip():
                    level = len(line) - len(line.lstrip())
                    max_level = max(max_level, level // 4)  # Assuming 4-space indentation
        else:
            # Brace-based languages
            level = 0
            for char in code:
                if char == '{':
                    level += 1
                    max_level = max(max_level, level)
                elif char == '}':
                    level -= 1
        
        return max_level
    
    def calculate_structural_similarity(self, features1: Dict[str, int], 
                                      features2: Dict[str, int]) -> float:
        """Calculate structural similarity between two sets of features"""
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0
        
        # Normalize features to 0-1 range
        normalized1 = {}
        normalized2 = {}
        
        for key in common_keys:
            max_val = max(features1[key], features2[key], 1)  # Avoid division by zero
            normalized1[key] = features1[key] / max_val
            normalized2[key] = features2[key] / max_val
        
        # Calculate cosine similarity
        dot_product = sum(normalized1[k] * normalized2[k] for k in common_keys)
        norm1 = sum(v * v for v in normalized1.values()) ** 0.5
        norm2 = sum(v * v for v in normalized2.values()) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

class ConceptExtractor:
    """Extracts algorithmic concepts from code"""
    
    def __init__(self):
        self.concept_patterns = {
            "sorting": [
                r'sort\s*\(', r'sorted\s*\(', r'\.sort\s*\(', r'Arrays\.sort',
                r'std::sort', r'sort\.Slice', r'\.sort_by'
            ],
            "searching": [
                r'binary_search', r'indexOf', r'find\s*\(', r'search',
                r'std::find', r'Contains'
            ],
            "iteration": [
                r'map\s*\(', r'filter\s*\(', r'reduce\s*\(', r'forEach',
                r'iter\(\)', r'\.iter\s*\(', r'range\s*\('
            ],
            "recursion": [
                r'def\s+\w+\s*\([^)]*\):[^}]*\1\s*\(', 
                r'function\s+\w+\s*\([^)]*\)[^}]*\1\s*\(',
                r'fn\s+\w+\s*\([^)]*\)[^}]*\1\s*\('
            ],
            "dynamic_programming": [
                r'memo', r'cache', r'dp\s*\[', r'@lru_cache',
                r'memoize', r'tabulation'
            ],
            "graph_algorithms": [
                r'dfs', r'bfs', r'dijkstra', r'topological',
                r'adjacency', r'visited', r'graph'
            ],
            "string_processing": [
                r'regex', r'replace', r'substring', r'split',
                r'join', r'strip', r'trim'
            ],
            "mathematical": [
                r'math\.', r'Math\.', r'sqrt', r'pow', r'abs',
                r'min\s*\(', r'max\s*\(', r'sum\s*\('
            ]
        }
    
    def extract_concepts(self, code: str) -> List[str]:
        """Extract algorithmic concepts from code"""
        concepts = []
        
        code_lower = code.lower()
        
        for concept, patterns in self.concept_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code_lower, re.IGNORECASE):
                    concepts.append(concept)
                    break  # Avoid duplicates
        
        return concepts

class CrossLanguageProblemMapper:
    """
    Main class for mapping equivalent problems across languages
    """
    
    def __init__(self, problems: Dict[str, PolyglotProblem]):
        """
        Initialize mapper with extracted problems
        
        Args:
            problems: Dictionary of problem_id -> PolyglotProblem
        """
        self.problems = problems
        self.structure_analyzer = CodeStructureAnalyzer()
        self.concept_extractor = ConceptExtractor()
        
        # Computed mappings
        self.cross_language_mappings: Dict[str, CrossLanguageMapping] = {}
        self.language_similarities: Dict[Tuple[str, str], float] = {}
        self.transfer_examples: List[TransferLearningExample] = []
        
        # Language compatibility matrix
        self.language_compatibility = self._compute_language_compatibility()
    
    def _compute_language_compatibility(self) -> Dict[Tuple[str, str], float]:
        """Compute compatibility scores between language pairs"""
        languages = ["python", "javascript", "java", "cpp", "go", "rust"]
        compatibility = {}
        
        # Define base compatibility based on language families
        family_scores = {
            # (lang1, lang2): base_compatibility_score
            ("python", "javascript"): 0.7,  # Dynamic typing
            ("python", "java"): 0.5,        # Different paradigms
            ("python", "cpp"): 0.4,         # Very different
            ("python", "go"): 0.6,          # Clean syntax
            ("python", "rust"): 0.5,        # Different approaches
            
            ("javascript", "java"): 0.6,    # C-like syntax
            ("javascript", "cpp"): 0.7,     # C-like syntax
            ("javascript", "go"): 0.5,      # Different approaches
            ("javascript", "rust"): 0.4,    # Different paradigms
            
            ("java", "cpp"): 0.8,           # Very similar syntax
            ("java", "go"): 0.6,            # Similar concepts
            ("java", "rust"): 0.5,          # Different approaches
            
            ("cpp", "go"): 0.5,             # Different approaches
            ("cpp", "rust"): 0.7,           # Systems programming
            
            ("go", "rust"): 0.6,            # Modern systems languages
        }
        
        # Make matrix symmetric
        for lang1 in languages:
            for lang2 in languages:
                if lang1 == lang2:
                    compatibility[(lang1, lang2)] = 1.0
                elif (lang1, lang2) in family_scores:
                    compatibility[(lang1, lang2)] = family_scores[(lang1, lang2)]
                elif (lang2, lang1) in family_scores:
                    compatibility[(lang1, lang2)] = family_scores[(lang2, lang1)]
                else:
                    compatibility[(lang1, lang2)] = 0.3  # Default low compatibility
        
        return compatibility
    
    def analyze_problem_equivalence(self, problem: PolyglotProblem) -> CrossLanguageMapping:
        """Analyze equivalence of a problem across languages"""
        logger.debug(f"Analyzing problem equivalence: {problem.problem_id}")
        
        mapping = CrossLanguageMapping(
            problem_id=problem.problem_id,
            languages=list(problem.files_by_language.keys()),
            function_mappings={},
            concept_mappings={},
            difficulty_by_language={},
            transfer_difficulty={}
        )
        
        # Analyze each language implementation
        lang_features = {}
        lang_concepts = {}
        
        for language, files in problem.files_by_language.items():
            impl_files = [f for f in files if f.file_type == "implementation"]
            
            if impl_files:
                # Take the main implementation file
                main_file = impl_files[0]
                code = main_file.content
                
                # Extract structural features
                features = self.structure_analyzer.extract_structural_features(code, language)
                lang_features[language] = features
                
                # Extract concepts
                concepts = self.concept_extractor.extract_concepts(code)
                lang_concepts[language] = concepts
                
                # Calculate difficulty
                difficulty = self._calculate_language_difficulty(features, concepts, language)
                mapping.difficulty_by_language[language] = difficulty
                
                # Extract function signatures
                functions = self._extract_function_info(code, language)
                mapping.function_mappings[language] = functions
                
                # Map concepts to implementations
                mapping.concept_mappings[language] = {
                    concept: self._find_concept_implementation(code, concept, language)
                    for concept in concepts
                }
        
        # Calculate transfer difficulties between language pairs
        for source_lang in mapping.languages:
            for target_lang in mapping.languages:
                if source_lang != target_lang:
                    transfer_diff = self._calculate_transfer_difficulty(
                        lang_features.get(source_lang, {}),
                        lang_features.get(target_lang, {}),
                        lang_concepts.get(source_lang, []),
                        lang_concepts.get(target_lang, []),
                        source_lang,
                        target_lang
                    )
                    mapping.transfer_difficulty[(source_lang, target_lang)] = transfer_diff
        
        return mapping
    
    def _calculate_language_difficulty(self, features: Dict[str, int], 
                                     concepts: List[str], language: str) -> float:
        """Calculate implementation difficulty for a specific language"""
        # Base difficulty from code complexity
        complexity_score = (
            features.get("lines", 0) / 50 +
            features.get("functions", 0) * 0.2 +
            features.get("loops", 0) * 0.3 +
            features.get("conditionals", 0) * 0.2 +
            features.get("indentation_levels", 0) * 0.1
        )
        
        # Concept difficulty bonus
        concept_difficulty = {
            "sorting": 0.1, "searching": 0.2, "recursion": 0.4,
            "dynamic_programming": 0.8, "graph_algorithms": 0.7,
            "mathematical": 0.3, "string_processing": 0.2
        }
        
        concept_score = sum(concept_difficulty.get(concept, 0.1) for concept in concepts)
        
        # Language-specific modifiers
        language_modifiers = {
            "python": 0.8,    # Generally easier
            "javascript": 0.9, # Moderately easy
            "java": 1.1,      # More verbose
            "cpp": 1.3,       # Complex memory management
            "go": 1.0,        # Balanced
            "rust": 1.2       # Ownership complexity
        }
        
        modifier = language_modifiers.get(language, 1.0)
        
        return min(1.0, (complexity_score + concept_score) * modifier)
    
    def _calculate_transfer_difficulty(self, source_features: Dict[str, int],
                                     target_features: Dict[str, int],
                                     source_concepts: List[str],
                                     target_concepts: List[str],
                                     source_lang: str, target_lang: str) -> float:
        """Calculate difficulty of transferring from source to target language"""
        
        # Base compatibility
        base_difficulty = 1.0 - self.language_compatibility.get((source_lang, target_lang), 0.3)
        
        # Structural similarity
        if source_features and target_features:
            structural_sim = self.structure_analyzer.calculate_structural_similarity(
                source_features, target_features
            )
            structural_difficulty = 1.0 - structural_sim
        else:
            structural_difficulty = 0.5
        
        # Concept overlap
        common_concepts = set(source_concepts) & set(target_concepts)
        total_concepts = set(source_concepts) | set(target_concepts)
        concept_similarity = len(common_concepts) / max(len(total_concepts), 1)
        concept_difficulty = 1.0 - concept_similarity
        
        # Weighted average
        transfer_difficulty = (
            0.4 * base_difficulty +
            0.3 * structural_difficulty +
            0.3 * concept_difficulty
        )
        
        return min(1.0, transfer_difficulty)
    
    def _extract_function_info(self, code: str, language: str) -> Dict[str, str]:
        """Extract function names and signatures"""
        functions = {}
        
        patterns = {
            "python": r'def\s+(\w+)\s*\(([^)]*)\):',
            "javascript": r'(?:function\s+(\w+)\s*\(([^)]*)\)|const\s+(\w+)\s*=.*?\(([^)]*)\))',
            "java": r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(([^)]*)\)',
            "cpp": r'(?:\w+\s+)?(\w+)\s*\(([^)]*)\)\s*{',
            "go": r'func\s+(\w+)\s*\(([^)]*)\)',
            "rust": r'(?:pub\s+)?fn\s+(\w+)\s*\(([^)]*)\)'
        }
        
        if language in patterns:
            matches = re.finditer(patterns[language], code, re.MULTILINE)
            for match in matches:
                if language == "javascript" and len(match.groups()) == 4:
                    # Handle JavaScript function and const patterns
                    name = match.group(1) or match.group(3)
                    params = match.group(2) or match.group(4)
                else:
                    name = match.group(1)
                    params = match.group(2) if len(match.groups()) > 1 else ""
                
                if name:
                    functions[name] = f"{name}({params})"
        
        return functions
    
    def _find_concept_implementation(self, code: str, concept: str, language: str) -> str:
        """Find how a concept is implemented in the code"""
        # This is a simplified implementation
        # In practice, you'd want more sophisticated analysis
        
        concept_lines = []
        lines = code.split('\n')
        
        # Look for lines that might implement the concept
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in [concept, concept.replace('_', '')]):
                concept_lines.append(line.strip())
        
        return '\n'.join(concept_lines[:3])  # Return first 3 relevant lines
    
    def generate_transfer_examples(self, max_examples_per_pair: int = 5) -> List[TransferLearningExample]:
        """Generate transfer learning examples between language pairs"""
        logger.info("Generating cross-language transfer examples...")
        
        transfer_examples = []
        
        # Analyze all problems first
        for problem_id, problem in self.problems.items():
            if len(problem.files_by_language) >= 2:  # Need at least 2 languages
                mapping = self.analyze_problem_equivalence(problem)
                self.cross_language_mappings[problem_id] = mapping
        
        # Generate transfer examples
        for problem_id, mapping in self.cross_language_mappings.items():
            problem = self.problems[problem_id]
            
            # Consider all language pairs
            languages = mapping.languages
            for i, source_lang in enumerate(languages):
                for j, target_lang in enumerate(languages):
                    if i < j:  # Avoid duplicates and self-transfers
                        examples = self._create_transfer_examples(
                            problem, mapping, source_lang, target_lang
                        )
                        transfer_examples.extend(examples[:max_examples_per_pair])
        
        self.transfer_examples = transfer_examples
        logger.info(f"Generated {len(transfer_examples)} transfer learning examples")
        
        return transfer_examples
    
    def _create_transfer_examples(self, problem: PolyglotProblem, 
                                mapping: CrossLanguageMapping,
                                source_lang: str, target_lang: str) -> List[TransferLearningExample]:
        """Create transfer examples between two specific languages"""
        examples = []
        
        # Get source and target files
        source_files = [f for f in problem.files_by_language.get(source_lang, []) 
                       if f.file_type == "implementation"]
        target_files = [f for f in problem.files_by_language.get(target_lang, []) 
                       if f.file_type == "implementation"]
        
        if not source_files or not target_files:
            return examples
        
        source_code = source_files[0].content
        target_code = target_files[0].content
        
        # Extract shared concepts
        source_concepts = self.concept_extractor.extract_concepts(source_code)
        target_concepts = self.concept_extractor.extract_concepts(target_code)
        shared_concepts = list(set(source_concepts) & set(target_concepts))
        
        # Create language-specific patterns
        language_patterns = {
            source_lang: self._extract_language_patterns(source_code, source_lang),
            target_lang: self._extract_language_patterns(target_code, target_lang)
        }
        
        # Generate transfer instruction
        transfer_instruction = self._generate_transfer_instruction(
            source_lang, target_lang, shared_concepts, mapping
        )
        
        # Calculate difficulty
        transfer_difficulty = mapping.transfer_difficulty.get((source_lang, target_lang), 0.5)
        
        example = TransferLearningExample(
            problem_id=problem.problem_id,
            source_language=source_lang,
            target_language=target_lang,
            source_code=source_code,
            target_code=target_code,
            shared_concepts=shared_concepts,
            language_specific_patterns=language_patterns,
            transfer_instructions=transfer_instruction,
            difficulty_score=transfer_difficulty
        )
        
        examples.append(example)
        
        return examples
    
    def _extract_language_patterns(self, code: str, language: str) -> List[str]:
        """Extract language-specific patterns from code"""
        patterns = []
        
        # Language-specific idioms and patterns
        idiom_patterns = {
            "python": [
                r'if\s+__name__\s*==\s*["\']__main__["\']',
                r'with\s+open\s*\([^)]+\)\s+as\s+\w+',
                r'\[\s*[^]]+\s+for\s+[^]]+\s*\]',  # List comprehensions
                r'lambda\s+[^:]+:'
            ],
            "javascript": [
                r'function\s*\([^)]*\)\s*{\s*return\s+[^}]+}',
                r'\.map\s*\([^)]+\)',
                r'\.filter\s*\([^)]+\)',
                r'=>\s*[^{]'  # Arrow functions
            ],
            "java": [
                r'public\s+static\s+void\s+main',
                r'System\.out\.println',
                r'new\s+\w+\s*\([^)]*\)',
                r'@Override'
            ],
            "cpp": [
                r'#include\s*<[^>]+>',
                r'std::[^(\s]+',
                r'cout\s*<<',
                r'cin\s*>>'
            ],
            "go": [
                r'package\s+\w+',
                r'import\s+[^)]+',
                r'func\s+main\s*\(\)',
                r'make\s*\([^)]+\)'
            ],
            "rust": [
                r'fn\s+main\s*\(\)',
                r'let\s+mut\s+\w+',
                r'println!\s*\(',
                r'match\s+[^{]+{'
            ]
        }
        
        if language in idiom_patterns:
            for pattern in idiom_patterns[language]:
                matches = re.findall(pattern, code, re.MULTILINE)
                patterns.extend(matches)
        
        return patterns
    
    def _generate_transfer_instruction(self, source_lang: str, target_lang: str,
                                     shared_concepts: List[str], 
                                     mapping: CrossLanguageMapping) -> str:
        """Generate instruction for transferring code between languages"""
        
        # Get language-specific characteristics
        lang_characteristics = {
            "python": "dynamic typing, indentation-based structure, built-in data structures",
            "javascript": "dynamic typing, prototype-based objects, event-driven",
            "java": "static typing, object-oriented, verbose syntax",
            "cpp": "static typing, manual memory management, low-level control",
            "go": "static typing, simple syntax, built-in concurrency",
            "rust": "static typing, ownership system, memory safety"
        }
        
        source_chars = lang_characteristics.get(source_lang, "")
        target_chars = lang_characteristics.get(target_lang, "")
        
        instruction_parts = [
            f"Convert this {source_lang} code to {target_lang}.",
            f"The original code uses {source_chars}.",
            f"The target should use {target_chars}."
        ]
        
        if shared_concepts:
            concept_str = ", ".join(shared_concepts)
            instruction_parts.append(f"Key algorithmic concepts to preserve: {concept_str}.")
        
        # Add specific guidance based on transfer difficulty
        transfer_diff = mapping.transfer_difficulty.get((source_lang, target_lang), 0.5)
        
        if transfer_diff > 0.7:
            instruction_parts.append("This is a challenging conversion requiring significant syntax changes.")
        elif transfer_diff > 0.4:
            instruction_parts.append("Pay attention to language-specific idioms and syntax differences.")
        else:
            instruction_parts.append("The languages are similar, focus on syntax adaptation.")
        
        return " ".join(instruction_parts)
    
    def get_language_compatibility_matrix(self) -> Dict:
        """Get the full language compatibility matrix"""
        languages = ["python", "javascript", "java", "cpp", "go", "rust"]
        matrix = {}
        
        for lang1 in languages:
            matrix[lang1] = {}
            for lang2 in languages:
                matrix[lang1][lang2] = self.language_compatibility.get((lang1, lang2), 0.0)
        
        return matrix
    
    def get_cross_language_statistics(self) -> Dict:
        """Get statistics about cross-language mappings"""
        if not self.cross_language_mappings:
            return {}
        
        stats = {
            "total_problems": len(self.cross_language_mappings),
            "language_coverage": defaultdict(int),
            "concept_frequency": defaultdict(int),
            "difficulty_distribution": {"easy": 0, "medium": 0, "hard": 0},
            "transfer_difficulties": {}
        }
        
        # Analyze mappings
        for mapping in self.cross_language_mappings.values():
            # Language coverage
            for lang in mapping.languages:
                stats["language_coverage"][lang] += 1
            
            # Concept frequency
            for lang_concepts in mapping.concept_mappings.values():
                for concept in lang_concepts.keys():
                    stats["concept_frequency"][concept] += 1
            
            # Difficulty distribution
            avg_difficulty = sum(mapping.difficulty_by_language.values()) / len(mapping.difficulty_by_language)
            if avg_difficulty < 0.3:
                stats["difficulty_distribution"]["easy"] += 1
            elif avg_difficulty > 0.7:
                stats["difficulty_distribution"]["hard"] += 1
            else:
                stats["difficulty_distribution"]["medium"] += 1
        
        # Transfer difficulties
        all_transfers = []
        for mapping in self.cross_language_mappings.values():
            all_transfers.extend(mapping.transfer_difficulty.values())
        
        if all_transfers:
            stats["transfer_difficulties"] = {
                "mean": sum(all_transfers) / len(all_transfers),
                "min": min(all_transfers),
                "max": max(all_transfers)
            }
        
        return stats
    
    def export_mappings(self, output_path: str):
        """Export cross-language mappings to JSON"""
        logger.info(f"Exporting cross-language mappings to {output_path}")
        
        export_data = {
            "metadata": {
                "total_mappings": len(self.cross_language_mappings),
                "language_compatibility": {
                    f"{k[0]}-{k[1]}": v for k, v in self.language_compatibility.items()
                },
                "statistics": self.get_cross_language_statistics()
            },
            "mappings": {},
            "transfer_examples": []
        }
        
        # Export mappings
        for problem_id, mapping in self.cross_language_mappings.items():
            export_data["mappings"][problem_id] = {
                "problem_id": mapping.problem_id,
                "languages": mapping.languages,
                "function_mappings": mapping.function_mappings,
                "concept_mappings": mapping.concept_mappings,
                "difficulty_by_language": mapping.difficulty_by_language,
                "transfer_difficulty": {
                    f"{k[0]}-{k[1]}": v for k, v in mapping.transfer_difficulty.items()
                }
            }
        
        # Export transfer examples
        for example in self.transfer_examples:
            export_data["transfer_examples"].append({
                "problem_id": example.problem_id,
                "source_language": example.source_language,
                "target_language": example.target_language,
                "source_code": example.source_code,
                "target_code": example.target_code,
                "shared_concepts": example.shared_concepts,
                "language_specific_patterns": example.language_specific_patterns,
                "transfer_instructions": example.transfer_instructions,
                "difficulty_score": example.difficulty_score
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Export complete: {len(self.cross_language_mappings)} mappings, {len(self.transfer_examples)} examples")

def main():
    """Main function for testing the mapper"""
    import argparse
    from .polyglot_benchmark_extractor import PolyglotBenchmarkExtractor
    
    parser = argparse.ArgumentParser(description="Map equivalent problems across languages")
    parser.add_argument("--problems-json", required=True,
                       help="JSON file with extracted problems")
    parser.add_argument("--output", default="cross_language_mappings.json",
                       help="Output mappings file")
    parser.add_argument("--max-examples", type=int, default=5,
                       help="Maximum transfer examples per language pair")
    
    args = parser.parse_args()
    
    # Load problems
    extractor = PolyglotBenchmarkExtractor("")
    extractor.load_from_json(args.problems_json)
    
    # Create mappings
    mapper = CrossLanguageProblemMapper(extractor.problems_cache)
    
    # Generate transfer examples
    transfer_examples = mapper.generate_transfer_examples(args.max_examples)
    
    # Export mappings
    mapper.export_mappings(args.output)
    
    # Show statistics
    stats = mapper.get_cross_language_statistics()
    print(f"\nGenerated mappings for {stats['total_problems']} problems:")
    print(f"Language coverage: {dict(stats['language_coverage'])}")
    print(f"Most common concepts: {dict(list(stats['concept_frequency'].items())[:5])}")
    print(f"Difficulty distribution: {stats['difficulty_distribution']}")
    
    compatibility = mapper.get_language_compatibility_matrix()
    print(f"\nLanguage compatibility matrix:")
    languages = ["python", "javascript", "java", "cpp", "go", "rust"]
    for lang1 in languages:
        row = [f"{compatibility[lang1][lang2]:.2f}" for lang2 in languages]
        print(f"{lang1:>10}: {' '.join(row)}")

if __name__ == "__main__":
    main()