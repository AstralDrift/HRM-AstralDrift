"""
Polyglot Benchmark Data Extraction and Processing Module

This module handles the extraction and processing of 225 Exercism problems
across 6 programming languages (C++, Go, Java, JavaScript, Python, Rust).
It creates the foundation for multi-language code generation training data.
"""

import os
import json
import glob
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import re
import subprocess
import logging
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageConfig(Enum):
    """Configuration for each supported programming language"""
    PYTHON = {
        "extension": ".py",
        "test_pattern": "*_test.py",
        "implementation_pattern": "*.py",
        "build_command": None,
        "test_command": "python -m pytest {test_file}",
        "exclude_patterns": ["__pycache__", ".pytest_cache"]
    }
    JAVASCRIPT = {
        "extension": ".js", 
        "test_pattern": "*.spec.js",
        "implementation_pattern": "*.js",
        "build_command": "npm install",
        "test_command": "npm test",
        "exclude_patterns": ["node_modules", "babel.config.js"]
    }
    JAVA = {
        "extension": ".java",
        "test_pattern": "*Test.java",
        "implementation_pattern": "*.java", 
        "build_command": "./gradlew build",
        "test_command": "./gradlew test",
        "exclude_patterns": ["build", ".gradle", "gradlew*"]
    }
    CPP = {
        "extension": ".cpp",
        "test_pattern": "*_test.cpp",
        "implementation_pattern": "*.cpp",
        "build_command": "cmake . && make",
        "test_command": "./test_runner",
        "exclude_patterns": ["CMakeFiles", "CMakeCache.txt", "Makefile"]
    }
    GO = {
        "extension": ".go",
        "test_pattern": "*_test.go",
        "implementation_pattern": "*.go",
        "build_command": "go mod tidy",
        "test_command": "go test",
        "exclude_patterns": ["go.sum"]
    }
    RUST = {
        "extension": ".rs",
        "test_pattern": "**/tests/*.rs",
        "implementation_pattern": "src/*.rs",
        "build_command": "cargo build",
        "test_command": "cargo test",
        "exclude_patterns": ["target", "Cargo.lock"]
    }

@dataclass
class ProblemFile:
    """Represents a single file in a coding problem"""
    filename: str
    content: str
    file_type: str  # 'implementation', 'test', 'build', 'other'
    language: str

@dataclass
class PolyglotProblem:
    """Represents a coding problem across multiple languages"""
    problem_id: str  # e.g., "beer-song"
    problem_name: str
    description: str
    difficulty: Optional[str] = None
    files_by_language: Dict[str, List[ProblemFile]] = None
    cross_language_mapping: Dict[str, str] = None  # language -> main function name
    complexity_score: float = 0.0
    
    def __post_init__(self):
        if self.files_by_language is None:
            self.files_by_language = {}
        if self.cross_language_mapping is None:
            self.cross_language_mapping = {}

class PolyglotBenchmarkExtractor:
    """
    Main class for extracting and processing Polyglot benchmark data
    """
    
    def __init__(self, benchmark_root: str):
        """
        Initialize the extractor
        
        Args:
            benchmark_root: Path to polyglot-benchmark directory
        """
        self.benchmark_root = Path(benchmark_root)
        self.languages = {
            "python": LanguageConfig.PYTHON.value,
            "javascript": LanguageConfig.JAVASCRIPT.value,
            "java": LanguageConfig.JAVA.value,
            "cpp": LanguageConfig.CPP.value,
            "go": LanguageConfig.GO.value,
            "rust": LanguageConfig.RUST.value
        }
        
        # Cache for extracted data
        self.problems_cache: Dict[str, PolyglotProblem] = {}
        self.language_stats: Dict[str, Dict] = {}
        
        # Validate benchmark structure
        self._validate_benchmark_structure()
    
    def _validate_benchmark_structure(self):
        """Validate that the benchmark directory has expected structure"""
        for lang in self.languages.keys():
            lang_path = self.benchmark_root / lang / "exercises" / "practice"
            if not lang_path.exists():
                logger.warning(f"Missing language directory: {lang_path}")
        
        logger.info(f"Initialized PolyglotBenchmarkExtractor with root: {self.benchmark_root}")
    
    def _get_problem_directories(self, language: str) -> List[Path]:
        """Get all problem directories for a specific language"""
        practice_dir = self.benchmark_root / language / "exercises" / "practice"
        if not practice_dir.exists():
            return []
        
        # Get all subdirectories (each is a problem)
        problem_dirs = [d for d in practice_dir.iterdir() if d.is_dir()]
        return sorted(problem_dirs)
    
    def _classify_file_type(self, filename: str, language: str) -> str:
        """Classify a file as implementation, test, build, or other"""
        config = self.languages[language]
        
        # Test files
        if (filename.endswith("_test" + config["extension"]) or 
            filename.endswith(".spec.js") or
            filename.endswith("Test.java") or
            "test" in filename.lower()):
            return "test"
        
        # Build files
        build_files = {
            "CMakeLists.txt", "Makefile", "build.gradle", "gradlew", "gradlew.bat",
            "package.json", "babel.config.js", "go.mod", "Cargo.toml"
        }
        if filename in build_files:
            return "build"
        
        # Implementation files (source code)
        if filename.endswith(config["extension"]) and "test" not in filename.lower():
            return "implementation"
        
        # Headers and other auxiliary files
        if filename.endswith((".h", ".hpp")):
            return "implementation"
        
        return "other"
    
    def _extract_problem_metadata(self, problem_dir: Path, language: str) -> Dict:
        """Extract metadata about a problem from its directory"""
        metadata = {
            "problem_id": problem_dir.name,
            "language": language,
            "file_count": 0,
            "has_tests": False,
            "has_implementation": False,
            "complexity_indicators": {}
        }
        
        # Count files and analyze structure
        for file_path in problem_dir.rglob("*"):
            if file_path.is_file():
                file_type = self._classify_file_type(file_path.name, language)
                metadata["file_count"] += 1
                
                if file_type == "test":
                    metadata["has_tests"] = True
                elif file_type == "implementation":
                    metadata["has_implementation"] = True
        
        return metadata
    
    def _read_file_safely(self, file_path: Path) -> Optional[str]:
        """Safely read a file with proper encoding handling"""
        try:
            # Try UTF-8 first
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                # Fallback to latin-1
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                logger.warning(f"Could not read file {file_path}: {e}")
                return None
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            return None
    
    def _extract_function_signatures(self, content: str, language: str) -> List[str]:
        """Extract function signatures from code content"""
        signatures = []
        
        patterns = {
            "python": r'def\s+(\w+)\s*\([^)]*\):',
            "javascript": r'(?:function\s+(\w+)|const\s+(\w+)\s*=|(\w+)\s*[:=]\s*(?:function|\([^)]*\)\s*=>))',
            "java": r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\([^)]*\)',
            "cpp": r'(?:\w+\s+)?(\w+)\s*\([^)]*\)\s*(?:const)?\s*{',
            "go": r'func\s+(\w+)\s*\([^)]*\)',
            "rust": r'(?:pub\s+)?fn\s+(\w+)\s*\([^)]*\)'
        }
        
        if language in patterns:
            matches = re.findall(patterns[language], content, re.MULTILINE)
            # Flatten tuples from JavaScript regex groups
            for match in matches:
                if isinstance(match, tuple):
                    signatures.extend([m for m in match if m])
                else:
                    signatures.append(match)
        
        return signatures
    
    def extract_single_problem(self, problem_dir: Path, language: str) -> Optional[PolyglotProblem]:
        """Extract a single problem from a directory"""
        try:
            problem_id = problem_dir.name
            logger.debug(f"Extracting problem {problem_id} for {language}")
            
            # Create problem object
            problem = PolyglotProblem(
                problem_id=problem_id,
                problem_name=problem_id.replace("-", " ").title(),
                description=f"Exercism problem: {problem_id}",
                files_by_language={language: []}
            )
            
            config = self.languages[language]
            exclude_patterns = config["exclude_patterns"]
            
            # Extract all relevant files
            for file_path in problem_dir.rglob("*"):
                if file_path.is_file():
                    # Skip excluded files
                    if any(pattern in str(file_path) for pattern in exclude_patterns):
                        continue
                    
                    content = self._read_file_safely(file_path)
                    if content is None:
                        continue
                    
                    file_type = self._classify_file_type(file_path.name, language)
                    relative_path = file_path.relative_to(problem_dir)
                    
                    problem_file = ProblemFile(
                        filename=str(relative_path),
                        content=content,
                        file_type=file_type,
                        language=language
                    )
                    
                    problem.files_by_language[language].append(problem_file)
                    
                    # Extract function signatures for cross-language mapping
                    if file_type == "implementation" and content.strip():
                        signatures = self._extract_function_signatures(content, language)
                        if signatures:
                            problem.cross_language_mapping[language] = signatures[0]
            
            # Calculate complexity score
            problem.complexity_score = self._calculate_complexity_score(problem, language)
            
            return problem
            
        except Exception as e:
            logger.error(f"Error extracting problem {problem_dir}: {e}")
            return None
    
    def _calculate_complexity_score(self, problem: PolyglotProblem, language: str) -> float:
        """Calculate complexity score based on code characteristics"""
        total_lines = 0
        function_count = 0
        test_count = 0
        
        for file in problem.files_by_language.get(language, []):
            lines = len(file.content.split('\n'))
            total_lines += lines
            
            if file.file_type == "implementation":
                functions = self._extract_function_signatures(file.content, language)
                function_count += len(functions)
            elif file.file_type == "test":
                # Count test methods/functions
                test_patterns = {
                    "python": r'def\s+test_\w+',
                    "javascript": r'(?:it|test)\s*\(',
                    "java": r'@Test|test\w+',
                    "cpp": r'TEST_CASE',
                    "go": r'func\s+Test\w+',
                    "rust": r'#\[test\]'
                }
                if language in test_patterns:
                    test_matches = re.findall(test_patterns[language], file.content)
                    test_count += len(test_matches)
        
        # Normalize complexity score (0.0 to 1.0)
        complexity = min(1.0, (total_lines / 200 + function_count * 0.1 + test_count * 0.05))
        return complexity
    
    def extract_all_problems(self) -> Dict[str, PolyglotProblem]:
        """Extract all problems across all languages"""
        logger.info("Starting extraction of all Polyglot benchmark problems...")
        
        all_problems = {}
        language_stats = {lang: {"problems": 0, "files": 0, "total_lines": 0} 
                         for lang in self.languages.keys()}
        
        # Process each language
        for language in self.languages.keys():
            logger.info(f"Processing {language} problems...")
            problem_dirs = self._get_problem_directories(language)
            
            for problem_dir in problem_dirs:
                problem = self.extract_single_problem(problem_dir, language)
                if problem:
                    problem_id = problem.problem_id
                    
                    # Merge with existing problem or create new one
                    if problem_id in all_problems:
                        # Add this language's files to existing problem
                        existing = all_problems[problem_id]
                        existing.files_by_language[language] = problem.files_by_language[language]
                        existing.cross_language_mapping.update(problem.cross_language_mapping)
                    else:
                        all_problems[problem_id] = problem
                    
                    # Update stats
                    stats = language_stats[language]
                    stats["problems"] += 1
                    stats["files"] += len(problem.files_by_language[language])
                    for file in problem.files_by_language[language]:
                        stats["total_lines"] += len(file.content.split('\n'))
        
        # Store stats and cache results
        self.language_stats = language_stats
        self.problems_cache = all_problems
        
        logger.info(f"Extraction complete! Found {len(all_problems)} unique problems across {len(self.languages)} languages")
        self._print_extraction_summary()
        
        return all_problems
    
    def _print_extraction_summary(self):
        """Print a summary of the extraction results"""
        print("\n" + "="*60)
        print("POLYGLOT BENCHMARK EXTRACTION SUMMARY")
        print("="*60)
        
        total_problems = len(self.problems_cache)
        print(f"Total unique problems: {total_problems}")
        
        print("\nPer-language statistics:")
        print("-" * 50)
        for lang, stats in self.language_stats.items():
            print(f"{lang.upper():>10}: {stats['problems']:>3} problems, "
                  f"{stats['files']:>4} files, {stats['total_lines']:>6} lines")
        
        # Cross-language coverage analysis
        print("\nCross-language problem coverage:")
        print("-" * 40)
        coverage_stats = {}
        for problem in self.problems_cache.values():
            lang_count = len(problem.files_by_language)
            coverage_stats[lang_count] = coverage_stats.get(lang_count, 0) + 1
        
        for lang_count in sorted(coverage_stats.keys(), reverse=True):
            problem_count = coverage_stats[lang_count]
            print(f"{lang_count} languages: {problem_count} problems")
        
        # Find problems available in all 6 languages
        complete_problems = [p for p in self.problems_cache.values() 
                           if len(p.files_by_language) == 6]
        print(f"\nProblems available in all 6 languages: {len(complete_problems)}")
        
        if complete_problems[:5]:  # Show first 5 as examples
            print("Examples:")
            for problem in complete_problems[:5]:
                print(f"  - {problem.problem_id}")
    
    def get_cross_language_problems(self, min_languages: int = 2) -> List[PolyglotProblem]:
        """Get problems that are available in at least min_languages"""
        return [problem for problem in self.problems_cache.values()
                if len(problem.files_by_language) >= min_languages]
    
    def get_problems_by_complexity(self, min_complexity: float = 0.0, 
                                  max_complexity: float = 1.0) -> List[PolyglotProblem]:
        """Get problems within a complexity range"""
        return [problem for problem in self.problems_cache.values()
                if min_complexity <= problem.complexity_score <= max_complexity]
    
    def export_to_json(self, output_path: str):
        """Export extracted problems to JSON format"""
        logger.info(f"Exporting problems to {output_path}")
        
        export_data = {
            "metadata": {
                "total_problems": len(self.problems_cache),
                "languages": list(self.languages.keys()),
                "extraction_stats": self.language_stats
            },
            "problems": {}
        }
        
        # Convert problems to serializable format
        for problem_id, problem in self.problems_cache.items():
            export_data["problems"][problem_id] = asdict(problem)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Export complete: {output_path}")
    
    def load_from_json(self, input_path: str):
        """Load previously extracted problems from JSON"""
        logger.info(f"Loading problems from {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.language_stats = data["metadata"]["extraction_stats"]
        
        # Reconstruct problem objects
        self.problems_cache = {}
        for problem_id, problem_data in data["problems"].items():
            # Convert file dictionaries back to ProblemFile objects
            files_by_language = {}
            for lang, files in problem_data["files_by_language"].items():
                files_by_language[lang] = [
                    ProblemFile(**file_data) for file_data in files
                ]
            
            problem = PolyglotProblem(
                problem_id=problem_data["problem_id"],
                problem_name=problem_data["problem_name"],
                description=problem_data["description"],
                difficulty=problem_data.get("difficulty"),
                files_by_language=files_by_language,
                cross_language_mapping=problem_data["cross_language_mapping"],
                complexity_score=problem_data["complexity_score"]
            )
            
            self.problems_cache[problem_id] = problem
        
        logger.info(f"Loaded {len(self.problems_cache)} problems from cache")

def main():
    """Main function for testing the extractor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract Polyglot benchmark problems")
    parser.add_argument("--benchmark-root", required=True, 
                       help="Path to polyglot-benchmark directory")
    parser.add_argument("--output", default="polyglot_problems.json",
                       help="Output JSON file")
    parser.add_argument("--languages", nargs="+", 
                       choices=["python", "javascript", "java", "cpp", "go", "rust"],
                       help="Languages to process (default: all)")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = PolyglotBenchmarkExtractor(args.benchmark_root)
    
    # Extract problems
    problems = extractor.extract_all_problems()
    
    # Export results
    extractor.export_to_json(args.output)
    
    # Show some examples
    print("\nExample problems with high cross-language coverage:")
    cross_lang_problems = extractor.get_cross_language_problems(min_languages=4)
    for problem in cross_lang_problems[:10]:
        languages = list(problem.files_by_language.keys())
        print(f"  {problem.problem_id}: {len(languages)} languages ({', '.join(languages)})")

if __name__ == "__main__":
    main()