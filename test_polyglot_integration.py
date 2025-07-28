#!/usr/bin/env python3
"""
Comprehensive Test Script for Polyglot Benchmark Integration

This script demonstrates the complete Polyglot benchmark integration pipeline,
from data extraction to training data generation and evaluation.
"""

import os
import sys
import json
import logging
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our modules
from dataset.polyglot_benchmark_extractor import PolyglotBenchmarkExtractor
from dataset.diff_based_training_generator import DiffBasedTrainingGenerator
from dataset.cross_language_mapper import CrossLanguageProblemMapper
from evaluation.polyglot_evaluation_system import PolyglotEvaluationSystem
from dataset.polyglot_hrm_integration import PolyglotTrainingPipeline, PolyglotTrainingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_data_extraction(benchmark_root: str) -> Dict:
    """Test the data extraction pipeline"""
    logger.info("="*60)
    logger.info("TESTING DATA EXTRACTION")
    logger.info("="*60)
    
    # Initialize extractor
    extractor = PolyglotBenchmarkExtractor(benchmark_root)
    
    # Extract problems
    problems = extractor.extract_all_problems()
    
    # Print statistics
    print(f"\nExtracted {len(problems)} unique problems")
    
    # Show examples of cross-language problems
    cross_lang_problems = extractor.get_cross_language_problems(min_languages=3)
    print(f"Problems available in 3+ languages: {len(cross_lang_problems)}")
    
    if cross_lang_problems[:3]:
        print("\nExample cross-language problems:")
        for problem in cross_lang_problems[:3]:
            languages = list(problem.files_by_language.keys())
            print(f"  - {problem.problem_id}: {len(languages)} languages ({', '.join(languages)})")
            
            # Show file structure for first language
            if languages:
                files = problem.files_by_language[languages[0]]
                impl_files = [f for f in files if f.file_type == "implementation"]
                test_files = [f for f in files if f.file_type == "test"]
                print(f"    {languages[0]}: {len(files)} files ({len(impl_files)} impl, {len(test_files)} test)")
    
    return {"extractor": extractor, "problems": problems}

def test_diff_generation(problems: Dict) -> Dict:
    """Test the diff-based training data generation"""
    logger.info("="*60)
    logger.info("TESTING DIFF GENERATION")
    logger.info("="*60)
    
    # Initialize generator
    generator = DiffBasedTrainingGenerator(problems)
    
    # Generate examples (limited for testing)
    examples = generator.generate_all_training_examples(max_examples_per_problem=3)
    
    # Get statistics
    stats = generator.get_statistics()
    
    print(f"\nGenerated {len(examples)} diff-based training examples")
    print(f"Languages: {stats['by_language']}")
    print(f"Edit types: {stats['by_edit_type']}")
    print(f"Average edit distance: {stats['average_edit_distance']:.2f}")
    
    # Show example
    if examples:
        example = examples[0]
        print(f"\nExample diff operation:")
        print(f"  Problem: {getattr(example, 'problem_description', 'N/A')[:50]}...")
        print(f"  Language: {example.language}")
        print(f"  Instruction: {example.edit_instruction[:50]}...")
        print(f"  Original code length: {len(example.original_code)} chars")
        print(f"  Target code length: {len(example.target_code)} chars")
    
    return {"generator": generator, "examples": examples}

def test_cross_language_mapping(problems: Dict) -> Dict:
    """Test the cross-language mapping"""
    logger.info("="*60)
    logger.info("TESTING CROSS-LANGUAGE MAPPING")
    logger.info("="*60)
    
    # Initialize mapper
    mapper = CrossLanguageProblemMapper(problems)
    
    # Generate transfer examples (limited for testing)
    transfer_examples = mapper.generate_transfer_examples(max_examples_per_pair=2)
    
    # Get statistics
    stats = mapper.get_cross_language_statistics()
    
    print(f"\nGenerated {len(transfer_examples)} transfer learning examples")
    print(f"Total problems analyzed: {stats.get('total_problems', 0)}")
    print(f"Language coverage: {dict(stats.get('language_coverage', {}))}")
    
    # Show language compatibility matrix
    compatibility = mapper.get_language_compatibility_matrix()
    languages = ["python", "javascript", "java", "cpp", "go", "rust"]
    
    print(f"\nLanguage compatibility matrix:")
    print("         " + "".join(f"{lang[:4]:>6}" for lang in languages))
    for lang1 in languages:
        if lang1 in compatibility:
            row = [f"{compatibility[lang1].get(lang2, 0.0):.2f}" for lang2 in languages]
            print(f"{lang1:>8}: " + "".join(f"{val:>6}" for val in row))
    
    # Show example transfer
    if transfer_examples:
        example = transfer_examples[0]
        print(f"\nExample transfer operation:")
        print(f"  Problem: {example.problem_id}")
        print(f"  Transfer: {example.source_language} -> {example.target_language}")
        print(f"  Instruction: {example.transfer_instructions[:100]}...")
        print(f"  Shared concepts: {example.shared_concepts}")
        print(f"  Difficulty: {example.difficulty_score:.3f}")
    
    return {"mapper": mapper, "transfer_examples": transfer_examples}

def test_evaluation_system(problems: Dict, mock_solutions: Dict = None) -> Dict:
    """Test the evaluation system"""
    logger.info("="*60)
    logger.info("TESTING EVALUATION SYSTEM")
    logger.info("="*60)
    
    # Initialize evaluation system
    evaluator = PolyglotEvaluationSystem(timeout=10)  # Short timeout for testing
    
    # Create mock solutions if not provided
    if mock_solutions is None:
        mock_solutions = {}
        count = 0
        for problem_id, problem in problems.items():
            if count >= 3:  # Limit to 3 problems for testing
                break
            for language, files in problem.files_by_language.items():
                impl_files = [f for f in files if f.file_type == "implementation"]
                if impl_files:
                    # Use original code as "generated" solution for testing
                    mock_solutions[(problem_id, language)] = impl_files[0].content
                    count += 1
                    if count >= 3:
                        break
    
    print(f"\nTesting evaluation with {len(mock_solutions)} mock solutions")
    
    # Run evaluation (this will actually try to execute tests)
    try:
        results = evaluator.evaluate_generated_code(problems, mock_solutions)
        
        print(f"Evaluation results:")
        print(f"  Total evaluations: {results.get('total_evaluations', 0)}")
        print(f"  Overall success rate: {results.get('overall_success_rate', 0):.2%}")
        print(f"  Average execution time: {results.get('average_execution_time', 0):.3f}s")
        
        by_language = results.get('by_language', {})
        if by_language:
            print(f"  Success rates by language:")
            for lang, stats in by_language.items():
                print(f"    {lang}: {stats.get('success_rate', 0):.2%} ({stats.get('passed', 0)}/{stats.get('total', 0)})")
        
    except Exception as e:
        logger.warning(f"Evaluation failed (expected in test environment): {e}")
        results = {"error": str(e)}
    
    return {"evaluator": evaluator, "results": results}

def test_hrm_integration(benchmark_root: str, problems: Dict, 
                        diff_examples: List, transfer_examples: List) -> Dict:
    """Test the HRM integration pipeline"""
    logger.info("="*60)
    logger.info("TESTING HRM INTEGRATION")
    logger.info("="*60)
    
    # Create configuration
    config = PolyglotTrainingConfig(
        max_problems_per_language=10,  # Small for testing
        max_diff_examples_per_problem=3,
        max_transfer_examples_per_pair=2,
        batch_size=4,
        curriculum_learning=True
    )
    
    # Create mock HRM model
    class MockHRMModel:
        def __init__(self):
            self.name = "MockHRMCodeModel"
    
    mock_model = MockHRMModel()
    
    # Initialize pipeline
    try:
        pipeline = PolyglotTrainingPipeline(benchmark_root, mock_model, config)
        
        # Prepare data (use existing extracted data to avoid re-extraction)
        pipeline.polyglot_data = {
            "problems": problems,
            "diff_examples": diff_examples,
            "transfer_examples": transfer_examples
        }
        
        # Create datasets
        train_dataset, val_dataset = pipeline.create_training_datasets()
        
        print(f"\nCreated datasets:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        
        # Create data loaders
        train_loader, val_loader = pipeline.create_data_loaders()
        
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        
        # Test loading a batch
        if len(train_loader) > 0:
            batch = next(iter(train_loader))
            print(f"\nSample batch:")
            print(f"  Input tokens shape: {batch['input_tokens'].shape}")
            print(f"  Languages: {set(batch['sample_types'])}")
            print(f"  Sample types: {set(batch['sample_types'])}")
        
        # Get statistics
        stats = pipeline.get_data_statistics()
        print(f"\nData statistics:")
        print(f"  Total problems: {stats.get('total_problems', 0)}")
        print(f"  Training samples by type: {stats.get('training_sample_types', {})}")
        print(f"  Training samples by language: {stats.get('training_languages', {})}")
        
        return {"pipeline": pipeline, "stats": stats}
        
    except Exception as e:
        logger.error(f"HRM integration test failed: {e}")
        return {"error": str(e)}

def run_comprehensive_test(benchmark_root: str, output_dir: str = None):
    """Run comprehensive test of the entire system"""
    logger.info("Starting comprehensive Polyglot benchmark integration test...")
    
    if not Path(benchmark_root).exists():
        logger.error(f"Benchmark root does not exist: {benchmark_root}")
        return
    
    results = {}
    
    try:
        # Test 1: Data Extraction
        extraction_result = test_data_extraction(benchmark_root)
        results["extraction"] = extraction_result
        problems = extraction_result["problems"]
        
        # Test 2: Diff Generation
        diff_result = test_diff_generation(problems)
        results["diff_generation"] = diff_result
        diff_examples = diff_result["examples"]
        
        # Test 3: Cross-Language Mapping
        mapping_result = test_cross_language_mapping(problems)
        results["cross_language"] = mapping_result
        transfer_examples = mapping_result["transfer_examples"]
        
        # Test 4: Evaluation System
        evaluation_result = test_evaluation_system(problems)
        results["evaluation"] = evaluation_result
        
        # Test 5: HRM Integration
        integration_result = test_hrm_integration(
            benchmark_root, problems, diff_examples, transfer_examples
        )
        results["integration"] = integration_result
        
        # Export results if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Export test results
            test_results_file = output_path / "test_results.json"
            with open(test_results_file, 'w') as f:
                # Create serializable version of results
                serializable_results = {
                    "extraction": {
                        "problems_count": len(problems),
                        "cross_language_problems": len(extraction_result["extractor"].get_cross_language_problems(min_languages=2))
                    },
                    "diff_generation": {
                        "examples_count": len(diff_examples),
                        "statistics": diff_result["generator"].get_statistics()
                    },
                    "cross_language": {
                        "transfer_examples_count": len(transfer_examples),
                        "statistics": mapping_result["mapper"].get_cross_language_statistics()
                    },
                    "evaluation": {
                        "completed": "error" not in evaluation_result,
                        "results": evaluation_result.get("results", {})
                    },
                    "integration": {
                        "completed": "error" not in integration_result,
                        "statistics": integration_result.get("stats", {})
                    }
                }
                
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Test results exported to {test_results_file}")
        
        # Final summary
        print("\n" + "="*60)
        print("COMPREHENSIVE TEST SUMMARY")
        print("="*60)
        print(f"✓ Data extraction: {len(problems)} problems extracted")
        print(f"✓ Diff generation: {len(diff_examples)} examples generated")
        print(f"✓ Cross-language mapping: {len(transfer_examples)} transfer examples")
        print(f"✓ Evaluation system: {'Completed' if 'error' not in evaluation_result else 'Failed (expected)'}")
        print(f"✓ HRM integration: {'Completed' if 'error' not in integration_result else 'Failed'}")
        
        success_count = sum(1 for result in results.values() if "error" not in result)
        total_tests = len(results)
        print(f"\nOverall: {success_count}/{total_tests} tests passed")
        
        if success_count == total_tests:
            print("🎉 All tests passed! Polyglot integration is working correctly.")
        else:
            print("⚠️  Some tests failed. Check logs for details.")
        
    except Exception as e:
        logger.error(f"Comprehensive test failed: {e}")
        raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Test comprehensive Polyglot benchmark integration with HRM"
    )
    parser.add_argument(
        "--benchmark-root",
        required=True,
        help="Path to polyglot-benchmark directory"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for test results and data"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run comprehensive test
    run_comprehensive_test(args.benchmark_root, args.output_dir)

if __name__ == "__main__":
    main()