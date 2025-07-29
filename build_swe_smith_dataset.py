#!/usr/bin/env python3
"""
SWE-smith Dataset Builder for HRM Code Generation Training

This script builds a comprehensive training dataset from SWE-smith's 52K+ 
real GitHub issues using the sophisticated task analysis and multi-agent 
coordination system for world-class code generation performance.

Usage:
    python build_swe_smith_dataset.py --output-dir data/swe-smith-full --num-samples 1000
    python build_swe_smith_dataset.py --output-dir data/swe-smith-small --num-samples 200 --quick-test
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import our sophisticated SWE-smith integration
from dataset.swe_smith_integration import HRMSWESmithDataset, SWESmithTaskRegistry


def setup_logging(verbose: bool = False):
    """Configure logging for the dataset builder"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('swe_smith_dataset_build.log')
        ]
    )


def print_header(title: str):
    """Print formatted header"""
    print(f"\n{'='*70}")
    print(f"🚀 {title}")
    print(f"{'='*70}")


def print_success(message: str):
    """Print success message"""
    print(f"✅ {message}")


def print_error(message: str):
    """Print error message"""
    print(f"❌ {message}")


def print_info(message: str):
    """Print info message"""
    print(f"ℹ️  {message}")


def print_stats(stats: Dict[str, Any]):
    """Print processing statistics"""
    print(f"\n📊 Processing Statistics:")
    print(f"   Tasks processed: {stats['tasks_processed']}")
    print(f"   Tasks failed: {stats['tasks_failed']}")
    print(f"   Success rate: {(stats['tasks_processed'] / (stats['tasks_processed'] + stats['tasks_failed']) * 100):.1f}%")
    print(f"   Average processing time: {stats['average_processing_time']:.2f}s")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Cache misses: {stats['cache_misses']}")
    if stats['cache_hits'] + stats['cache_misses'] > 0:
        cache_hit_rate = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) * 100
        print(f"   Cache hit rate: {cache_hit_rate:.1f}%")


async def build_swe_smith_dataset(
    output_dir: str,
    num_samples: int = 1000,
    batch_size: int = 32,
    clear_cache: bool = False,
    quick_test: bool = False
) -> bool:
    """
    Build comprehensive SWE-smith dataset for HRM training
    
    Args:
        output_dir: Directory to save the dataset
        num_samples: Number of GitHub issues to process
        batch_size: Batch size for parallel processing
        clear_cache: Whether to clear the processing cache
        quick_test: Use smaller batch sizes for testing
        
    Returns:
        True if successful, False otherwise
    """
    
    print_header("SWE-smith Dataset Builder for HRM Code Generation")
    
    # Configure for quick testing
    if quick_test:
        batch_size = min(batch_size, 8)
        num_samples = min(num_samples, 50)
        print_info(f"Quick test mode: limiting to {num_samples} samples with batch size {batch_size}")
    
    # Setup configuration
    config = {
        'cache_dir': 'cache/swe_smith_hrm',
        'max_workers': min(8, batch_size),
        'enable_parallel_processing': True,
        'use_repository_profiles': True
    }
    
    print_info(f"Configuration: {config}")
    
    try:
        # Initialize the sophisticated dataset creator
        print_info("Initializing SWE-smith integration...")
        dataset_creator = HRMSWESmithDataset(config)
        
        # Clear cache if requested
        if clear_cache:
            print_info("Clearing processing cache...")
            dataset_creator.registry.clear_cache()
        
        # Create the dataset using the advanced processing pipeline
        print_info(f"Building dataset with {num_samples} GitHub issues...")
        print_info("This uses sophisticated task analysis including:")
        print_info("  • Multi-language detection and complexity analysis")
        print_info("  • Tool workflow planning and coordination")  
        print_info("  • Domain-specific agent assignment")
        print_info("  • Repository profile integration")
        print_info("  • Hierarchical reasoning data structure generation")
        
        start_time = time.time()
        
        # Build the dataset
        metadata = await dataset_creator.create_hrm_dataset(
            num_samples=num_samples,
            output_dir=output_dir
        )
        
        build_time = time.time() - start_time
        
        # Get processing statistics
        stats = dataset_creator.registry.get_processing_stats()
        print_stats(stats)
        
        print_success(f"Dataset created successfully in {build_time:.1f} seconds!")
        print_info(f"Output directory: {output_dir}")
        print_info(f"Dataset metadata: {metadata.__dict__}")
        
        # Validate dataset
        print_info("Validating dataset...")
        if validate_dataset(output_dir):
            print_success("Dataset validation passed!")
            return True
        else:
            print_error("Dataset validation failed!")
            return False
            
    except Exception as e:
        print_error(f"Failed to build dataset: {e}")
        logging.exception("Dataset building failed")
        return False


def validate_dataset(output_dir: str) -> bool:
    """Validate the created dataset"""
    
    output_path = Path(output_dir)
    
    # Check required files exist
    required_files = ['instances.json', 'metadata.json']
    for filename in required_files:
        file_path = output_path / filename
        if not file_path.exists():
            print_error(f"Missing required file: {filename}")
            return False
    
    try:
        # Validate instances file
        import json
        
        instances_file = output_path / 'instances.json'
        with open(instances_file, 'r') as f:
            instances = json.load(f)
        
        if not isinstance(instances, list) or len(instances) == 0:
            print_error("Instances file is empty or invalid")
            return False
        
        # Validate sample instance structure
        sample = instances[0]
        required_keys = ['puzzle_id', 'instance_id', 'input_text', 'target_text', 'coordination_target', 'metadata']
        for key in required_keys:
            if key not in sample:
                print_error(f"Missing required key in instance: {key}")
                return False
        
        # Validate metadata
        metadata_file = output_path / 'metadata.json'
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Check if metadata has expected structure (flexibility for different formats)
        expected_count = len(instances)
        metadata_count = metadata.get('num_instances', metadata.get('mean_puzzle_examples', 0))
        
        if abs(metadata_count - expected_count) > 0.1:  # Allow small floating point differences
            print_error(f"Metadata count ({metadata_count}) doesn't match actual instances ({expected_count})")
            return False
        
        print_info(f"Dataset validation successful: {len(instances)} instances")
        return True
        
    except Exception as e:
        print_error(f"Dataset validation error: {e}")
        return False


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Build sophisticated SWE-smith dataset for HRM code generation training"
    )
    
    parser.add_argument(
        '--output-dir', 
        default='data/swe-smith-hrm',
        help='Output directory for the dataset'
    )
    
    parser.add_argument(
        '--num-samples', 
        type=int, 
        default=1000,
        help='Number of GitHub issues to process (default: 1000)'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=32,
        help='Batch size for parallel processing (default: 32)'
    )
    
    parser.add_argument(
        '--clear-cache', 
        action='store_true',
        help='Clear the processing cache before building'
    )
    
    parser.add_argument(
        '--quick-test', 
        action='store_true',
        help='Run in quick test mode with smaller batches'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Run the dataset builder
    success = asyncio.run(build_swe_smith_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        clear_cache=args.clear_cache,
        quick_test=args.quick_test
    ))
    
    if success:
        print_header("Dataset Build Complete")
        print_success("SWE-smith dataset ready for HRM training!")
        print_info("Next steps:")
        print_info("1. Use train_code_generation.py with --config swe_smith_small")
        print_info("2. Monitor training with W&B")
        print_info("3. Evaluate on code generation benchmarks")
        return 0
    else:
        print_error("Dataset build failed")
        return 1


if __name__ == "__main__":
    exit(main())