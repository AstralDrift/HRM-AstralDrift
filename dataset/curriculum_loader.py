#!/usr/bin/env python3
"""
Curriculum Learning Loader for HRM Phase 1 Foundation Training

Implements weighted sampling and progressive difficulty scaling across
Tested-Python-Alpaca, Glaive-Function-Calling, and AgentInstruct datasets.

Key Features:
- Progressive difficulty increase during Phase 1 training
- Weighted sampling respecting dataset characteristics
- Seamless integration with existing HRM training pipeline
- Preserves hierarchical reasoning patterns from AgentInstruct
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass
import logging

from multi_dataset_config import MultiDatasetManager, DatasetConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CurriculumSample:
    """Single training sample with curriculum metadata"""
    instance_id: str
    input_text: str
    target_text: str
    dataset_name: str
    difficulty: float
    features: List[str]
    metadata: Dict

class CurriculumLoader:
    """
    Curriculum learning loader for Phase 1 foundation training
    
    Implements strategic sampling across three core datasets:
    1. Glaive (simple tool use) -> 2. Tested-Python (coding) -> 3. AgentInstruct (hierarchical)
    """
    
    def __init__(self, 
                 config_path: str = "config/multi_dataset_config.json",
                 phase: int = 1,
                 seed: int = 42):
        self.manager = MultiDatasetManager(config_path)
        self.phase = phase
        self.datasets = self.manager.get_phase_datasets(phase)
        self.sampling_weights = self.manager.get_sampling_weights(phase)
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Load all datasets
        self.loaded_datasets: Dict[str, List[CurriculumSample]] = {}
        self._load_all_datasets()
        
        # Curriculum parameters
        self.curriculum_schedule = self._setup_curriculum_schedule()
        
        logger.info(f"🎓 Curriculum Loader initialized for Phase {phase}")
        logger.info(f"📊 Loaded {sum(len(ds) for ds in self.loaded_datasets.values()):,} total samples")
        
    def _load_all_datasets(self):
        """Load all datasets for the current phase"""
        for dataset_config in self.datasets:
            logger.info(f"📚 Loading {dataset_config.name}...")
            
            instances_path = Path(dataset_config.path) / "instances.json"
            if not instances_path.exists():
                logger.warning(f"⚠️  Dataset path not found: {instances_path}")
                continue
                
            with open(instances_path, 'r') as f:
                raw_instances = json.load(f)
            
            # Convert to curriculum samples
            samples = []
            for instance in raw_instances:
                sample = CurriculumSample(
                    instance_id=instance.get("instance_id", f"unknown_{len(samples)}"),
                    input_text=instance.get("input_text", ""),
                    target_text=instance.get("target_text", ""),
                    dataset_name=dataset_config.name,
                    difficulty=dataset_config.difficulty,
                    features=dataset_config.features,
                    metadata=instance.get("metadata", {})
                )
                samples.append(sample)
            
            self.loaded_datasets[dataset_config.name] = samples
            logger.info(f"✅ Loaded {len(samples):,} samples from {dataset_config.name}")
    
    def _setup_curriculum_schedule(self) -> Dict[str, Dict]:
        """Setup curriculum progression schedule for Phase 1"""
        return {
            # Early training: Focus on simple tool use
            "early": {
                "epoch_range": (5, 10),
                "weights": {
                    "Glaive-Function-Calling-v2.1": 0.5,  # Boost simple tool use
                    "Tested-143k-Python-Alpaca": 0.3,
                    "AgentInstruct-1M-CoT": 0.2
                },
                "difficulty_threshold": 0.4
            },
            # Mid training: Balanced coding and reasoning
            "middle": {
                "epoch_range": (11, 15),
                "weights": {
                    "Glaive-Function-Calling-v2.1": 0.3,
                    "Tested-143k-Python-Alpaca": 0.4,  # Boost coding
                    "AgentInstruct-1M-CoT": 0.3
                },
                "difficulty_threshold": 0.6
            },
            # Late training: Hierarchical reasoning emphasis
            "late": {
                "epoch_range": (16, 20),
                "weights": {
                    "Glaive-Function-Calling-v2.1": 0.25,
                    "Tested-143k-Python-Alpaca": 0.35,
                    "AgentInstruct-1M-CoT": 0.4  # Boost hierarchical
                },
                "difficulty_threshold": 0.8
            }
        }
    
    def get_curriculum_stage(self, epoch: int) -> str:
        """Determine current curriculum stage based on epoch"""
        for stage, config in self.curriculum_schedule.items():
            start, end = config["epoch_range"]
            if start <= epoch <= end:
                return stage
        return "late"  # Default to late stage
    
    def sample_batch(self, 
                     batch_size: int, 
                     epoch: int,
                     exclude_features: Optional[List[str]] = None) -> List[CurriculumSample]:
        """
        Sample a batch with curriculum-aware weighting
        
        Args:
            batch_size: Number of samples to return
            epoch: Current training epoch for curriculum scheduling
            exclude_features: Features to avoid (e.g., for ablation studies)
        """
        stage = self.get_curriculum_stage(epoch)
        stage_config = self.curriculum_schedule[stage]
        
        # Get current weights
        current_weights = stage_config["weights"]
        difficulty_threshold = stage_config["difficulty_threshold"]
        
        # Sample from each dataset proportionally
        batch_samples = []
        
        for dataset_name, weight in current_weights.items():
            if dataset_name not in self.loaded_datasets:
                logger.warning(f"⚠️  Dataset {dataset_name} not loaded, skipping")
                continue
                
            # Calculate samples needed from this dataset
            n_samples = int(batch_size * weight)
            if n_samples == 0:
                continue
                
            # Get available samples (filtered by difficulty and features)
            available_samples = self._filter_samples(
                self.loaded_datasets[dataset_name],
                difficulty_threshold,
                exclude_features
            )
            
            if not available_samples:
                logger.warning(f"⚠️  No suitable samples from {dataset_name}")
                continue
            
            # Sample with replacement if needed
            if len(available_samples) < n_samples:
                sampled = np.random.choice(available_samples, n_samples, replace=True)
            else:
                sampled = np.random.choice(available_samples, n_samples, replace=False)
                
            batch_samples.extend(sampled)
        
        # Fill remaining slots if batch is incomplete
        remaining = batch_size - len(batch_samples)
        if remaining > 0:
            all_available = []
            for dataset_name in current_weights.keys():
                if dataset_name in self.loaded_datasets:
                    all_available.extend(
                        self._filter_samples(
                            self.loaded_datasets[dataset_name],
                            difficulty_threshold,
                            exclude_features
                        )
                    )
            
            if all_available:
                additional = np.random.choice(all_available, remaining, replace=True)
                batch_samples.extend(additional)
        
        # Shuffle final batch
        random.shuffle(batch_samples)
        
        return batch_samples[:batch_size]
    
    def _filter_samples(self, 
                       samples: List[CurriculumSample],
                       difficulty_threshold: float,
                       exclude_features: Optional[List[str]]) -> List[CurriculumSample]:
        """Filter samples by difficulty and feature constraints"""
        filtered = []
        
        for sample in samples:
            # Check difficulty
            if sample.difficulty > difficulty_threshold:
                continue
                
            # Check feature exclusions
            if exclude_features:
                if any(feature in sample.features for feature in exclude_features):
                    continue
            
            filtered.append(sample)
        
        return filtered
    
    def get_batch_iterator(self,
                          batch_size: int,
                          epoch: int,
                          shuffle: bool = True) -> Iterator[List[CurriculumSample]]:
        """
        Get iterator over all samples for an epoch
        
        Args:
            batch_size: Size of each batch
            epoch: Current epoch for curriculum scheduling
            shuffle: Whether to shuffle samples
        """
        # Get all samples for current curriculum stage
        stage = self.get_curriculum_stage(epoch)
        stage_config = self.curriculum_schedule[stage]
        
        all_samples = []
        for dataset_name, weight in stage_config["weights"].items():
            if dataset_name in self.loaded_datasets:
                # Sample proportionally from each dataset
                dataset_samples = self.loaded_datasets[dataset_name]
                n_samples = int(len(dataset_samples) * weight)
                
                if shuffle:
                    sampled = random.sample(dataset_samples, min(n_samples, len(dataset_samples)))
                else:
                    sampled = dataset_samples[:n_samples]
                
                all_samples.extend(sampled)
        
        if shuffle:
            random.shuffle(all_samples)
        
        # Yield batches
        for i in range(0, len(all_samples), batch_size):
            yield all_samples[i:i + batch_size]
    
    def print_curriculum_summary(self, epoch: int):
        """Print current curriculum state"""
        stage = self.get_curriculum_stage(epoch)
        stage_config = self.curriculum_schedule[stage]
        
        print(f"\n🎓 Curriculum Summary - Epoch {epoch} ({stage.upper()} stage)")
        print("=" * 60)
        print(f"📊 Difficulty threshold: {stage_config['difficulty_threshold']}")
        print(f"⚖️  Dataset weights:")
        
        for dataset_name, weight in stage_config["weights"].items():
            if dataset_name in self.loaded_datasets:
                count = len(self.loaded_datasets[dataset_name])
                expected_samples = int(count * weight)
                print(f"   • {dataset_name}: {weight:.1%} ({expected_samples:,} samples)")
        
        total_samples = sum(len(ds) for ds in self.loaded_datasets.values())
        print(f"📈 Total available: {total_samples:,} samples")
    
    def get_dataset_stats(self) -> Dict:
        """Get comprehensive dataset statistics"""
        stats = {
            "total_samples": sum(len(ds) for ds in self.loaded_datasets.values()),
            "datasets": {},
            "feature_distribution": {},
            "difficulty_distribution": {}
        }
        
        # Per-dataset stats
        for name, samples in self.loaded_datasets.items():
            stats["datasets"][name] = {
                "count": len(samples),
                "avg_difficulty": np.mean([s.difficulty for s in samples]),
                "features": list(set().union(*[s.features for s in samples]))
            }
        
        # Feature distribution across all samples
        all_features = []
        all_difficulties = []
        for samples in self.loaded_datasets.values():
            for sample in samples:
                all_features.extend(sample.features)
                all_difficulties.append(sample.difficulty)
        
        from collections import Counter
        stats["feature_distribution"] = dict(Counter(all_features))
        stats["difficulty_distribution"] = {
            "mean": np.mean(all_difficulties),
            "std": np.std(all_difficulties),
            "min": np.min(all_difficulties),
            "max": np.max(all_difficulties)
        }
        
        return stats

def main():
    """Test the curriculum loader"""
    loader = CurriculumLoader()
    
    # Test sampling at different epochs
    for epoch in [5, 10, 15, 20]:
        print(f"\n🧪 Testing epoch {epoch}")
        loader.print_curriculum_summary(epoch)
        
        batch = loader.sample_batch(batch_size=32, epoch=epoch)
        print(f"✅ Sampled batch of {len(batch)} samples")
        
        # Show dataset distribution in batch
        dataset_counts = {}
        for sample in batch:
            dataset_counts[sample.dataset_name] = dataset_counts.get(sample.dataset_name, 0) + 1
        
        print("📊 Batch composition:")
        for dataset, count in dataset_counts.items():
            print(f"   • {dataset}: {count} samples ({count/len(batch):.1%})")
    
    # Print overall stats
    stats = loader.get_dataset_stats()
    print(f"\n📈 Overall Statistics:")
    print(f"Total samples: {stats['total_samples']:,}")
    print(f"Feature distribution: {stats['feature_distribution']}")

if __name__ == "__main__":
    main()