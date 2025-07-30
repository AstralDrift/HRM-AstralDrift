"""
Multi-Dataset Configuration for HRM SOTA Challenge
Supports phased training with curriculum learning across 8+ datasets
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from pathlib import Path
import json

@dataclass
class DatasetConfig:
    """Configuration for individual dataset"""
    name: str
    path: str
    format: str  # 'json', 'jsonl', 'huggingface'
    size: int
    weight: float  # Sampling weight during training
    phase: int  # Training phase (1-4)
    difficulty: float  # Curriculum difficulty (0.0-1.0)
    features: List[str] = field(default_factory=list)  # ['tool_use', 'coding', 'multimodal']
    
@dataclass 
class PhaseConfig:
    """Configuration for training phase"""
    phase_id: int
    name: str
    epochs: tuple  # (start, end)
    datasets: List[str]
    target_metrics: Dict[str, float]
    description: str

class MultiDatasetManager:
    """Manages multiple datasets across training phases"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.datasets: Dict[str, DatasetConfig] = {}
        self.phases: Dict[int, PhaseConfig] = {}
        
        if config_path:
            self.load_config(config_path)
        else:
            self._setup_default_config()
    
    def _setup_default_config(self):
        """Setup default SOTA challenge configuration"""
        
        # Phase 1: Foundation (Epochs 5-20)
        self.datasets.update({
            "tested_python_alpaca": DatasetConfig(
                name="Tested-143k-Python-Alpaca",
                path="data/tested_python_alpaca",
                format="json",
                size=143327,
                weight=0.4,
                phase=1,
                difficulty=0.3,
                features=["coding", "python", "verified"]
            ),
            "glaive_function_calling": DatasetConfig(
                name="Glaive-Function-Calling-v2.1", 
                path="data/glaive_function_calling",
                format="json",
                size=118000,  # 113k + 5k multilingual
                weight=0.3,
                phase=1,
                difficulty=0.4,
                features=["tool_use", "function_calling", "multilingual"]
            ),
            "agent_instruct": DatasetConfig(
                name="AgentInstruct-1M-CoT",
                path="data/agent_instruct_1m",
                format="json", 
                size=999465,  # Actual built size
                weight=0.3,
                phase=1,
                difficulty=0.5,
                features=["hierarchical_reasoning", "cot_enhanced", "agent_coordination", "multi_turn"]
            )
        })
        
        # Phase 2: Advanced Tool Use (Epochs 21-50)
        self.datasets.update({
            "toolbench_v2": DatasetConfig(
                name="ToolBench-v2.0",
                path="data/toolbench_v2", 
                format="json",
                size=18000,  # 16k + 2k failure modes
                weight=0.3,
                phase=2,
                difficulty=0.6,
                features=["tool_use", "multi_tool", "failure_handling"]
            ),
            "api_bench": DatasetConfig(
                name="APIBench-v1.2",
                path="data/api_bench",
                format="json",
                size=1800,
                weight=0.2,
                phase=2, 
                difficulty=0.7,
                features=["api_calls", "noise_injection", "robust"]
            ),
            "magicoder_evol": DatasetConfig(
                name="Magicoder-Evol-Instruct-110K",
                path="data/magicoder_evol",
                format="json",
                size=110000,
                weight=0.25,
                phase=2,
                difficulty=0.6,
                features=["code_evolution", "debugging", "iterative"]
            ),
            "code_feedback": DatasetConfig(
                name="CodeFeedback-Filtered-Instruction", 
                path="data/code_feedback",
                format="json",
                size=156000,
                weight=0.25,
                phase=2,
                difficulty=0.7,
                features=["feedback_loops", "plan_code_test", "filtered"]
            )
        })
        
        # Phase 3: RLHF & Autonomous (Epochs 51-80)
        self.datasets.update({
            "mle_bench": DatasetConfig(
                name="MLE-Bench",
                path="data/mle_bench",
                format="json", 
                size=85,  # 75 + 10 new competitions
                weight=0.4,
                phase=3,
                difficulty=0.8,
                features=["ml_pipelines", "end_to_end", "kaggle"]
            ),
            "api_bench_expanded": DatasetConfig(
                name="APIBench-Expanded",
                path="data/api_bench_expanded",
                format="json",
                size=10000,  # Synthetic expansion
                weight=0.6,
                phase=3,
                difficulty=0.8,
                features=["tool_orchestration", "synthetic", "scale"]
            )
        })
        
        # Phase 4: Multimodal Polish (Epochs 81-100)
        self.datasets.update({
            "gaia_expanded": DatasetConfig(
                name="GAIA-Expanded",
                path="data/gaia_expanded", 
                format="json",
                size=10000,  # 450 -> 10k synthetic
                weight=0.4,
                phase=4,
                difficulty=0.9,
                features=["multimodal", "tool_use", "complex_reasoning"]
            ),
            "agent_flan": DatasetConfig(
                name="Agent-FLAN",
                path="data/agent_flan",
                format="json",
                size=34000,
                weight=0.3,
                phase=4,
                difficulty=0.8,
                features=["multi_agent", "coordination", "filtered"]
            ),
            "swe_bench_hard": DatasetConfig(
                name="SWE-Bench-Hard",
                path="data/swe_bench_hard",
                format="json", 
                size=100,  # Harder subset
                weight=0.3,
                phase=4,
                difficulty=1.0,
                features=["real_world", "github", "complex"]
            )
        })
        
        # Setup phases  
        self.phases = {
            1: PhaseConfig(
                phase_id=1,
                name="Foundation Building",
                epochs=(5, 20),
                datasets=["tested_python_alpaca", "glaive_function_calling", "agent_instruct"],
                target_metrics={
                    "token_accuracy": 0.5,
                    "compilation_success": 1.0,
                    "loss": 15.0
                },
                description="Build robust tool-use and coding fundamentals"
            ),
            2: PhaseConfig(
                phase_id=2, 
                name="Advanced Tool Use",
                epochs=(21, 50),
                datasets=["toolbench_v2", "api_bench", "magicoder_evol", "code_feedback"],
                target_metrics={
                    "loss": 1.2,
                    "swe_search_candidates": 15.0,
                    "tiered_accuracy": 0.25
                },
                description="Master complex tool chains and multi-step reasoning"
            ),
            3: PhaseConfig(
                phase_id=3,
                name="RLHF & Autonomous",
                epochs=(51, 80), 
                datasets=["mle_bench", "code_feedback", "api_bench_expanded"],
                target_metrics={
                    "loss": 0.8,
                    "tiered_accuracy": 0.8,
                    "reverse_integration_gate": 0.7
                },
                description="Optimize hierarchical reasoning and autonomy"
            ),
            4: PhaseConfig(
                phase_id=4,
                name="Multimodal Polish", 
                epochs=(81, 100),
                datasets=["gaia_expanded", "agent_flan", "swe_bench_hard"],
                target_metrics={
                    "loss": 1.0,
                    "tiered_accuracy": 0.95,
                    "compilation_success": 1.0
                },
                description="Achieve SOTA performance across all benchmarks"
            )
        }
    
    def get_phase_datasets(self, phase: int) -> List[DatasetConfig]:
        """Get datasets for specific phase"""
        if phase not in self.phases:
            raise ValueError(f"Phase {phase} not found")
            
        phase_config = self.phases[phase]
        return [self.datasets[name] for name in phase_config.datasets if name in self.datasets]
    
    def get_sampling_weights(self, phase: int) -> Dict[str, float]:
        """Get sampling weights for phase datasets"""
        datasets = self.get_phase_datasets(phase)
        return {ds.name: ds.weight for ds in datasets}
    
    def save_config(self, path: str):
        """Save configuration to JSON"""
        config = {
            "datasets": {name: {
                "name": ds.name,
                "path": ds.path, 
                "format": ds.format,
                "size": ds.size,
                "weight": ds.weight,
                "phase": ds.phase,
                "difficulty": ds.difficulty,
                "features": ds.features
            } for name, ds in self.datasets.items()},
            "phases": {str(pid): {
                "phase_id": phase.phase_id,
                "name": phase.name,
                "epochs": phase.epochs,
                "datasets": phase.datasets,
                "target_metrics": phase.target_metrics,
                "description": phase.description
            } for pid, phase in self.phases.items()}
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self, path: str):
        """Load configuration from JSON"""
        with open(path, 'r') as f:
            config = json.load(f)
        
        # Load datasets
        for name, ds_config in config["datasets"].items():
            self.datasets[name] = DatasetConfig(**ds_config)
        
        # Load phases
        for pid, phase_config in config["phases"].items():
            self.phases[int(pid)] = PhaseConfig(**phase_config)
    
    def print_summary(self):
        """Print configuration summary"""
        print("🚀 HRM SOTA Challenge - Multi-Dataset Configuration")
        print("=" * 60)
        
        for phase_id, phase in self.phases.items():
            print(f"\n📋 Phase {phase_id}: {phase.name} (Epochs {phase.epochs[0]}-{phase.epochs[1]})")
            print(f"   Description: {phase.description}")
            print(f"   Target Metrics: {phase.target_metrics}")
            
            datasets = self.get_phase_datasets(phase_id)
            total_size = sum(ds.size for ds in datasets)
            print(f"   Datasets ({len(datasets)}, {total_size:,} total examples):")
            
            for ds in datasets:
                print(f"     • {ds.name}: {ds.size:,} examples (weight: {ds.weight}, difficulty: {ds.difficulty})")
                print(f"       Features: {', '.join(ds.features)}")

if __name__ == "__main__":
    # Test the configuration
    manager = MultiDatasetManager()
    manager.print_summary()
    manager.save_config("config/multi_dataset_config.json")
    print(f"\n✅ Configuration saved to config/multi_dataset_config.json")