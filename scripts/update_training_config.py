#!/usr/bin/env python3
"""
Update training configuration for optimized run
"""

import json
import yaml
from pathlib import Path

def create_optimized_config():
    """Create optimized configuration for extended training"""
    config = {
        # Core training parameters
        "epochs": 100,  # Extend to 100 epochs
        "learning_rate": 2e-5,  # Slightly reduce LR for fine convergence
        "batch_size": 8,  # Increase if GPU memory allows
        
        # ACT parameters - optimize for better halting
        "halt_exploration_prob": 0.5,  # Increase exploration
        "halt_max_steps": 8,  # Allow more computation steps
        "act_threshold": 0.95,  # Slightly lower threshold
        
        # SWE-Search parameters - boost performance
        "swe_candidates": 15,  # Increase from 10
        "swe_search_weight": 0.3,  # Increase weight
        "swe_temperature": 0.8,  # Add temperature for diversity
        
        # Reverse Learning - fine-tune
        "reverse_learning_weight": 0.15,  # Slight increase
        "reverse_momentum": 0.95,  # Add momentum term
        
        # Code-specific parameters
        "code_syntax_weight": 0.4,  # Increase focus on syntax
        "compilation_weight": 0.3,  # Add compilation loss weight
        
        # Scheduler updates
        "scheduler_t0": 100,  # Extend for 100 epochs
        "min_lr": 1e-6,  # Lower minimum LR
        
        # Evaluation frequency
        "eval_interval": 500,  # More frequent evaluation
        "checkpoint_interval": 5,  # Save every 5 epochs
        
        # Dataset
        "use_real_livecodebench": True,
        "data_augmentation": True,
        "augmentation_factor": 2
    }
    
    return config

def update_train_script():
    """Generate commands to update training script"""
    commands = []
    
    # Update hyperparameters
    commands.append("""
# Update train_hrm_optimized.py hyperparameters
sed -i '' 's/halt_exploration_prob=0.4/halt_exploration_prob=0.5/g' train_hrm_optimized.py
sed -i '' 's/halt_max_steps=6/halt_max_steps=8/g' train_hrm_optimized.py
sed -i '' 's/swe_search_weight=0.2/swe_search_weight=0.3/g' train_hrm_optimized.py
sed -i '' 's/reverse_learning_weight=0.1/reverse_learning_weight=0.15/g' train_hrm_optimized.py
""")
    
    # Add new parameters
    commands.append("""
# Add new parameters to config dict
echo "Adding optimized parameters..."
""")
    
    return commands

def main():
    print("🚀 Creating optimized configuration...")
    
    config = create_optimized_config()
    
    # Save config
    with open("config/optimized_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Configuration saved to config/optimized_config.json")
    
    # Generate update commands
    commands = update_train_script()
    
    with open("scripts/apply_optimizations.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Apply optimization updates\n\n")
        for cmd in commands:
            f.write(cmd + "\n")
    
    print("📝 Update commands saved to scripts/apply_optimizations.sh")
    print("\nKey optimizations:")
    print("- SWE candidates: 10 → 15")
    print("- Exploration prob: 0.4 → 0.5")
    print("- Halt max steps: 6 → 8")
    print("- SWE weight: 0.2 → 0.3")
    print("- Reverse learning: 0.1 → 0.15")

if __name__ == "__main__":
    main()