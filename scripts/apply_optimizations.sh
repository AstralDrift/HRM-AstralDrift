#!/bin/bash
# Apply optimization updates


# Update train_hrm_optimized.py hyperparameters
sed -i '' 's/halt_exploration_prob=0.4/halt_exploration_prob=0.5/g' train_hrm_optimized.py
sed -i '' 's/halt_max_steps=6/halt_max_steps=8/g' train_hrm_optimized.py
sed -i '' 's/swe_search_weight=0.2/swe_search_weight=0.3/g' train_hrm_optimized.py
sed -i '' 's/reverse_learning_weight=0.1/reverse_learning_weight=0.15/g' train_hrm_optimized.py


# Add new parameters to config dict
echo "Adding optimized parameters..."

