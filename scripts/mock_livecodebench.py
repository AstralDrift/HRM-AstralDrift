#!/usr/bin/env python3
"""
Mock LiveCodeBench Dataset for HRM Integration
Creates a mock LiveCodeBench-style dataset for immediate testing
"""

import json
import os
from pathlib import Path

def create_mock_livecodebench():
    """Create mock LiveCodeBench data for testing"""
    
    # Mock problems covering different complexity levels
    mock_problems = [
        {
            "instance_id": "lcb_001",
            "domain": "algorithms",
            "language": "python",
            "complexity": 0.3,
            "input_text": "Problem: Two Sum\nGiven an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.\n\nConstraints: 2 <= nums.length <= 10^4",
            "target_text": "def two_sum(nums, target):\n    seen = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in seen:\n            return [seen[complement], i]\n        seen[num] = i\n    return []",
            "metadata": {
                "source": "mock_livecodebench",
                "difficulty": "easy",
                "contest": "weekly_contest_1",
                "tags": ["array", "hash_table"]
            }
        },
        {
            "instance_id": "lcb_002", 
            "domain": "data_structures",
            "language": "python",
            "complexity": 0.6,
            "input_text": "Problem: Binary Tree Inorder Traversal\nGiven the root of a binary tree, return the inorder traversal of its nodes' values.\n\nConstraints: The number of nodes in the tree is in the range [0, 100]",
            "target_text": "def inorder_traversal(root):\n    def dfs(node, result):\n        if node:\n            dfs(node.left, result)\n            result.append(node.val)\n            dfs(node.right, result)\n    \n    result = []\n    dfs(root, result)\n    return result",
            "metadata": {
                "source": "mock_livecodebench",
                "difficulty": "medium",
                "contest": "weekly_contest_2",
                "tags": ["tree", "depth_first_search", "binary_tree"]
            }
        },
        {
            "instance_id": "lcb_003",
            "domain": "dynamic_programming", 
            "language": "python",
            "complexity": 0.8,
            "input_text": "Problem: Longest Increasing Subsequence\nGiven an integer array nums, return the length of the longest strictly increasing subsequence.\n\nConstraints: 1 <= nums.length <= 2500",
            "target_text": "def length_of_lis(nums):\n    if not nums:\n        return 0\n    \n    dp = [1] * len(nums)\n    \n    for i in range(1, len(nums)):\n        for j in range(i):\n            if nums[i] > nums[j]:\n                dp[i] = max(dp[i], dp[j] + 1)\n    \n    return max(dp)",
            "metadata": {
                "source": "mock_livecodebench",
                "difficulty": "hard",
                "contest": "weekly_contest_3",
                "tags": ["array", "binary_search", "dynamic_programming"]
            }
        }
    ]
    
    # Create output directory
    output_dir = Path("data/mock-livecodebench")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save mock dataset
    output_file = output_dir / "mock_livecodebench.json"
    with open(output_file, 'w') as f:
        json.dump(mock_problems, f, indent=2)
    
    print(f"✅ Created mock LiveCodeBench dataset: {output_file}")
    print(f"   Problems: {len(mock_problems)}")
    print(f"   Languages: python")
    print(f"   Difficulty range: easy → hard")
    
    return str(output_file)

if __name__ == "__main__":
    create_mock_livecodebench()