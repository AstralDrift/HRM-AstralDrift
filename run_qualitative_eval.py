#!/usr/bin/env python3
"""
Qualitative Evaluation of HRM Code Generation

Test the model on 5 diverse coding scenarios:
1. Simple algorithm (Two Sum)
2. Data structure (Binary Tree)
3. String manipulation 
4. Dynamic programming
5. Tool/CLI workflow
"""

import json
import sys
from pathlib import Path

# Test cases for qualitative evaluation
test_cases = [
    {
        "id": "eval_001",
        "type": "algorithm",
        "description": "Implement a function to find two numbers in an array that sum to a target",
        "input_text": "Problem: Two Sum\nGiven an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.\n\nExample:\nInput: nums = [2,7,11,15], target = 9\nOutput: [0,1]",
        "expected_complexity": 0.3
    },
    {
        "id": "eval_002", 
        "type": "data_structure",
        "description": "Binary tree level order traversal",
        "input_text": "Problem: Binary Tree Level Order Traversal\nGiven the root of a binary tree, return the level order traversal of its nodes' values (left to right, level by level).\n\nExample:\nInput: root = [3,9,20,null,null,15,7]\nOutput: [[3],[9,20],[15,7]]",
        "expected_complexity": 0.6
    },
    {
        "id": "eval_003",
        "type": "string_processing", 
        "description": "Valid parentheses checker",
        "input_text": "Problem: Valid Parentheses\nGiven a string s containing just '(', ')', '{', '}', '[' and ']', determine if the input string is valid.\n\nRules:\n1. Open brackets must be closed by the same type\n2. Open brackets must be closed in correct order\n\nExample: '()[]{}' -> true, '([)]' -> false",
        "expected_complexity": 0.4
    },
    {
        "id": "eval_004",
        "type": "dynamic_programming",
        "description": "Longest common subsequence",
        "input_text": "Problem: Longest Common Subsequence\nGiven two strings text1 and text2, return the length of their longest common subsequence.\n\nExample:\nInput: text1 = 'abcde', text2 = 'ace'\nOutput: 3 (subsequence 'ace')",
        "expected_complexity": 0.8
    },
    {
        "id": "eval_005",
        "type": "tool_workflow",
        "description": "Git workflow automation",
        "input_text": "Problem: Git Workflow\nCreate a script that:\n1. Checks git status\n2. Adds all changed files\n3. Commits with a message\n4. Pushes to remote\n\nHandle errors gracefully and provide status feedback.",
        "expected_complexity": 0.5
    }
]

def save_evaluation_cases():
    """Save evaluation cases to JSON file"""
    eval_dir = Path("data/qualitative_eval")
    eval_dir.mkdir(exist_ok=True)
    
    with open(eval_dir / "eval_cases.json", 'w') as f:
        json.dump(test_cases, f, indent=2)
    
    print(f"✅ Saved {len(test_cases)} evaluation cases to {eval_dir}/eval_cases.json")
    print("\n📋 Evaluation Cases:")
    for case in test_cases:
        print(f"   {case['id']}: {case['type']} - {case['description']}")

if __name__ == "__main__":
    save_evaluation_cases()