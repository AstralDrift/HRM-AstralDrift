#!/usr/bin/env python3
"""
LiveCodeBench Integration for HRM Code Generation
Prepares LiveCodeBench dataset for HRM training and evaluation
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

def download_livecodebench():
    """Download and prepare LiveCodeBench dataset"""
    commands = [
        "cd data",
        "git clone https://github.com/LiveCodeBench/LiveCodeBench.git",
        "cd LiveCodeBench",
        "pip install -e .",
        "python -m livecodebench.data.download"
    ]
    
    for cmd in commands:
        print(f"🔄 Running: {cmd}")
        os.system(cmd)

def convert_to_hrm_format(livecodebench_path: str, output_path: str):
    """Convert LiveCodeBench format to HRM training format"""
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Process problems
    problems_file = os.path.join(livecodebench_path, "problems.json")
    if not os.path.exists(problems_file):
        print(f"❌ Problems file not found: {problems_file}")
        return
    
    with open(problems_file, 'r') as f:
        problems = json.load(f)
    
    hrm_data = []
    for problem_id, problem_data in problems.items():
        # Convert to HRM format
        hrm_instance = {
            "instance_id": problem_id,
            "domain": "code_generation",
            "language": problem_data.get("language", "python"),
            "complexity": problem_data.get("difficulty", 0.5),
            "input_text": f"Problem: {problem_data.get('description', '')}\n\nConstraints: {problem_data.get('constraints', '')}",
            "target_text": problem_data.get("solution", ""),
            "metadata": {
                "source": "livecodebench",
                "difficulty": problem_data.get("difficulty"),
                "contest": problem_data.get("contest"),
                "tags": problem_data.get("tags", [])
            }
        }
        hrm_data.append(hrm_instance)
    
    # Save HRM format
    output_file = os.path.join(output_path, "livecodebench_hrm.json")
    with open(output_file, 'w') as f:
        json.dump(hrm_data, f, indent=2)
    
    print(f"✅ Converted {len(hrm_data)} problems to HRM format: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Prepare LiveCodeBench for HRM")
    parser.add_argument("--download", action="store_true", help="Download LiveCodeBench")
    parser.add_argument("--convert", action="store_true", help="Convert to HRM format")
    parser.add_argument("--input-path", default="data/LiveCodeBench", help="LiveCodeBench path")
    parser.add_argument("--output-path", default="data/livecodebench-hrm", help="Output path")
    
    args = parser.parse_args()
    
    if args.download:
        download_livecodebench()
    
    if args.convert:
        convert_to_hrm_format(args.input_path, args.output_path)

if __name__ == "__main__":
    main()