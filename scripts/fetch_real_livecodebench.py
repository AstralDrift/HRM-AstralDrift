#!/usr/bin/env python3
"""
Fetch and integrate real LiveCodeBench dataset for HRM training
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
from typing import Dict, List, Any
import argparse
from datetime import datetime

class LiveCodeBenchIntegrator:
    def __init__(self, output_dir: str = "data/livecodebench_real"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_dataset(self, version_tag: str = "release_v2", use_lite: bool = True):
        """Fetch LiveCodeBench dataset from HuggingFace"""
        dataset_name = "livecodebench/code_generation_lite" if use_lite else "livecodebench/code_generation"
        
        print(f"📦 Fetching {dataset_name} (version: {version_tag})")
        print("   This may take several minutes for initial download...")
        
        try:
            # Load dataset with specific version
            dataset = load_dataset(dataset_name, version_tag=version_tag)
            
            print(f"✅ Dataset loaded successfully")
            print(f"   Available splits: {list(dataset.keys())}")
            
            # Get test split (main evaluation data)
            test_data = dataset["test"]
            print(f"   Test problems: {len(test_data)}")
            
            return test_data
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def analyze_dataset_structure(self, dataset) -> Dict:
        """Analyze the structure and content of the dataset"""
        if not dataset:
            return {"status": "error", "message": "No dataset provided"}
        
        print("🔍 Analyzing dataset structure...")
        
        # Get sample problem
        sample = dataset[0]
        
        # Analyze fields
        fields = list(sample.keys())
        
        # Collect statistics
        stats = {
            "total_problems": len(dataset),
            "fields": fields,
            "sample_problem": sample,
            "platforms": {},
            "difficulties": {},
            "contest_dates": {},
            "avg_public_tests": 0,
            "avg_private_tests": 0
        }
        
        # Analyze all problems for statistics
        total_public = 0
        total_private = 0
        
        for i, problem in enumerate(dataset):
            # Platform distribution
            platform = problem.get("platform", "unknown")
            stats["platforms"][platform] = stats["platforms"].get(platform, 0) + 1
            
            # Difficulty distribution
            difficulty = problem.get("difficulty", "unknown")
            stats["difficulties"][difficulty] = stats["difficulties"].get(difficulty, 0) + 1
            
            # Contest dates
            contest_date = problem.get("contest_date", "unknown")[:7]  # YYYY-MM
            stats["contest_dates"][contest_date] = stats["contest_dates"].get(contest_date, 0) + 1
            
            # Test case counts
            if "public_test_cases" in problem:
                total_public += len(problem["public_test_cases"])
            if "private_test_cases" in problem:
                total_private += len(problem["private_test_cases"])
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"   Analyzed {i + 1}/{len(dataset)} problems...")
        
        stats["avg_public_tests"] = total_public / len(dataset) if len(dataset) > 0 else 0
        stats["avg_private_tests"] = total_private / len(dataset) if len(dataset) > 0 else 0
        
        return {"status": "success", "stats": stats}
    
    def convert_to_hrm_format(self, dataset, analysis: Dict) -> List[Dict]:
        """Convert LiveCodeBench format to HRM training format"""
        print("🔄 Converting to HRM format...")
        
        hrm_problems = []
        
        for i, problem in enumerate(dataset):
            # Create HRM-compatible problem entry
            hrm_problem = {
                # Core identification
                "id": f"lcb_{problem.get('question_id', i)}",
                "title": problem.get("question_title", f"Problem {i}"),
                "platform": problem.get("platform", "unknown"),
                "difficulty": problem.get("difficulty", "medium"),
                "contest_date": problem.get("contest_date", "unknown"),
                
                # Problem content
                "prompt": problem.get("question_content", ""),
                "starter_code": problem.get("starter_code", ""),
                
                # Test cases
                "public_tests": problem.get("public_test_cases", []),
                "private_tests": problem.get("private_test_cases", []),
                
                # HRM-specific additions
                "task_type": "code_generation",
                "language": "python",  # Default - could be inferred from starter_code
                "complexity_level": self._map_difficulty_to_complexity(problem.get("difficulty", "medium")),
                
                # Multi-scenario support
                "scenarios": {
                    "code_generation": {
                        "input_text": self._create_code_gen_prompt(problem),
                        "target_text": problem.get("starter_code", "# TODO: Implement solution")
                    },
                    "self_repair": {
                        "input_text": self._create_repair_prompt(problem),
                        "target_text": "# Fixed version would go here"
                    },
                    "test_prediction": {
                        "input_text": self._create_test_pred_prompt(problem),
                        "target_text": "# Expected output would go here"
                    }
                },
                
                # Metadata for training
                "metadata": {
                    "contest_id": problem.get("contest_id", ""),
                    "original_index": i,
                    "test_case_count": len(problem.get("public_test_cases", [])) + len(problem.get("private_test_cases", [])),
                    "conversion_timestamp": datetime.now().isoformat()
                }
            }
            
            hrm_problems.append(hrm_problem)
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"   Converted {i + 1}/{len(dataset)} problems...")
        
        return hrm_problems
    
    def _map_difficulty_to_complexity(self, difficulty: str) -> int:
        """Map difficulty string to numerical complexity level"""
        mapping = {
            "easy": 1,
            "medium": 2, 
            "hard": 3,
            "unknown": 2
        }
        return mapping.get(difficulty.lower(), 2)
    
    def _create_code_gen_prompt(self, problem: Dict) -> str:
        """Create code generation prompt from problem"""
        prompt_parts = []
        
        if problem.get("question_title"):
            prompt_parts.append(f"Problem: {problem['question_title']}")
        
        if problem.get("question_content"):
            prompt_parts.append(f"Description: {problem['question_content']}")
        
        # Add public test cases as examples
        if problem.get("public_test_cases"):
            prompt_parts.append("Examples:")
            for i, test in enumerate(problem["public_test_cases"][:3]):  # Limit to first 3
                prompt_parts.append(f"Input: {test.get('input', '')}")
                prompt_parts.append(f"Output: {test.get('output', '')}")
                if i < len(problem["public_test_cases"][:3]) - 1:
                    prompt_parts.append("")
        
        prompt_parts.append("Generate a Python solution:")
        
        return "\n".join(prompt_parts)
    
    def _create_repair_prompt(self, problem: Dict) -> str:
        """Create self-repair prompt from problem"""
        return f"Fix the following code for problem '{problem.get('question_title', 'Unknown')}':\n\n{problem.get('starter_code', '# Code to fix')}"
    
    def _create_test_pred_prompt(self, problem: Dict) -> str:
        """Create test prediction prompt from problem"""
        return f"Predict the output for problem '{problem.get('question_title', 'Unknown')}' with the given input."
    
    def save_dataset(self, hrm_problems: List[Dict], analysis: Dict, version_tag: str):
        """Save converted dataset and analysis"""
        # Save main dataset
        dataset_file = self.output_dir / "livecodebench_real.json"
        with open(dataset_file, 'w') as f:
            json.dump(hrm_problems, f, indent=2)
        
        print(f"💾 Dataset saved to: {dataset_file}")
        print(f"   Total problems: {len(hrm_problems)}")
        
        # Save analysis
        analysis_file = self.output_dir / "dataset_analysis.json"
        analysis_data = {
            "version_tag": version_tag,
            "conversion_timestamp": datetime.now().isoformat(),
            "hrm_problem_count": len(hrm_problems),
            "original_analysis": analysis
        }
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"📊 Analysis saved to: {analysis_file}")
        
        # Create sample for inspection
        sample_file = self.output_dir / "sample_problems.json"
        sample_problems = hrm_problems[:5]  # First 5 problems
        
        with open(sample_file, 'w') as f:
            json.dump(sample_problems, f, indent=2)
        
        print(f"🔍 Sample problems saved to: {sample_file}")
        
        return {
            "dataset_file": str(dataset_file),
            "analysis_file": str(analysis_file),
            "sample_file": str(sample_file),
            "problem_count": len(hrm_problems)
        }
    
    def create_training_config(self, problem_count: int):
        """Create training configuration for HRM with LiveCodeBench"""
        config = {
            "dataset_info": {
                "name": "LiveCodeBench Real",
                "path": str(self.output_dir / "livecodebench_real.json"),
                "problem_count": problem_count,
                "task_type": "code_generation",
                "format": "hrm_compatible"
            },
            "training_params": {
                "batch_size": 4,  # Smaller for code problems
                "learning_rate": 3e-5,
                "max_seq_len": 1024,  # Longer for code
                "warmup_steps": 500,
                "eval_steps": 100,
                "save_steps": 500
            },
            "model_adaptations": {
                "enable_code_tokenization": True,
                "syntax_aware_training": True,
                "multi_scenario_training": True,
                "difficulty_aware_sampling": True
            },
            "evaluation_config": {
                "test_split_ratio": 0.1,
                "eval_scenarios": ["code_generation", "self_repair", "test_prediction"],
                "metrics": ["pass@1", "pass@5", "syntax_validity", "compilation_success"]
            }
        }
        
        config_file = self.output_dir / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"⚙️ Training config saved to: {config_file}")
        return config_file
    
    def run_full_integration(self, version_tag: str = "release_v2", use_lite: bool = True):
        """Run complete LiveCodeBench integration process"""
        print("🚀 Starting LiveCodeBench Integration")
        print("=" * 50)
        
        # Step 1: Fetch dataset
        dataset = self.fetch_dataset(version_tag, use_lite)
        if not dataset:
            return {"status": "failed", "step": "fetch"}
        
        # Step 2: Analyze structure
        analysis = self.analyze_dataset_structure(dataset)
        if analysis["status"] != "success":
            return {"status": "failed", "step": "analysis", "error": analysis}
        
        # Print analysis summary
        stats = analysis["stats"]
        print(f"\n📊 Dataset Analysis Summary:")
        print(f"   Total problems: {stats['total_problems']}")
        print(f"   Platforms: {dict(stats['platforms'])}")
        print(f"   Difficulties: {dict(stats['difficulties'])}")
        print(f"   Avg public tests: {stats['avg_public_tests']:.1f}")
        print(f"   Avg private tests: {stats['avg_private_tests']:.1f}")
        
        # Step 3: Convert to HRM format
        hrm_problems = self.convert_to_hrm_format(dataset, analysis)
        
        # Step 4: Save everything
        save_result = self.save_dataset(hrm_problems, analysis, version_tag)
        
        # Step 5: Create training config
        config_file = self.create_training_config(len(hrm_problems))
        
        print(f"\n✅ LiveCodeBench Integration Complete!")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🔢 Problems converted: {len(hrm_problems)}")
        
        return {
            "status": "success",
            "output_dir": str(self.output_dir),
            "files": save_result,
            "config_file": str(config_file),
            "problem_count": len(hrm_problems),
            "analysis": stats
        }

def main():
    """Main integration function"""
    parser = argparse.ArgumentParser(description="Fetch and integrate LiveCodeBench dataset")
    parser.add_argument("--version", default="release_v2", 
                       choices=["release_v1", "release_v2", "release_v3", "release_v4", "release_v5"],
                       help="Dataset version to fetch")
    parser.add_argument("--output-dir", default="data/livecodebench_real", 
                       help="Output directory for processed dataset")
    parser.add_argument("--use-lite", action="store_true", default=True,
                       help="Use lite version of dataset (faster, smaller)")
    parser.add_argument("--no-lite", action="store_false", dest="use_lite",
                       help="Use full version of dataset")
    
    args = parser.parse_args()
    
    integrator = LiveCodeBenchIntegrator(args.output_dir)
    result = integrator.run_full_integration(args.version, args.use_lite)
    
    if result["status"] == "success":
        print("\n🎉 Integration successful! Ready for HRM training.")
        print(f"📝 Next steps:")
        print(f"   1. Review sample problems: {result['files']['sample_file']}")
        print(f"   2. Update training script to use: {result['files']['dataset_file']}")
        print(f"   3. Apply training config: {result['config_file']}")
    else:
        print(f"\n❌ Integration failed at step: {result['step']}")
        if "error" in result:
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()