#!/usr/bin/env python3
"""
Enhanced Training Monitor for HRM Code Generation
Monitors training progress with focus on code-specific metrics
"""

import time
import os
import re
from pathlib import Path

def extract_latest_metrics(log_file="hrm_training.log"):
    """Extract the most recent metrics from training logs"""
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Find the most recent epoch results
    latest_metrics = {}
    current_epoch = None
    
    # Look for the last epoch results
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        
        # Find epoch header
        if "📊 Epoch" in line and "Results:" in line:
            current_epoch = re.search(r'Epoch (\d+)', line)
            if current_epoch:
                current_epoch = int(current_epoch.group(1))
                break
    
    if current_epoch is None:
        return None
    
    # Extract metrics from that epoch
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        
        if f"📊 Epoch {current_epoch} Results:" in line:
            # Extract metrics from subsequent lines
            for j in range(i + 1, min(i + 50, len(lines))):
                metric_line = lines[j].strip()
                
                if not metric_line.startswith("   "):
                    break
                
                # Parse metric
                if ":" in metric_line:
                    key, value = metric_line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    try:
                        latest_metrics[key] = float(value)
                    except:
                        latest_metrics[key] = value
            break
    
    return {"epoch": current_epoch, "metrics": latest_metrics}

def display_metrics_dashboard(metrics_data):
    """Display a comprehensive metrics dashboard"""
    if not metrics_data:
        print("❌ No metrics data available")
        return
    
    epoch = metrics_data["epoch"]
    metrics = metrics_data["metrics"]
    
    print(f"\n{'='*80}")
    print(f"🚀 HRM Code Generation Training Dashboard - Epoch {epoch}")
    print(f"{'='*80}")
    
    # Core Training Metrics
    print(f"\n📈 Core Training Metrics:")
    print(f"   Loss: {metrics.get('Loss', 'N/A')}")
    print(f"   Avg Batch Time: {metrics.get('Avg batch time', 'N/A')}s")
    print(f"   Total Batches: {metrics.get('Total batches', 'N/A')}")
    print(f"   Epoch Time: {metrics.get('Epoch time', 'N/A')}s")
    
    # Progressive Metrics (work without halting)
    print(f"\n🎯 Progressive Accuracy Metrics:")
    print(f"   Token Accuracy (All): {metrics.get('token_accuracy_all', 'N/A'):.4f}")
    print(f"   Prefix Match Rate: {metrics.get('prefix_match_rate', 'N/A'):.4f}")
    print(f"   Partial Sequence Acc: {metrics.get('partial_sequence_accuracy', 'N/A'):.4f}")
    
    # Halting Behavior
    print(f"\n⏱️  ACT Halting Metrics:")
    print(f"   Halted Samples: {metrics.get('halted_samples', 'N/A')}")
    print(f"   Total Samples: {metrics.get('total_samples', 'N/A')}")
    halted = metrics.get('halted_samples', 0)
    total = metrics.get('total_samples', 1)
    halt_rate = (halted / total * 100) if total > 0 else 0
    print(f"   Halting Rate: {halt_rate:.1f}%")
    print(f"   Avg Steps (All): {metrics.get('avg_steps_all', 'N/A'):.2f}")
    
    # Code-Specific Enhanced Metrics
    print(f"\n💻 Code-Specific Metrics:")
    print(f"   Syntax Validity: {metrics.get('code_syntax_validity', 'N/A'):.3f}")
    print(f"   Compilation Success: {metrics.get('code_compilation_success', 'N/A'):.3f}")
    print(f"   Edit Distance: {metrics.get('code_edit_distance', 'N/A'):.3f}")
    print(f"   Syntax Accuracy: {metrics.get('code_syntax_accuracy', 'N/A'):.3f}")
    print(f"   Logical Accuracy: {metrics.get('code_logical_accuracy', 'N/A'):.3f}")
    print(f"   Exact Match: {metrics.get('code_exact_match', 'N/A'):.3f}")
    print(f"   🏆 Tiered Accuracy: {metrics.get('code_tiered_accuracy', 'N/A'):.3f}")
    
    # SWE-Search Integration
    print(f"\n🔍 SWE-Search Metrics:")
    print(f"   Search Score: {metrics.get('swe_search_score', 'N/A'):.3f}")
    print(f"   Search Iterations: {metrics.get('swe_search_iterations', 'N/A'):.1f}")
    print(f"   Convergence Rate: {metrics.get('swe_search_convergence_rate', 'N/A'):.3f}")
    print(f"   Search Efficiency: {metrics.get('swe_search_efficiency', 'N/A'):.3f}")
    
    # Reverse Learning
    print(f"\n🔄 Reverse Learning Metrics:")
    print(f"   Insight Strength: {metrics.get('reverse_insight_strength', 'N/A'):.3f}")
    print(f"   Integration Gate: {metrics.get('reverse_integration_gate', 'N/A'):.3f}")
    print(f"   Feedback Magnitude: {metrics.get('reverse_feedback_magnitude', 'N/A'):.3f}")
    print(f"   Planning Refinement: {metrics.get('reverse_planning_refinement', 'N/A'):.3f}")
    
    # Loss Components
    print(f"\n💰 Loss Components:")
    print(f"   LM Loss: {metrics.get('lm_loss', 'N/A'):.2f}")
    print(f"   Q Halt Loss: {metrics.get('q_halt_loss', 'N/A'):.2f}")
    print(f"   Q Continue Loss: {metrics.get('q_continue_loss', 'N/A'):.2f}")
    print(f"   SWE Search Loss: {metrics.get('swe_search_loss', 'N/A'):.2f}")
    print(f"   Reverse Learning Loss: {metrics.get('reverse_learning_loss', 'N/A'):.2f}")

def main():
    """Main monitoring loop"""
    print("🚀 Starting HRM Code Generation Training Monitor")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            # Clear screen for better readability
            os.system('clear' if os.name == 'posix' else 'cls')
            
            # Extract and display metrics
            metrics_data = extract_latest_metrics()
            display_metrics_dashboard(metrics_data)
            
            # Wait before next update
            print(f"\n🔄 Refreshing in 30 seconds... (Press Ctrl+C to stop)")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print(f"\n\n✅ Monitoring stopped by user")

if __name__ == "__main__":
    main()