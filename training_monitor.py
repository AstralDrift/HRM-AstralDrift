#!/usr/bin/env python3
"""
HRM Training Monitor & Dashboard

Monitor HRM code generation training with real-time metrics and analysis.
Provides insights into:
- Training convergence
- SWE-Search performance
- Reverse Learning effectiveness  
- Multi-agent coordination
- Performance optimization recommendations
"""

import argparse
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import subprocess
import sys


class HRMTrainingMonitor:
    """Monitor and analyze HRM training progress"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.training_history = []
        self.latest_checkpoint = None
        
        # Setup plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        print(f"🔍 HRM Training Monitor")
        print(f"   Checkpoint directory: {self.checkpoint_dir}")
    
    def load_training_history(self) -> bool:
        """Load training history from checkpoints"""
        
        if not self.checkpoint_dir.exists():
            print(f"❌ Checkpoint directory not found: {self.checkpoint_dir}")
            return False
        
        # Find latest checkpoint
        checkpoint_files = list(self.checkpoint_dir.glob("*.pt"))
        if not checkpoint_files:
            print(f"❌ No checkpoints found in {self.checkpoint_dir}")
            return False
        
        # Load the best model for analysis
        best_checkpoint = self.checkpoint_dir / "best_model.pt"
        if best_checkpoint.exists():
            self.latest_checkpoint = torch.load(best_checkpoint, map_location='cpu')
            self.training_history = self.latest_checkpoint.get('training_history', [])
            print(f"✅ Loaded training history: {len(self.training_history)} epochs")
            return True
        
        # Fallback to latest epoch
        latest_file = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
        self.latest_checkpoint = torch.load(latest_file, map_location='cpu')
        self.training_history = self.latest_checkpoint.get('training_history', [])
        
        print(f"✅ Loaded training history from {latest_file.name}: {len(self.training_history)} epochs")
        return True
    
    def analyze_convergence(self) -> Dict[str, Any]:
        """Analyze training convergence patterns"""
        
        if not self.training_history:
            return {}
        
        losses = [epoch['loss'] for epoch in self.training_history]
        epochs = list(range(len(losses)))
        
        # Calculate convergence metrics
        if len(losses) > 5:
            recent_losses = losses[-5:]
            early_losses = losses[:5]
            
            recent_avg = np.mean(recent_losses)
            early_avg = np.mean(early_losses)
            improvement = (early_avg - recent_avg) / early_avg * 100
            
            # Loss stability (lower is better)
            stability = np.std(recent_losses) / recent_avg
            
            # Convergence rate
            if len(losses) > 10:
                mid_point = len(losses) // 2
                first_half_avg = np.mean(losses[:mid_point])
                second_half_avg = np.mean(losses[mid_point:])
                convergence_rate = (first_half_avg - second_half_avg) / len(losses)
            else:
                convergence_rate = 0
        else:
            improvement = 0
            stability = 0
            convergence_rate = 0
        
        return {
            'total_epochs': len(losses),
            'current_loss': losses[-1] if losses else 0,
            'best_loss': min(losses) if losses else 0,
            'improvement_percent': improvement,
            'loss_stability': stability,
            'convergence_rate': convergence_rate,
            'is_converging': improvement > 5 and stability < 0.1
        }
    
    def analyze_swe_search_performance(self) -> Dict[str, Any]:
        """Analyze SWE-Search framework performance"""
        
        swe_metrics = {}
        
        for epoch in self.training_history:
            for key in ['swe_search_score', 'swe_search_iterations', 'swe_search_candidates', 
                       'swe_search_convergence_rate', 'swe_search_efficiency']:
                if key in epoch:
                    if key not in swe_metrics:
                        swe_metrics[key] = []
                    swe_metrics[key].append(epoch[key])
        
        if not swe_metrics:
            return {}
        
        analysis = {}
        for key, values in swe_metrics.items():
            analysis[key] = {
                'current': values[-1] if values else 0,
                'average': np.mean(values),
                'trend': np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0,
                'best': max(values) if 'score' in key or 'efficiency' in key else min(values)
            }
        
        return analysis
    
    def analyze_reverse_learning_performance(self) -> Dict[str, Any]:
        """Analyze Reverse Learning effectiveness"""
        
        reverse_metrics = {}
        
        for epoch in self.training_history:
            for key in ['reverse_insight_strength', 'reverse_integration_gate', 
                       'reverse_feedback_magnitude', 'reverse_planning_refinement']:
                if key in epoch:
                    if key not in reverse_metrics:
                        reverse_metrics[key] = []
                    reverse_metrics[key].append(epoch[key])
        
        if not reverse_metrics:
            return {}
        
        analysis = {}
        for key, values in reverse_metrics.items():
            analysis[key] = {
                'current': values[-1] if values else 0,
                'average': np.mean(values),
                'trend': np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0,
                'stability': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
            }
        
        return analysis
    
    def create_training_dashboard(self, save_path: Optional[str] = None):
        """Create comprehensive training dashboard"""
        
        if not self.training_history:
            print("❌ No training history available")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('HRM Code Generation Training Dashboard', fontsize=16, fontweight='bold')
        
        epochs = list(range(len(self.training_history)))
        
        # 1. Loss Convergence
        losses = [epoch['loss'] for epoch in self.training_history]
        axes[0, 0].plot(epochs, losses, 'b-', linewidth=2, alpha=0.8)
        axes[0, 0].set_title('Training Loss Convergence', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add trend line
        if len(epochs) > 1:
            z = np.polyfit(epochs, losses, 1)
            p = np.poly1d(z)
            axes[0, 0].plot(epochs, p(epochs), "r--", alpha=0.8, linewidth=1, label=f'Trend: {z[0]:.4f}')
            axes[0, 0].legend()
        
        # 2. SWE-Search Performance
        swe_scores = [epoch.get('swe_search_score', 0) for epoch in self.training_history]
        swe_efficiency = [epoch.get('swe_search_efficiency', 0) for epoch in self.training_history]
        
        ax2 = axes[0, 1]
        ax2.plot(epochs, swe_scores, 'g-', linewidth=2, label='Search Score', alpha=0.8)
        ax2.set_ylabel('Search Score', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(epochs, swe_efficiency, 'orange', linewidth=2, label='Efficiency', alpha=0.8)
        ax2_twin.set_ylabel('Efficiency', color='orange')
        ax2_twin.tick_params(axis='y', labelcolor='orange')
        
        ax2.set_title('SWE-Search Performance', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.grid(True, alpha=0.3)
        
        # 3. Reverse Learning Insights
        insight_strength = [epoch.get('reverse_insight_strength', 0) for epoch in self.training_history]
        integration_gate = [epoch.get('reverse_integration_gate', 0) for epoch in self.training_history]
        
        axes[0, 2].plot(epochs, insight_strength, 'purple', linewidth=2, alpha=0.8, label='Insight Strength')
        axes[0, 2].plot(epochs, integration_gate, 'brown', linewidth=2, alpha=0.8, label='Integration Gate')
        axes[0, 2].set_title('Reverse Learning Performance', fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Value')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Batch Time Performance
        batch_times = [epoch.get('avg_batch_time', 0) for epoch in self.training_history]
        axes[1, 0].plot(epochs, batch_times, 'red', linewidth=2, alpha=0.8)
        axes[1, 0].set_title('Training Speed', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Avg Batch Time (s)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Gradient Norms (if available)
        if hasattr(self, 'gradient_norms'):  # Would need to be logged
            axes[1, 1].plot(epochs, self.gradient_norms, 'cyan', linewidth=2, alpha=0.8)
            axes[1, 1].set_title('Gradient Norms', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Show loss components instead
            lm_losses = [epoch.get('lm_loss', 0) for epoch in self.training_history]
            swe_losses = [epoch.get('swe_search_loss', 0) for epoch in self.training_history]
            reverse_losses = [epoch.get('reverse_learning_loss', 0) for epoch in self.training_history]
            
            axes[1, 1].plot(epochs, lm_losses, 'blue', linewidth=2, alpha=0.7, label='LM Loss')
            axes[1, 1].plot(epochs, swe_losses, 'green', linewidth=2, alpha=0.7, label='SWE Loss')
            axes[1, 1].plot(epochs, reverse_losses, 'purple', linewidth=2, alpha=0.7, label='Reverse Loss')
            axes[1, 1].set_title('Loss Components', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss Value')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Performance Summary
        convergence_analysis = self.analyze_convergence()
        
        axes[1, 2].axis('off')
        summary_text = f"""
Performance Summary

Current Loss: {convergence_analysis.get('current_loss', 0):.4f}
Best Loss: {convergence_analysis.get('best_loss', 0):.4f}
Improvement: {convergence_analysis.get('improvement_percent', 0):.1f}%

SWE-Search Score: {swe_scores[-1] if swe_scores else 0:.3f}
Search Efficiency: {swe_efficiency[-1] if swe_efficiency else 0:.3f}

Insight Strength: {insight_strength[-1] if insight_strength else 0:.2f}
Integration Gate: {integration_gate[-1] if integration_gate else 0:.3f}

Avg Batch Time: {batch_times[-1] if batch_times else 0:.2f}s
Epochs Trained: {len(self.training_history)}

Status: {"🟢 Converging" if convergence_analysis.get('is_converging', False) else "🟡 Training"}
        """
        
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Dashboard saved to: {save_path}")
        
        plt.show()
    
    def generate_training_report(self) -> str:
        """Generate comprehensive training report"""
        
        if not self.training_history:
            return "No training history available"
        
        convergence = self.analyze_convergence()
        swe_analysis = self.analyze_swe_search_performance()
        reverse_analysis = self.analyze_reverse_learning_performance()
        
        report = f"""
🚀 HRM Code Generation Training Report
{'='*60}

📊 TRAINING OVERVIEW
   Total Epochs: {convergence['total_epochs']}
   Current Loss: {convergence['current_loss']:.4f}
   Best Loss: {convergence['best_loss']:.4f}
   Improvement: {convergence['improvement_percent']:.1f}%
   Status: {"🟢 Converging" if convergence['is_converging'] else "🟡 Training"}

🔍 SWE-SEARCH PERFORMANCE
"""
        
        if swe_analysis:
            for metric, data in swe_analysis.items():
                metric_name = metric.replace('swe_search_', '').replace('_', ' ').title()
                report += f"   {metric_name}: {data['current']:.3f} (avg: {data['average']:.3f}, trend: {data['trend']:+.4f})\n"
        else:
            report += "   No SWE-Search metrics available\n"
        
        report += "\n🔄 REVERSE LEARNING PERFORMANCE\n"
        
        if reverse_analysis:
            for metric, data in reverse_analysis.items():
                metric_name = metric.replace('reverse_', '').replace('_', ' ').title()
                report += f"   {metric_name}: {data['current']:.3f} (avg: {data['average']:.3f}, stability: {data['stability']:.3f})\n"
        else:
            report += "   No Reverse Learning metrics available\n"
        
        # Performance recommendations
        report += "\n💡 RECOMMENDATIONS\n"
        
        if convergence['improvement_percent'] < 1:
            report += "   • Consider increasing learning rate or adjusting architecture\n"
        
        if convergence['loss_stability'] > 0.2:
            report += "   • Training appears unstable, consider reducing learning rate\n"
        
        if swe_analysis and swe_analysis.get('swe_search_efficiency', {}).get('current', 0) < 0.3:
            report += "   • SWE-Search efficiency is low, consider tuning search parameters\n"
        
        if reverse_analysis and reverse_analysis.get('reverse_integration_gate', {}).get('current', 0) > 0.8:
            report += "   • Integration gate is high, reverse learning may be over-correcting\n"
        
        avg_batch_time = np.mean([epoch.get('avg_batch_time', 0) for epoch in self.training_history])
        if avg_batch_time > 3.0:
            report += f"   • Batch time is high ({avg_batch_time:.1f}s), consider optimizing model size\n"
        
        report += f"\n📈 Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return report
    
    def monitor_live_training(self, refresh_interval: int = 30):
        """Monitor training in real-time"""
        
        print(f"🔴 Live Training Monitor (refresh every {refresh_interval}s)")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                # Reload training history
                if self.load_training_history():
                    # Clear screen
                    subprocess.call('clear' if sys.platform != 'win32' else 'cls', shell=True)
                    
                    # Print current status
                    print(self.generate_training_report())
                    
                    # Check if training is still active
                    if self.latest_checkpoint:
                        last_epoch = self.latest_checkpoint.get('epoch', 0)
                        print(f"\n🕐 Last update: Epoch {last_epoch}")
                
                time.sleep(refresh_interval)
        
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped")


def main():
    parser = argparse.ArgumentParser(description="HRM Training Monitor")
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoint directory to monitor')
    parser.add_argument('--live', action='store_true', help='Enable live monitoring')
    parser.add_argument('--dashboard', action='store_true', help='Show training dashboard')
    parser.add_argument('--report', action='store_true', help='Generate training report')
    parser.add_argument('--save-dashboard', help='Save dashboard to file')
    
    args = parser.parse_args()
    
    # Find the most recent checkpoint directory if not specified
    if args.checkpoint_dir == 'checkpoints':
        checkpoint_path = Path('checkpoints')
        if checkpoint_path.exists():
            subdirs = [d for d in checkpoint_path.iterdir() if d.is_dir()]
            if subdirs:
                # Get most recent directory
                args.checkpoint_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
    
    monitor = HRMTrainingMonitor(args.checkpoint_dir)
    
    if not monitor.load_training_history():
        return 1
    
    if args.live:
        monitor.monitor_live_training()
    elif args.dashboard:
        monitor.create_training_dashboard(args.save_dashboard)
    elif args.report:
        print(monitor.generate_training_report())
    else:
        # Default: show brief report and dashboard
        print(monitor.generate_training_report())
        monitor.create_training_dashboard(args.save_dashboard)
    
    return 0


if __name__ == "__main__":
    exit(main())