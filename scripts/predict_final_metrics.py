#!/usr/bin/env python3
"""
Predict final metrics based on current training trajectory
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import math

class MetricsPredictor:
    def __init__(self, convergence_config_path: str = "config/convergence_criteria.json"):
        self.config_path = convergence_config_path
        self.load_config()
        
    def load_config(self):
        """Load convergence criteria configuration"""
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        self.prediction_config = self.config['epoch50_prediction']['prediction_model']
        
    def exponential_decay(self, epoch: int, initial: float, current: float, 
                         current_epoch: int, decay_rate: float, asymptote: float) -> float:
        """Exponential decay model for loss prediction"""
        # Calculate effective decay from current state
        time_diff = epoch - current_epoch
        return asymptote + (current - asymptote) * np.exp(-decay_rate * time_diff)
    
    def sigmoid_growth(self, epoch: int, max_value: float, growth_rate: float, 
                      midpoint_epoch: float) -> float:
        """Sigmoid growth model for accuracy metrics"""
        return max_value / (1 + np.exp(-growth_rate * (epoch - midpoint_epoch)))
    
    def linear_growth(self, epoch: int, start_epoch: int, growth_rate: float, 
                     max_value: float) -> float:
        """Linear growth model with saturation"""
        if epoch < start_epoch:
            return 0
        growth = growth_rate * (epoch - start_epoch)
        return min(growth, max_value)
    
    def predict_loss(self, target_epoch: int, current_epoch: int = 22) -> Tuple[float, float]:
        """Predict loss at target epoch"""
        params = self.prediction_config['loss_prediction']['parameters']
        predicted = self.exponential_decay(
            target_epoch, 
            params['initial_loss'],
            params['current_loss'],
            current_epoch,
            params['decay_rate'],
            params['asymptote']
        )
        confidence = self.prediction_config['loss_prediction']['confidence']
        return predicted, confidence
    
    def predict_syntax_validity(self, target_epoch: int) -> Tuple[float, float]:
        """Predict syntax validity at target epoch"""
        params = self.prediction_config['syntax_validity_prediction']['parameters']
        predicted = self.sigmoid_growth(
            target_epoch,
            params['max_value'],
            params['growth_rate'],
            params['midpoint_epoch']
        )
        confidence = self.prediction_config['syntax_validity_prediction']['confidence']
        return predicted, confidence
    
    def predict_compilation_success(self, target_epoch: int) -> Tuple[float, float]:
        """Predict compilation success at target epoch"""
        params = self.prediction_config['compilation_success_prediction']['parameters']
        predicted = self.sigmoid_growth(
            target_epoch,
            params['max_value'],
            params['growth_rate'],
            params['midpoint_epoch']
        )
        confidence = self.prediction_config['compilation_success_prediction']['confidence']
        return predicted, confidence
    
    def predict_swe_convergence(self, target_epoch: int) -> Tuple[float, float]:
        """Predict SWE convergence at target epoch"""
        params = self.prediction_config['swe_convergence_prediction']['parameters']
        predicted = self.linear_growth(
            target_epoch,
            params['start_epoch'],
            params['growth_rate'],
            params['max_value']
        )
        confidence = self.prediction_config['swe_convergence_prediction']['confidence']
        return predicted, confidence
    
    def predict_combined_score(self, target_epoch: int) -> Tuple[float, float]:
        """Predict overall combined score"""
        # Get individual predictions
        loss, loss_conf = self.predict_loss(target_epoch)
        syntax, syntax_conf = self.predict_syntax_validity(target_epoch)
        compilation, comp_conf = self.predict_compilation_success(target_epoch)
        swe, swe_conf = self.predict_swe_convergence(target_epoch)
        
        # Get weights
        weights = self.config['convergence_criteria']['combined_score_formula']['weights']
        
        # Calculate combined score (loss needs to be inverted - lower is better)
        loss_component = max(0, 1 - loss)  # Invert loss (assuming max reasonable loss is 1)
        combined_score = (
            loss_component * weights['loss_component'] +
            syntax * weights['syntax_validity'] +
            compilation * weights['compilation_success'] +
            swe * weights['swe_convergence'] +
            0.95 * weights['halting_efficiency']  # Assume good halting
        )
        
        # Combined confidence (weighted average)
        combined_confidence = (
            loss_conf * weights['loss_component'] +
            syntax_conf * weights['syntax_validity'] +
            comp_conf * weights['compilation_success'] +
            swe_conf * weights['swe_convergence'] +
            0.9 * weights['halting_efficiency']  # Assume high confidence for halting
        )
        
        return combined_score, combined_confidence
    
    def generate_trajectory(self, start_epoch: int = 22, end_epoch: int = 50) -> Dict:
        """Generate full trajectory from start to end epoch"""
        epochs = list(range(start_epoch, end_epoch + 1))
        trajectory = {
            'epochs': epochs,
            'loss': [],
            'syntax_validity': [],
            'compilation_success': [],
            'swe_convergence': [],
            'combined_score': []
        }
        
        for epoch in epochs:
            loss, _ = self.predict_loss(epoch, start_epoch)
            syntax, _ = self.predict_syntax_validity(epoch)
            compilation, _ = self.predict_compilation_success(epoch)
            swe, _ = self.predict_swe_convergence(epoch)
            combined, _ = self.predict_combined_score(epoch)
            
            trajectory['loss'].append(loss)
            trajectory['syntax_validity'].append(syntax)
            trajectory['compilation_success'].append(compilation)
            trajectory['swe_convergence'].append(swe)
            trajectory['combined_score'].append(combined)
        
        return trajectory
    
    def plot_predictions(self, trajectory: Dict, output_path: str = "evaluations/predicted_trajectory.png"):
        """Plot prediction trajectory"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        epochs = trajectory['epochs']
        
        # Loss plot
        ax1.plot(epochs, trajectory['loss'], 'b-', linewidth=2, label='Predicted Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Prediction')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Code metrics plot
        ax2.plot(epochs, trajectory['syntax_validity'], 'g-', linewidth=2, label='Syntax Validity')
        ax2.plot(epochs, trajectory['compilation_success'], 'r-', linewidth=2, label='Compilation Success')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Code Quality Metrics')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        # SWE convergence plot
        ax3.plot(epochs, trajectory['swe_convergence'], 'm-', linewidth=2, label='SWE Convergence')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Convergence Score')
        ax3.set_title('SWE Search Convergence')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_ylim(0, 1)
        
        # Combined score plot
        ax4.plot(epochs, trajectory['combined_score'], 'k-', linewidth=2, label='Combined Score')
        target_score = self.config['convergence_criteria']['combined_score_formula']['target_combined_score']
        ax4.axhline(y=target_score, color='r', linestyle='--', alpha=0.7, label=f'Target ({target_score})')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Combined Score')
        ax4.set_title('Overall Performance Prediction')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Prediction plot saved to: {output_path}")
        return output_path
    
    def analyze_convergence_timeline(self, trajectory: Dict) -> Dict:
        """Analyze when convergence criteria will be met"""
        criteria = self.config['convergence_criteria']
        target_combined = criteria['combined_score_formula']['target_combined_score']
        
        analysis = {
            'convergence_predictions': {},
            'timeline': {},
            'recommendations': []
        }
        
        # Find when each metric reaches target
        epochs = trajectory['epochs']
        
        # Combined score convergence
        combined_scores = trajectory['combined_score']
        convergence_epoch = None
        for i, score in enumerate(combined_scores):
            if score >= target_combined:
                convergence_epoch = epochs[i]
                break
        
        analysis['convergence_predictions']['combined_score'] = {
            'target': target_combined,
            'convergence_epoch': convergence_epoch,
            'final_predicted_score': combined_scores[-1]
        }
        
        # Syntax validity
        syntax_target = criteria['primary_metrics']['syntax_validity']['target_percentage'] / 100
        syntax_convergence = None
        for i, score in enumerate(trajectory['syntax_validity']):
            if score >= syntax_target:
                syntax_convergence = epochs[i]
                break
        
        analysis['convergence_predictions']['syntax_validity'] = {
            'target': syntax_target,
            'convergence_epoch': syntax_convergence,
            'final_predicted_score': trajectory['syntax_validity'][-1]
        }
        
        # Generate recommendations
        if convergence_epoch and convergence_epoch <= 50:
            analysis['recommendations'].append(f"✅ Model likely to converge by epoch {convergence_epoch}")
        elif convergence_epoch:
            analysis['recommendations'].append(f"⚠️ Model may need extension to epoch {convergence_epoch} for full convergence")
        else:
            analysis['recommendations'].append("❌ Model unlikely to reach convergence criteria in current plan")
        
        return analysis
    
    def generate_full_report(self, target_epoch: int = 50) -> Dict:
        """Generate comprehensive prediction report"""
        print(f"🔮 Generating Prediction Report for Epoch {target_epoch}")
        print("=" * 50)
        
        # Individual predictions
        loss, loss_conf = self.predict_loss(target_epoch)
        syntax, syntax_conf = self.predict_syntax_validity(target_epoch)
        compilation, comp_conf = self.predict_compilation_success(target_epoch)
        swe, swe_conf = self.predict_swe_convergence(target_epoch)
        combined, combined_conf = self.predict_combined_score(target_epoch)
        
        # Generate trajectory
        trajectory = self.generate_trajectory(22, target_epoch)
        
        # Convergence analysis
        convergence_analysis = self.analyze_convergence_timeline(trajectory)
        
        report = {
            'target_epoch': target_epoch,
            'predictions': {
                'loss': {'value': loss, 'confidence': loss_conf},
                'syntax_validity': {'value': syntax, 'confidence': syntax_conf},
                'compilation_success': {'value': compilation, 'confidence': comp_conf},
                'swe_convergence': {'value': swe, 'confidence': swe_conf},
                'combined_score': {'value': combined, 'confidence': combined_conf}
            },
            'trajectory': trajectory,
            'convergence_analysis': convergence_analysis,
            'summary': {
                'overall_outlook': 'positive' if combined >= 0.8 else 'moderate' if combined >= 0.7 else 'concerning',
                'key_strengths': [],
                'key_concerns': [],
                'confidence_level': combined_conf
            }
        }
        
        # Add qualitative assessment
        if syntax >= 0.75:
            report['summary']['key_strengths'].append("Strong syntax validity predicted")
        if compilation >= 0.65:
            report['summary']['key_strengths'].append("Good compilation success expected")
        if swe >= 0.7:
            report['summary']['key_strengths'].append("SWE search convergence on track")
            
        if loss > 0.8:
            report['summary']['key_concerns'].append("Loss may not reach optimal levels")
        if syntax < 0.6:
            report['summary']['key_concerns'].append("Syntax validity below target")
        if combined < 0.75:
            report['summary']['key_concerns'].append("Combined score below convergence threshold")
        
        return report
    
    def print_report_summary(self, report: Dict):
        """Print formatted report summary"""
        print("\n📊 PREDICTION SUMMARY")
        print("=" * 30)
        
        pred = report['predictions']
        target_epoch = report['target_epoch']
        
        print(f"🎯 Epoch {target_epoch} Predictions:")
        print(f"   Loss: {pred['loss']['value']:.3f} (conf: {pred['loss']['confidence']:.2f})")
        print(f"   Syntax: {pred['syntax_validity']['value']:.1%} (conf: {pred['syntax_validity']['confidence']:.2f})")
        print(f"   Compilation: {pred['compilation_success']['value']:.1%} (conf: {pred['compilation_success']['confidence']:.2f})")
        print(f"   SWE Conv: {pred['swe_convergence']['value']:.3f} (conf: {pred['swe_convergence']['confidence']:.2f})")
        print(f"   Combined: {pred['combined_score']['value']:.3f} (conf: {pred['combined_score']['confidence']:.2f})")
        
        print(f"\n🔮 Overall Outlook: {report['summary']['overall_outlook'].upper()}")
        
        if report['summary']['key_strengths']:
            print("\n✅ Strengths:")
            for strength in report['summary']['key_strengths']:
                print(f"   • {strength}")
        
        if report['summary']['key_concerns']:
            print("\n⚠️ Concerns:")
            for concern in report['summary']['key_concerns']:
                print(f"   • {concern}")
        
        # Convergence timeline
        conv_analysis = report['convergence_analysis']
        if conv_analysis['convergence_predictions']['combined_score']['convergence_epoch']:
            epoch = conv_analysis['convergence_predictions']['combined_score']['convergence_epoch']
            print(f"\n🎯 Predicted Convergence: Epoch {epoch}")
        else:
            print(f"\n🎯 Predicted Convergence: Beyond Epoch {target_epoch}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Predict final training metrics")
    parser.add_argument("--target-epoch", type=int, default=50, help="Target epoch for prediction")
    parser.add_argument("--current-epoch", type=int, default=22, help="Current training epoch")
    parser.add_argument("--config", default="config/convergence_criteria.json", help="Convergence config file")
    parser.add_argument("--plot", action="store_true", help="Generate prediction plots")
    parser.add_argument("--output-dir", default="evaluations", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Create predictor
    predictor = MetricsPredictor(args.config)
    
    # Generate report
    report = predictor.generate_full_report(args.target_epoch)
    
    # Save report
    report_path = Path(args.output_dir) / f"prediction_report_epoch{args.target_epoch}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"💾 Full report saved to: {report_path}")
    
    # Generate plots if requested
    if args.plot:
        plot_path = Path(args.output_dir) / f"trajectory_prediction_epoch{args.target_epoch}.png"
        predictor.plot_predictions(report['trajectory'], str(plot_path))
    
    # Print summary
    predictor.print_report_summary(report)
    
    return report


if __name__ == "__main__":
    main()