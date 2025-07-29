#!/usr/bin/env python3
"""
Advanced training monitoring with stall detection and auto-alerts
"""

import time
import os
import re
import json
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import sys

class TrainingMonitor:
    def __init__(self, log_file="production_training.log", process_name="train_hrm_optimized.py"):
        self.log_file = log_file
        self.process_name = process_name
        self.last_check = None
        self.stall_threshold = 1800  # 30 minutes
        self.alert_file = "alerts/training_alerts.json"
        
        # Create alerts directory
        Path("alerts").mkdir(exist_ok=True)
        
    def check_process_status(self):
        """Check if training process is running"""
        try:
            result = subprocess.run(
                f"ps aux | grep '{self.process_name}' | grep -v grep",
                shell=True, capture_output=True, text=True
            )
            
            if result.stdout.strip():
                # Extract PID and other info
                lines = result.stdout.strip().split('\n')
                process_info = []
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 11:
                        process_info.append({
                            'pid': parts[1],
                            'cpu_percent': parts[2],
                            'mem_percent': parts[3],
                            'start_time': parts[8],
                            'command': ' '.join(parts[10:])
                        })
                return {'running': True, 'processes': process_info}
            else:
                return {'running': False, 'processes': []}
        except Exception as e:
            return {'running': 'unknown', 'error': str(e)}
    
    def analyze_log_file(self):
        """Analyze training log for progress and stalls"""
        if not Path(self.log_file).exists():
            return {'status': 'no_log', 'message': f"Log file {self.log_file} not found"}
        
        try:
            # Get file stats
            stat_info = Path(self.log_file).stat()
            last_modified = datetime.fromtimestamp(stat_info.st_mtime)
            file_size = stat_info.st_size
            
            # Read recent log entries
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            # Get last 50 lines for analysis
            recent_lines = lines[-50:] if len(lines) > 50 else lines
            
            # Extract metrics from recent lines
            latest_metrics = self.extract_latest_metrics(recent_lines)
            
            # Check for stalls
            time_since_update = datetime.now() - last_modified
            is_stalled = time_since_update.total_seconds() > self.stall_threshold
            
            return {
                'status': 'active' if not is_stalled else 'stalled',
                'last_modified': last_modified.isoformat(),
                'time_since_update': time_since_update.total_seconds(),
                'file_size_mb': file_size / (1024 * 1024),
                'latest_metrics': latest_metrics,
                'is_stalled': is_stalled,
                'recent_activity': self.analyze_recent_activity(recent_lines)
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def extract_latest_metrics(self, lines):
        """Extract latest training metrics from log lines"""
        metrics = {}
        
        # Patterns to match common metrics
        patterns = {
            'epoch': r'Epoch[:\s]+(\d+)',
            'batch': r'[Bb]atch[:\s]+(\d+)',
            'loss': r'[Ll]oss[:\s]+([0-9.]+)',
            'token_accuracy': r'[Tt]oken.*[Aa]cc.*[:\s]+([0-9.]+)',
            'syntax_validity': r'[Ss]yntax.*[Vv]alid.*[:\s]+([0-9.]+)',
            'compilation_success': r'[Cc]ompil.*[Ss]uccess.*[:\s]+([0-9.]+)',
            'swe_convergence': r'[Ss][Ww][Ee].*[Cc]onv.*[:\s]+([0-9.]+)',
            'lr': r'[Ll]earning[_\s][Rr]ate[:\s]+([0-9.e-]+)'
        }
        
        # Search from most recent lines backward
        for line in reversed(lines):
            for metric, pattern in patterns.items():
                if metric not in metrics:  # Only get the most recent value
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        try:
                            value = float(match.group(1))
                            metrics[metric] = value
                        except ValueError:
                            metrics[metric] = match.group(1)
        
        return metrics
    
    def analyze_recent_activity(self, lines):
        """Analyze recent activity patterns"""
        if not lines:
            return {'pattern': 'no_data'}
        
        # Count different types of log entries
        epoch_count = sum(1 for line in lines if re.search(r'epoch', line, re.IGNORECASE))
        error_count = sum(1 for line in lines if re.search(r'error|fail|exception', line, re.IGNORECASE))
        warning_count = sum(1 for line in lines if re.search(r'warn', line, re.IGNORECASE))
        
        # Determine activity pattern
        if epoch_count > 0:
            pattern = 'training_active'
        elif error_count > warning_count:
            pattern = 'error_prone'
        elif warning_count > 0:
            pattern = 'warnings_present'
        else:
            pattern = 'minimal_activity'
        
        return {
            'pattern': pattern,
            'epoch_entries': epoch_count,
            'errors': error_count,
            'warnings': warning_count,
            'total_lines': len(lines)
        }
    
    def detect_training_issues(self, log_analysis, process_status):
        """Detect potential training issues"""
        issues = []
        
        # Process issues
        if not process_status['running']:
            issues.append({
                'type': 'critical',
                'message': 'Training process not found - may have crashed or completed'
            })
        
        # Log file issues
        if log_analysis['status'] == 'no_log':
            issues.append({
                'type': 'critical', 
                'message': 'Log file missing - training may not have started'
            })
        elif log_analysis['status'] == 'stalled':
            issues.append({
                'type': 'warning',
                'message': f"No log updates for {log_analysis['time_since_update']:.0f} seconds"
            })
        
        # Metrics issues
        if 'latest_metrics' in log_analysis:
            metrics = log_analysis['latest_metrics']
            
            if 'loss' in metrics and metrics['loss'] > 10:
                issues.append({
                    'type': 'warning',
                    'message': f"High loss detected: {metrics['loss']:.3f}"
                })
            
            if 'syntax_validity' in metrics and metrics['syntax_validity'] == 0:
                issues.append({
                    'type': 'info',
                    'message': "Syntax validity still at 0% - tokenizer fix may be needed"
                })
        
        # Activity pattern issues
        if 'recent_activity' in log_analysis:
            activity = log_analysis['recent_activity']
            if activity['errors'] > 5:
                issues.append({
                    'type': 'warning',
                    'message': f"High error count in recent logs: {activity['errors']}"
                })
        
        return issues
    
    def save_alert(self, alert_data):
        """Save alert to file"""
        alert_data['timestamp'] = datetime.now().isoformat()
        
        alerts = []
        if Path(self.alert_file).exists():
            try:
                with open(self.alert_file, 'r') as f:
                    alerts = json.load(f)
            except:
                alerts = []
        
        alerts.append(alert_data)
        
        # Keep only last 100 alerts
        alerts = alerts[-100:]
        
        with open(self.alert_file, 'w') as f:
            json.dump(alerts, f, indent=2)
    
    def run_monitoring_check(self):
        """Run comprehensive monitoring check"""
        print(f"🔍 Training Monitor Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Check process status
        process_status = self.check_process_status()
        print(f"📊 Process Status: {'✅ Running' if process_status['running'] else '❌ Not Running'}")
        
        if process_status['running'] and process_status['processes']:
            for proc in process_status['processes']:
                print(f"   PID: {proc['pid']}, CPU: {proc['cpu_percent']}%, Mem: {proc['mem_percent']}%")
        
        # Analyze log file
        log_analysis = self.analyze_log_file()
        print(f"📝 Log Status: {log_analysis['status'].replace('_', ' ').title()}")
        
        if 'time_since_update' in log_analysis:
            minutes_ago = log_analysis['time_since_update'] / 60
            print(f"   Last Update: {minutes_ago:.1f} minutes ago")
        
        if 'latest_metrics' in log_analysis and log_analysis['latest_metrics']:
            print("📈 Latest Metrics:")
            for key, value in log_analysis['latest_metrics'].items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
        
        # Detect issues
        issues = self.detect_training_issues(log_analysis, process_status)
        
        if issues:
            print("\n🚨 Issues Detected:")
            for issue in issues:
                icon = "🔴" if issue['type'] == 'critical' else "🟡" if issue['type'] == 'warning' else "🔵"
                print(f"   {icon} {issue['message']}")
                
            # Save alert
            self.save_alert({
                'type': 'monitoring_check',
                'issues': issues,
                'process_status': process_status,
                'log_analysis': log_analysis
            })
        else:
            print("\n✅ No issues detected")
        
        return {
            'process_status': process_status,
            'log_analysis': log_analysis,
            'issues': issues
        }

def main():
    """Main monitoring function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced training monitoring")
    parser.add_argument("--log-file", default="production_training.log", help="Training log file")
    parser.add_argument("--process", default="train_hrm_optimized.py", help="Process name to monitor")
    parser.add_argument("--stall-threshold", type=int, default=1800, help="Stall threshold in seconds")
    parser.add_argument("--continuous", action="store_true", help="Run continuous monitoring")
    parser.add_argument("--interval", type=int, default=300, help="Check interval for continuous mode (seconds)")
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.log_file, args.process)
    monitor.stall_threshold = args.stall_threshold
    
    if args.continuous:
        print(f"🔄 Starting continuous monitoring (interval: {args.interval}s)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                monitor.run_monitoring_check()
                print(f"\n⏰ Next check in {args.interval} seconds...\n")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped by user")
    else:
        result = monitor.run_monitoring_check()
        
        # Exit codes for automation
        if any(issue['type'] == 'critical' for issue in result['issues']):
            sys.exit(2)  # Critical issues
        elif any(issue['type'] == 'warning' for issue in result['issues']):
            sys.exit(1)  # Warnings
        else:
            sys.exit(0)  # All good

if __name__ == "__main__":
    main()