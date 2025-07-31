#!/usr/bin/env python3
"""
RTX 4090 Training Monitor

Real-time monitoring of RTX 4090 during HRM training.
Run this script in a separate terminal while training.
"""

import time
import subprocess
import json
from datetime import datetime

def get_gpu_stats():
    """Get current GPU statistics using nvidia-smi"""
    try:
        cmd = [
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory',
            '--format=csv,noheader,nounits'
        ]
        output = subprocess.check_output(cmd, text=True).strip()
        
        stats = []
        for i, line in enumerate(output.split('\n')):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 7:
                stats.append({
                    'gpu_id': i,
                    'utilization_percent': int(parts[0]) if parts[0].isdigit() else 0,
                    'memory_used_mb': int(parts[1]) if parts[1].isdigit() else 0,
                    'memory_total_mb': int(parts[2]) if parts[2].isdigit() else 0,
                    'temperature_c': int(parts[3]) if parts[3].isdigit() else 0,
                    'power_draw_w': float(parts[4]) if parts[4].replace('.', '').isdigit() else 0,
                    'gpu_clock_mhz': int(parts[5]) if parts[5].isdigit() else 0,
                    'memory_clock_mhz': int(parts[6]) if parts[6].isdigit() else 0,
                })
        return stats
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return []

def print_stats(stats):
    """Print formatted GPU statistics"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] RTX 4090 Status:")
    print("=" * 80)
    
    for gpu in stats:
        memory_percent = (gpu['memory_used_mb'] / gpu['memory_total_mb']) * 100
        print(f"GPU {gpu['gpu_id']}: {gpu['utilization_percent']:3d}% util | "
              f"{gpu['memory_used_mb']:5d}/{gpu['memory_total_mb']:5d}MB ({memory_percent:5.1f}%) | "
              f"{gpu['temperature_c']:2d}°C | {gpu['power_draw_w']:5.1f}W | "
              f"GPU: {gpu['gpu_clock_mhz']}MHz | MEM: {gpu['memory_clock_mhz']}MHz")

def monitor_loop():
    """Main monitoring loop"""
    print("🚀 RTX 4090 Training Monitor Started")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            stats = get_gpu_stats()
            if stats:
                print_stats(stats)
                
                # Check for issues
                for gpu in stats:
                    if gpu['temperature_c'] > 85:
                        print(f"⚠️  WARNING: GPU {gpu['gpu_id']} temperature high: {gpu['temperature_c']}°C")
                    if gpu['power_draw_w'] > 400:
                        print(f"⚠️  WARNING: GPU {gpu['gpu_id']} power draw high: {gpu['power_draw_w']}W")
                    if memory_percent > 95:
                        print(f"⚠️  WARNING: GPU {gpu['gpu_id']} memory usage high: {memory_percent:.1f}%")
            
            time.sleep(2)  # Update every 2 seconds
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")

if __name__ == "__main__":
    monitor_loop()
