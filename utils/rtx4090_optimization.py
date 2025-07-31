#!/usr/bin/env python3
"""
RTX 4090 Hardware Optimization for HRM

Provides RTX 4090-specific optimizations, CUDA configuration verification,
and performance monitoring for maximum training efficiency.
"""

import os
import sys
import subprocess
import platform
import json
import torch
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information"""
    info = {
        'platform': platform.system(),
        'platform_version': platform.version(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'pytorch_cuda_available': torch.cuda.is_available(),
        'pytorch_mps_available': torch.backends.mps.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(),
            'gpu_count': torch.cuda.device_count(),
            'gpu_devices': []
        })
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info = {
                'device_id': i,
                'name': props.name,
                'memory_total_gb': props.total_memory / (1024**3),
                'memory_allocated_gb': torch.cuda.memory_allocated(i) / (1024**3),
                'memory_reserved_gb': torch.cuda.memory_reserved(i) / (1024**3),
                'sm_count': props.multi_processor_count,
                'compute_capability': f"{props.major}.{props.minor}",
                'max_threads_per_block': props.max_threads_per_block,
                'max_threads_per_multiprocessor': props.max_threads_per_multiprocessor
            }
            info['gpu_devices'].append(gpu_info)
    
    return info


def check_rtx4090_compatibility() -> Dict[str, Any]:
    """Check RTX 4090 specific compatibility and optimizations"""
    results = {
        'rtx4090_detected': False,
        'optimal_settings': {},
        'warnings': [],
        'recommendations': []
    }
    
    if not torch.cuda.is_available():
        results['warnings'].append("CUDA not available - RTX 4090 optimizations not applicable")
        return results
    
    # Check for RTX 4090
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        if 'RTX 4090' in props.name or 'GeForce RTX 4090' in props.name:
            results['rtx4090_detected'] = True
            results['device_id'] = i
            results['device_name'] = props.name
            
            # RTX 4090 optimal settings
            results['optimal_settings'] = {
                'batch_size': 8,  # Large batch for 24GB VRAM
                'gradient_accumulation_steps': 2,
                'mixed_precision': True,
                'gradient_checkpointing': True,
                'max_sequence_length': 2048,  # Take advantage of memory
                'num_workers': 8,  # Multi-threaded data loading
                'pin_memory': True,
                'persistent_workers': True,
                'compile_model': True,  # torch.compile for Ada Lovelace
                'flash_attention': True,  # CUDA 11.8+ required
                'tensor_parallel': False,  # Single GPU optimization
            }
            
            # Memory-specific recommendations
            memory_gb = props.total_memory / (1024**3)
            if memory_gb >= 20:  # RTX 4090 has 24GB
                results['recommendations'].extend([
                    "Use larger batch sizes (6-8) for faster training",
                    "Enable gradient checkpointing for even larger models",
                    "Consider sequence lengths up to 2048 tokens",
                    "Use mixed precision (FP16/BF16) for 40%+ speedup"
                ])
            
            # Architecture-specific optimizations
            if props.major >= 8:  # Ada Lovelace (RTX 40 series)
                results['recommendations'].extend([
                    "Enable torch.compile() for 10-20% additional speedup",
                    "Use FlashAttention-2 for memory-efficient attention",
                    "Consider BFloat16 over Float16 for numerical stability"
                ])
            
            break
    
    if not results['rtx4090_detected']:
        results['warnings'].append("RTX 4090 not detected - using generic CUDA optimizations")
        
        # Generic high-end GPU settings
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            memory_gb = props.total_memory / (1024**3)
            
            if memory_gb >= 10:  # High-end GPU
                results['optimal_settings'] = {
                    'batch_size': min(6, int(memory_gb / 4)),
                    'mixed_precision': True,
                    'gradient_checkpointing': memory_gb < 16,
                    'compile_model': props.major >= 7,
                }
    
    return results


def verify_cuda_installation() -> Dict[str, Any]:
    """Verify CUDA installation and version compatibility"""
    results = {
        'cuda_available': torch.cuda.is_available(),
        'issues': [],
        'recommendations': []
    }
    
    if not torch.cuda.is_available():
        results['issues'].append("CUDA not available in PyTorch")
        results['recommendations'].append("Install CUDA-enabled PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return results
    
    # Check CUDA version
    cuda_version = torch.version.cuda
    if cuda_version:
        results['pytorch_cuda_version'] = cuda_version
        
        # Check for CUDA 12.x compatibility (RTX 4090 optimal)
        major_version = int(cuda_version.split('.')[0])
        if major_version >= 12:
            results['recommendations'].append("CUDA 12.x detected - optimal for RTX 4090")
        elif major_version == 11:
            if int(cuda_version.split('.')[1]) >= 8:
                results['recommendations'].append("CUDA 11.8+ detected - good RTX 4090 support")
            else:
                results['issues'].append(f"CUDA {cuda_version} may have limited RTX 4090 support")
                results['recommendations'].append("Consider upgrading to CUDA 12.x for optimal RTX 4090 performance")
        else:
            results['issues'].append(f"CUDA {cuda_version} is too old for RTX 4090")
    
    # Check cuDNN
    if torch.backends.cudnn.enabled:
        cudnn_version = torch.backends.cudnn.version()
        if cudnn_version:
            results['cudnn_version'] = cudnn_version
            if cudnn_version >= 8000:
                results['recommendations'].append(f"cuDNN {cudnn_version} is compatible")
            else:
                results['issues'].append(f"cuDNN {cudnn_version} may be outdated")
    else:
        results['issues'].append("cuDNN not enabled")
    
    # Check nvidia-smi availability
    try:
        nvidia_smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version,memory.total,memory.used,temperature.gpu,power.draw', '--format=csv,noheader,nounits'], 
                                                  text=True, stderr=subprocess.DEVNULL)
        
        lines = nvidia_smi_output.strip().split('\n')
        results['nvidia_smi_available'] = True
        results['gpu_status'] = []
        
        for i, line in enumerate(lines):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                gpu_status = {
                    'gpu_id': i,
                    'driver_version': parts[0],
                    'memory_total_mb': int(parts[1]) if parts[1].isdigit() else None,
                    'memory_used_mb': int(parts[2]) if parts[2].isdigit() else None,
                    'temperature_c': int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None,
                    'power_draw_w': float(parts[4]) if len(parts) > 4 and parts[4].replace('.', '').isdigit() else None
                }
                results['gpu_status'].append(gpu_status)
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        results['nvidia_smi_available'] = False
        results['issues'].append("nvidia-smi not available - cannot monitor GPU status")
    
    return results


def create_rtx4090_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create RTX 4090-optimized training configuration"""
    rtx_compat = check_rtx4090_compatibility()
    
    # Start with base config
    optimized_config = base_config.copy()
    
    if rtx_compat['rtx4090_detected']:
        # Apply RTX 4090 specific optimizations
        optimal = rtx_compat['optimal_settings']
        
        optimized_config.update({
            'batch_size': optimal['batch_size'],
            'gradient_accumulation_steps': optimal.get('gradient_accumulation_steps', 1),
            'mixed_precision': optimal['mixed_precision'],
            'gradient_checkpointing': optimal['gradient_checkpointing'],
            'compile_model': optimal.get('compile_model', False),
            'max_sequence_length': optimal.get('max_sequence_length', 1024),
            
            # Data loading optimizations
            'num_workers': optimal.get('num_workers', 4),
            'pin_memory': optimal.get('pin_memory', True),
            'persistent_workers': optimal.get('persistent_workers', True),
            
            # Memory optimizations
            'memory_efficient_attention': True,
            'flash_attention': optimal.get('flash_attention', False),
        })
        
        # Learning rate scaling for larger batch sizes
        if 'learning_rate' in optimized_config and optimal['batch_size'] > base_config.get('batch_size', 4):
            scale_factor = optimal['batch_size'] / base_config.get('batch_size', 4)
            optimized_config['learning_rate'] = base_config['learning_rate'] * scale_factor
            
    else:
        # Generic high-performance GPU optimizations
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            memory_gb = props.total_memory / (1024**3)
            
            if memory_gb >= 8:
                optimized_config.update({
                    'mixed_precision': True,
                    'gradient_checkpointing': memory_gb < 16,
                    'pin_memory': True,
                })
    
    return optimized_config


def create_monitoring_script() -> str:
    """Create GPU monitoring script for RTX 4090"""
    script_content = '''#!/usr/bin/env python3
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
        for i, line in enumerate(output.split('\\n')):
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
    print(f"\\n[{timestamp}] RTX 4090 Status:")
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
        print("\\n🛑 Monitoring stopped")

if __name__ == "__main__":
    monitor_loop()
'''
    return script_content


def setup_rtx4090_environment():
    """Setup optimal environment for RTX 4090 training"""
    print("🚀 Setting up RTX 4090 optimization environment...")
    
    # Get system information
    system_info = get_system_info()
    print(f"System: {system_info['platform']} {system_info['platform_version']}")
    print(f"Python: {system_info['python_version']}")
    print(f"PyTorch: {system_info['pytorch_version']}")
    
    if system_info['pytorch_cuda_available']:
        print(f"CUDA: {system_info['cuda_version']}")
        print(f"GPU Count: {system_info['gpu_count']}")
        
        for gpu in system_info['gpu_devices']:
            print(f"  GPU {gpu['device_id']}: {gpu['name']} ({gpu['memory_total_gb']:.1f}GB)")
    
    # Check RTX 4090 compatibility
    rtx_compat = check_rtx4090_compatibility()
    
    if rtx_compat['rtx4090_detected']:
        print(f"✅ RTX 4090 detected: {rtx_compat['device_name']}")
        print("🔧 Recommended optimizations:")
        for rec in rtx_compat['recommendations']:
            print(f"   • {rec}")
    else:
        print("⚠️  RTX 4090 not detected")
        if rtx_compat['warnings']:
            for warning in rtx_compat['warnings']:
                print(f"   ⚠️  {warning}")
    
    # Verify CUDA installation
    cuda_check = verify_cuda_installation()
    print("\\n🔍 CUDA Verification:")
    
    if cuda_check['issues']:
        print("Issues found:")
        for issue in cuda_check['issues']:
            print(f"   ❌ {issue}")
    
    if cuda_check['recommendations']:
        print("Recommendations:")
        for rec in cuda_check['recommendations']:
            print(f"   💡 {rec}")
    
    # Create monitoring script
    monitor_script_path = Path("monitor_rtx4090.py")
    with open(monitor_script_path, 'w') as f:
        f.write(create_monitoring_script())
    monitor_script_path.chmod(0o755)
    print(f"✅ Created monitoring script: {monitor_script_path}")
    print("   Run with: python monitor_rtx4090.py")
    
    # Create optimized config template
    config_template = {
        'data_path': 'mixed',
        'epochs': 40,
        'batch_size': 4,  # Will be optimized for RTX 4090
        'learning_rate': 4e-5,
        'mixed_precision': True,
        'gradient_checkpointing': True,
        'use_wandb': True,
        'save_every': 1000,
        'eval_every': 500
    }
    
    optimized_config = create_rtx4090_config(config_template)
    
    config_path = Path("rtx4090_optimized_config.json")
    with open(config_path, 'w') as f:
        json.dump(optimized_config, f, indent=2)
    print(f"✅ Created optimized config: {config_path}")
    
    # Create launch script
    launch_script = f'''#!/bin/bash
# RTX 4090 Optimized HRM Training Launch Script

echo "🚀 Starting RTX 4090 Optimized HRM Training"
echo "Device: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"

# Set optimal environment variables
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 4090 architecture
export CUDA_VISIBLE_DEVICES=0

# Launch training with optimized config
python train_hrm_optimized.py \\
    --data-path mixed \\
    --epochs {optimized_config['epochs']} \\
    --batch-size {optimized_config['batch_size']} \\
    --learning-rate {optimized_config['learning_rate']} \\
    --mixed-precision \\
    --gradient-checkpointing \\
    --use-wandb

echo "✅ Training completed"
'''
    
    launch_script_path = Path("launch_rtx4090_training.sh")
    with open(launch_script_path, 'w') as f:
        f.write(launch_script)
    launch_script_path.chmod(0o755)
    print(f"✅ Created launch script: {launch_script_path}")
    
    print("\\n🎉 RTX 4090 optimization setup complete!")
    print("\\nNext steps:")
    print("1. Run: python monitor_rtx4090.py (in separate terminal)")
    print("2. Run: ./launch_rtx4090_training.sh")
    print("3. Monitor performance and adjust batch size if needed")
    
    return {
        'system_info': system_info,
        'rtx_compatibility': rtx_compat,
        'cuda_check': cuda_check,
        'optimized_config': optimized_config,
        'monitoring_script': str(monitor_script_path),
        'launch_script': str(launch_script_path)
    }


if __name__ == "__main__":
    setup_rtx4090_environment()