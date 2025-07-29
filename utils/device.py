"""
Universal Device Detection for HRM-AstralDrift

Automatically detects and uses the best available device:
- CUDA (RTX 4090, Tesla, etc.)
- MPS (Apple Silicon M1/M2/M3)  
- CPU (fallback)

Usage:
    from utils.device import get_device, move_to_device
    
    device = get_device()
    tensor = move_to_device(tensor, device)
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device(prefer_device=None):
    """
    Get the best available device for training/inference.
    
    Args:
        prefer_device: Optional device preference ('cuda', 'mps', 'cpu')
        
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    
    # If user specifies a preference, try to honor it
    if prefer_device:
        if prefer_device == 'cuda' and torch.cuda.is_available():
            logger.info(f"Using preferred CUDA device: {torch.cuda.get_device_name()}")
            return 'cuda'
        elif prefer_device == 'mps' and torch.backends.mps.is_available():
            logger.info("Using preferred MPS device (Apple Silicon)")
            return 'mps'
        elif prefer_device == 'cpu':
            logger.info("Using preferred CPU device")
            return 'cpu'
        else:
            logger.warning(f"Preferred device '{prefer_device}' not available, falling back to auto-detection")
    
    # Auto-detection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"🚀 Using CUDA: {device_name} ({memory_gb:.1f}GB)")
        return 'cuda'
    
    elif torch.backends.mps.is_available():
        logger.info("🚀 Using MPS: Apple Silicon GPU acceleration")
        return 'mps'
    
    else:
        logger.info("⚠️  Using CPU: No GPU acceleration available")
        return 'cpu'


def move_to_device(tensor_or_dict, device=None):
    """
    Move tensor or dict of tensors to specified device.
    
    Args:
        tensor_or_dict: torch.Tensor or dict of tensors
        device: Target device (if None, uses get_device())
        
    Returns:
        Tensor or dict moved to device
    """
    if device is None:
        device = get_device()
    
    if isinstance(tensor_or_dict, dict):
        return {k: v.to(device) for k, v in tensor_or_dict.items()}
    else:
        return tensor_or_dict.to(device)


def get_device_info():
    """
    Get detailed information about available devices.
    
    Returns:
        dict: Device information
    """
    info = {
        'available_devices': [],
        'current_device': get_device(),
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available(),
    }
    
    # CUDA info
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            info['available_devices'].append({
                'type': 'cuda',
                'index': i,
                'name': gpu_props.name,
                'memory_gb': gpu_props.total_memory / 1024**3,
                'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
            })
    
    # MPS info  
    if torch.backends.mps.is_available():
        info['available_devices'].append({
            'type': 'mps',
            'name': 'Apple Silicon GPU',
            'note': 'Unified memory architecture'
        })
    
    # CPU info
    info['available_devices'].append({
        'type': 'cpu',
        'name': 'CPU',
        'note': 'Always available fallback'
    })
    
    return info


def print_device_info():
    """Print formatted device information."""
    info = get_device_info()
    
    print("🖥️  HRM-AstralDrift Device Information")
    print("=" * 50)
    print(f"Current device: {info['current_device'].upper()}")
    print(f"CUDA available: {info['cuda_available']}")
    print(f"MPS available: {info['mps_available']}")
    print()
    
    print("Available devices:")
    for device in info['available_devices']:
        if device['type'] == 'cuda':
            print(f"  🚀 {device['type'].upper()}: {device['name']} ({device['memory_gb']:.1f}GB)")
        elif device['type'] == 'mps':
            print(f"  🍎 {device['type'].upper()}: {device['name']}")
        else:
            print(f"  💻 {device['type'].upper()}: {device['name']}")


if __name__ == "__main__":
    # Test the device detection
    print_device_info()
    
    # Test tensor operations
    device = get_device()
    print(f"\nTesting tensor operations on {device}...")
    
    try:
        x = torch.randn(100, 100)
        x = move_to_device(x, device)
        y = torch.mm(x, x.t())
        print(f"✅ Matrix multiplication successful on {device}")
        print(f"Result tensor device: {y.device}")
    except Exception as e:
        print(f"❌ Error testing {device}: {e}")