#!/usr/bin/env python3
"""
Comprehensive Error Handling System for HRM

Provides robust error handling, recovery mechanisms, and detailed logging
for all critical components in the HRM training and inference pipeline.
"""

import functools
import logging
import traceback
import torch
import psutil
import gc
from typing import Any, Callable, Optional, Dict, Union
from pathlib import Path
import json
import time


class HRMError(Exception):
    """Base exception class for HRM-specific errors"""
    pass


class ModelInitializationError(HRMError):
    """Raised when model initialization fails"""
    pass


class DatasetError(HRMError):
    """Raised when dataset loading or processing fails"""
    pass


class TrainingError(HRMError):
    """Raised when training process encounters errors"""
    pass


class MemoryError(HRMError):
    """Raised when memory-related issues occur"""
    pass


class DeviceError(HRMError):
    """Raised when device-related issues occur"""
    pass


class ErrorRecoveryManager:
    """Manages error recovery and system state monitoring"""
    
    def __init__(self, logger: Optional[logging.Logger] = None, 
                 max_retries: int = 3,
                 recovery_strategies: Optional[Dict[str, Callable]] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.max_retries = max_retries
        self.error_history = []
        self.recovery_strategies = recovery_strategies or {}
        self.system_stats = {}
        
    def log_system_state(self):
        """Log current system state for debugging"""
        try:
            # Memory stats
            memory = psutil.virtual_memory()
            self.system_stats['memory_used_gb'] = memory.used / (1024**3)
            self.system_stats['memory_available_gb'] = memory.available / (1024**3)
            self.system_stats['memory_percent'] = memory.percent
            
            # GPU stats if available
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    gpu_used = torch.cuda.memory_allocated(i) / (1024**3)
                    self.system_stats[f'gpu_{i}_total_gb'] = gpu_memory
                    self.system_stats[f'gpu_{i}_used_gb'] = gpu_used
                    self.system_stats[f'gpu_{i}_utilization'] = (gpu_used / gpu_memory) * 100
            
            # MPS stats if available
            if torch.backends.mps.is_available():
                self.system_stats['mps_available'] = True
                
            self.logger.debug(f"System stats: {self.system_stats}")
            
        except Exception as e:
            self.logger.warning(f"Failed to collect system stats: {e}")
    
    def attempt_memory_recovery(self):
        """Attempt to recover memory"""
        try:
            self.logger.info("🔄 Attempting memory recovery...")
            
            # Clear Python garbage
            collected = gc.collect()
            self.logger.info(f"   Collected {collected} objects")
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("   Cleared CUDA cache")
                
            # Clear MPS cache if available
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                self.logger.info("   Cleared MPS cache")
                
            self.log_system_state()
            return True
            
        except Exception as e:
            self.logger.error(f"Memory recovery failed: {e}")
            return False
    
    def record_error(self, error: Exception, context: Dict[str, Any]):
        """Record error for analysis and recovery"""
        error_record = {
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'system_stats': self.system_stats.copy()
        }
        self.error_history.append(error_record)
        
        # Keep only last 100 errors
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if operation should be retried"""
        if attempt >= self.max_retries:
            return False
            
        # Don't retry certain critical errors
        non_retryable = (KeyboardInterrupt, SystemExit, ModelInitializationError)
        if isinstance(error, non_retryable):
            return False
            
        # Retry memory errors after recovery attempt
        if isinstance(error, (RuntimeError, torch.cuda.OutOfMemoryError)):
            return self.attempt_memory_recovery()
            
        return True


def robust_error_handler(max_retries: int = 3, 
                        recovery_manager: Optional[ErrorRecoveryManager] = None,
                        logger: Optional[logging.Logger] = None):
    """Decorator for robust error handling with automatic recovery"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or logging.getLogger(func.__module__)
            _recovery_manager = recovery_manager or ErrorRecoveryManager(_logger, max_retries)
            
            for attempt in range(max_retries + 1):
                try:
                    _recovery_manager.log_system_state()
                    result = func(*args, **kwargs)
                    
                    # Log success after previous failures
                    if attempt > 0:
                        _logger.info(f"✅ {func.__name__} succeeded on attempt {attempt + 1}")
                    
                    return result
                    
                except Exception as e:
                    context = {
                        'function': func.__name__,
                        'attempt': attempt + 1,
                        'args_types': [type(arg).__name__ for arg in args],
                        'kwargs_keys': list(kwargs.keys())
                    }
                    
                    _recovery_manager.record_error(e, context)
                    
                    if attempt == max_retries:
                        _logger.error(f"❌ {func.__name__} failed after {max_retries + 1} attempts")
                        _logger.error(f"   Final error: {e}")
                        _logger.error(f"   Traceback: {traceback.format_exc()}")
                        raise
                    
                    should_retry = _recovery_manager.should_retry(e, attempt)
                    if not should_retry:
                        _logger.error(f"❌ {func.__name__} failed with non-retryable error: {e}")
                        raise
                    
                    _logger.warning(f"⚠️  {func.__name__} failed on attempt {attempt + 1}, retrying... Error: {e}")
                    time.sleep(min(2 ** attempt, 10))  # Exponential backoff, max 10s
            
        return wrapper
    return decorator


def validate_model_config(config) -> None:
    """Validate model configuration with detailed error messages"""
    required_fields = [
        'batch_size', 'seq_len', 'vocab_size', 'hidden_size', 
        'num_heads', 'H_layers', 'L_layers', 'pos_encodings'
    ]
    
    for field in required_fields:
        if not hasattr(config, field):
            raise ModelInitializationError(f"Missing required config field: {field}")
    
    # Validate numeric constraints
    if config.hidden_size % config.num_heads != 0:
        raise ModelInitializationError(
            f"hidden_size ({config.hidden_size}) must be divisible by num_heads ({config.num_heads})"
        )
    
    if config.batch_size <= 0:
        raise ModelInitializationError(f"batch_size must be positive, got {config.batch_size}")
    
    if config.seq_len <= 0:
        raise ModelInitializationError(f"seq_len must be positive, got {config.seq_len}")
    
    # Validate positional encoding
    valid_pos_encodings = ["rope", "rotary", "learned"]
    if config.pos_encodings not in valid_pos_encodings:
        raise ModelInitializationError(
            f"Invalid pos_encodings '{config.pos_encodings}'. Must be one of: {valid_pos_encodings}"
        )


def validate_dataset_path(dataset_path: Union[str, Path]) -> Path:
    """Validate dataset path with detailed error messages"""
    path = Path(dataset_path)
    
    if not path.exists():
        raise DatasetError(f"Dataset path does not exist: {path}")
    
    if path.is_file():
        if not path.suffix.lower() in ['.json', '.jsonl']:
            raise DatasetError(f"Dataset file must be JSON/JSONL format, got: {path.suffix}")
    elif path.is_dir():
        # Check for expected files in directory
        expected_files = ['instances.json', 'metadata.json']
        missing_files = [f for f in expected_files if not (path / f).exists()]
        if missing_files:
            raise DatasetError(f"Missing required files in dataset directory: {missing_files}")
    else:
        raise DatasetError(f"Dataset path is neither file nor directory: {path}")
    
    return path


def check_device_compatibility(device: torch.device, model_dtype: torch.dtype) -> None:
    """Check device and dtype compatibility"""
    if device.type == 'mps':
        if model_dtype == torch.bfloat16:
            raise DeviceError("BFloat16 is not supported on MPS devices. Use float16 or float32.")
        
        if not torch.backends.mps.is_available():
            raise DeviceError("MPS backend is not available on this system")
    
    elif device.type == 'cuda':
        if not torch.cuda.is_available():
            raise DeviceError("CUDA is not available on this system")
        
        # Check GPU memory
        try:
            gpu_memory = torch.cuda.get_device_properties(device.index).total_memory
            if gpu_memory < 4 * (1024**3):  # 4GB minimum
                raise DeviceError(f"GPU has insufficient memory: {gpu_memory / (1024**3):.1f}GB < 4GB required")
        except Exception as e:
            raise DeviceError(f"Failed to check GPU memory: {e}")


def safe_save_checkpoint(model: torch.nn.Module, 
                        optimizer: torch.optim.Optimizer,
                        checkpoint_path: Path,
                        metadata: Dict[str, Any]) -> bool:
    """Safely save checkpoint with error handling and verification"""
    try:
        # Create backup if checkpoint exists
        if checkpoint_path.exists():
            backup_path = checkpoint_path.with_suffix('.backup')
            checkpoint_path.rename(backup_path)
        
        # Save checkpoint
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metadata': metadata,
            'timestamp': time.time()
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Verify checkpoint can be loaded
        verification = torch.load(checkpoint_path, map_location='cpu')
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'metadata']
        
        for key in required_keys:
            if key not in verification:
                raise RuntimeError(f"Checkpoint verification failed: missing key '{key}'")
        
        # Remove backup if save was successful
        backup_path = checkpoint_path.with_suffix('.backup')
        if backup_path.exists():
            backup_path.unlink()
        
        return True
        
    except Exception as e:
        logging.error(f"Checkpoint save failed: {e}")
        
        # Restore backup if available
        backup_path = checkpoint_path.with_suffix('.backup')
        if backup_path.exists():
            backup_path.rename(checkpoint_path)
        
        return False


def create_error_report(error_manager: ErrorRecoveryManager, 
                       output_path: Path) -> None:
    """Create comprehensive error report for debugging"""
    try:
        report = {
            'timestamp': time.time(),
            'system_info': {
                'python_version': str(torch.__version__),
                'torch_version': str(torch.__version__),
                'cuda_available': torch.cuda.is_available(),
                'mps_available': torch.backends.mps.is_available(),
            },
            'system_stats': error_manager.system_stats,
            'error_history': error_manager.error_history,
            'error_summary': {}
        }
        
        # Create error summary
        error_types = {}
        for error_record in error_manager.error_history:
            error_type = error_record['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        report['error_summary'] = error_types
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logging.info(f"Error report saved to: {output_path}")
        
    except Exception as e:
        logging.error(f"Failed to create error report: {e}")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test error handling
    @robust_error_handler(max_retries=2)
    def test_function(should_fail: bool = False):
        if should_fail:
            raise RuntimeError("Test error")
        return "Success!"
    
    print("🧪 Testing error handling system...")
    
    # Test success case
    result = test_function(False)
    print(f"✅ Success case: {result}")
    
    # Test retry mechanism
    try:
        test_function(True)
    except RuntimeError as e:
        print(f"✅ Retry mechanism working: {e}")
    
    print("🎉 Error handling system test complete!")