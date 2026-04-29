"""
Memory management utilities for TTS processing
"""

import gc
import torch
import psutil


def get_memory_info():
    """Get current memory usage information"""
    memory_info = {}
    
    # CPU memory
    process = psutil.Process()
    memory_info['cpu_memory_mb'] = process.memory_info().rss / 1024 / 1024
    memory_info['cpu_memory_percent'] = process.memory_percent()
    
    # GPU memory (CUDA)
    if torch.cuda.is_available():
        memory_info['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        memory_info['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        memory_info['gpu_memory_max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    # Apple Silicon GPU (MPS)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't have as many memory monitoring functions as CUDA yet
        # but we can at least flag it's being used
        memory_info['mps_available'] = True
        # Future versions of torch might add more MPS memory info
    
    return memory_info


def empty_gpu_cache():
    """Empty GPU cache for CUDA or MPS if available"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, 'mps') and torch.mps.is_available():
        torch.mps.empty_cache()


def cleanup_memory(force_cuda_clear=False):
    """Perform memory cleanup operations"""
    try:
        # Python garbage collection
        collected = gc.collect()
        
        # Clear PyTorch cache if requested
        if force_cuda_clear:
            empty_gpu_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            device_type = "CUDA" if torch.cuda.is_available() else "MPS" if hasattr(torch, 'mps') and torch.mps.is_available() else "None"
            if device_type != "None":
                print(f"🧹 {device_type} cache cleared (collected {collected} objects)")
        
        elif collected > 0:
            print(f"🧹 Memory cleanup (collected {collected} objects)")
            
        return collected
            
    except Exception as e:
        print(f"⚠️ Memory cleanup warning: {e}")
        return 0


def safe_delete_tensors(*tensors):
    """Safely delete tensors and free memory"""
    for tensor in tensors:
        if tensor is not None:
            try:
                if hasattr(tensor, 'cpu'):
                    # Move to CPU first to free GPU memory
                    tensor = tensor.cpu()
                del tensor
            except Exception as e:
                print(f"⚠️ Warning during tensor cleanup: {e}") 