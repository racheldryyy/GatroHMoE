"""
GPU utility functions - Single GPU mode only

Provides GPU detection, configuration and model setup
"""

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import os
import warnings


def get_gpu_info():
    """获取详细的GPU信息"""
    if not torch.cuda.is_available():
        return {
            'available': False,
            'count': 0,
            'devices': [],
            'total_memory': 0
        }
    
    num_gpus = torch.cuda.device_count()
    devices = []
    total_memory = 0
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        device_info = {
            'id': i,
            'name': props.name,
            'memory_total': props.total_memory / 1024**3,  # GB
            'memory_free': 0,  # 将在下面计算
            'compute_capability': f"{props.major}.{props.minor}",
            'multiprocessor_count': props.multi_processor_count
        }
        
        # 获取当前可用内存
        try:
            torch.cuda.set_device(i)
            device_info['memory_free'] = torch.cuda.get_device_properties(i).total_memory / 1024**3
            # 更精确的可用内存计算
            device_info['memory_allocated'] = torch.cuda.memory_allocated(i) / 1024**3
            device_info['memory_reserved'] = torch.cuda.memory_reserved(i) / 1024**3
            device_info['memory_free'] = device_info['memory_total'] - device_info['memory_reserved']
        except:
            pass
            
        devices.append(device_info)
        total_memory += device_info['memory_total']
    
    return {
        'available': True,
        'count': num_gpus,
        'devices': devices,
        'total_memory': total_memory
    }


def print_gpu_status():
    """Print GPU status information"""
    gpu_info = get_gpu_info()
    
    if not gpu_info['available']:
        print("CUDA not available")
        return gpu_info
    
    print(f"Found {gpu_info['count']} GPU(s) (total memory: {gpu_info['total_memory']:.1f}GB)")
    print("-" * 60)
    
    for device in gpu_info['devices']:
        status = "[OK]" if device['memory_free'] > 1.0 else "[LOW]"
        print(f"{status} GPU {device['id']}: {device['name']}")
        print(f"   Memory: {device['memory_total']:.1f}GB (available: {device['memory_free']:.1f}GB)")
        print(f"   Compute: {device['compute_capability']} | SMs: {device['multiprocessor_count']}")
        
        if device['memory_free'] < 1.0:
            print(f"   Warning: Low available memory, may affect training")
        print()
    
    return gpu_info


def select_best_gpus(min_memory_gb=2.0, max_gpus=None):
    """
    Select single GPU to avoid multi-GPU issues
    
    Args:
        min_memory_gb: Minimum memory requirement (GB)
        max_gpus: Ignored, always returns single GPU
    
    Returns:
        list: Selected GPU ID list (max 1 GPU)
    """
    gpu_info = get_gpu_info()
    
    if not gpu_info['available']:
        return []
    
    # Find first GPU with enough memory
    for device in gpu_info['devices']:
        if device['memory_free'] >= min_memory_gb:
            gpu_ids = [device['id']]
            print(f"Selected GPU {device['id']} (available memory: {device['memory_free']:.1f}GB)")
            print(f"Single GPU mode to avoid DataParallel issues")
            return gpu_ids
    
    print(f"No suitable GPU found (need >= {min_memory_gb}GB memory)")
    return []


def setup_multi_gpu(model, gpu_ids=None, strategy='dp'):
    """
    Setup single GPU training (no DataParallel)
    
    Args:
        model: PyTorch model
        gpu_ids: GPU ID list, None for auto-detect (limited to single GPU)
        strategy: Ignored, single GPU mode only
    
    Returns:
        wrapped_model, device, is_parallel
    """
    
    if gpu_ids is None:
        gpu_ids = select_best_gpus()
    
    if not gpu_ids:
        print("Using CPU mode")
        return model.cpu(), torch.device('cpu'), False
    
    # Force single GPU only
    if len(gpu_ids) > 1:
        print(f"Multiple GPUs provided, using only GPU {gpu_ids[0]}")
        gpu_ids = [gpu_ids[0]]
    
    primary_gpu = gpu_ids[0]
    device = torch.device(f'cuda:{primary_gpu}')
    
    # Move model to GPU
    model = model.to(device)
    
    print(f"Single GPU mode: GPU {primary_gpu}")
    return model, device, False


def get_optimal_batch_size(base_batch_size, num_gpus, gpu_memory_gb):
    """
    Optimize batch size for single GPU training
    
    Args:
        base_batch_size: Base batch size
        num_gpus: Number of GPUs (forced to 1 or 0)
        gpu_memory_gb: Single GPU memory size
    
    Returns:
        optimal_batch_size
    """
    if num_gpus == 0:
        return max(4, base_batch_size // 4)  # Small batch for CPU
    
    # Single GPU optimization
    if num_gpus > 1:
        print(f"Single GPU mode: ignoring multi-GPU optimization")
        num_gpus = 1
    
    # Adjust based on memory size
    memory_factor = min(gpu_memory_gb / 8.0, 2.0)  # 8GB baseline, max 2x
    
    optimal_size = int(base_batch_size * memory_factor)
    
    # Round to multiple of 8
    optimal_size = ((optimal_size + 7) // 8) * 8
    
    # Limit range for single GPU
    optimal_size = max(8, min(optimal_size, 128))  # Lower upper limit
    
    print(f"Batch size optimization: {base_batch_size} -> {optimal_size} "
          f"(memory: {gpu_memory_gb:.1f}GB)")
    
    return optimal_size


def auto_mixed_precision_setup(model, enabled=True):
    """
    自动混合精度设置
    
    Args:
        model: PyTorch模型
        enabled: 是否启用AMP
    
    Returns:
        scaler (GradScaler 或 None)
    """
    if not enabled or not torch.cuda.is_available():
        print("Mixed precision: disabled")
        return None
    
    # Check if GPU supports Tensor Cores (compute capability >= 7.0)
    device_cap = torch.cuda.get_device_capability()
    supports_amp = device_cap[0] >= 7
    
    if supports_amp:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        print(f"Mixed precision: enabled (compute capability: {device_cap[0]}.{device_cap[1]})")
        return scaler
    else:
        print(f"Mixed precision: not supported (compute capability: {device_cap[0]}.{device_cap[1]} < 7.0)")
        return None


def monitor_gpu_usage():
    """Monitor GPU usage"""
    if not torch.cuda.is_available():
        return
    
    print("GPU usage monitoring:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        
        utilization = (reserved / total) * 100
        status = "[HIGH]" if utilization > 80 else "[MED]" if utilization > 50 else "[LOW]"
        
        print(f"  {status} GPU {i}: {utilization:.1f}% "
              f"({allocated:.1f}GB / {reserved:.1f}GB / {total:.1f}GB)")


class SmartGPUManager:
    """GPU Manager - Single GPU mode only"""
    
    def __init__(self, min_memory_gb=2.0, enable_amp=True):
        self.min_memory_gb = min_memory_gb
        self.enable_amp = enable_amp
        self.gpu_info = None
        self.selected_gpus = []
        
    def initialize(self):
        """Initialize GPU environment (single GPU mode)"""
        print("Initializing GPU environment (single GPU mode)...")
        self.gpu_info = print_gpu_status()
        self.selected_gpus = select_best_gpus(self.min_memory_gb)
        
        if self.selected_gpus:
            # Set CUDA visible devices
            gpu_str = ','.join(map(str, self.selected_gpus))
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
            print(f"Set CUDA_VISIBLE_DEVICES={gpu_str}")
        
        return self.selected_gpus
    
    def setup_model(self, model, base_batch_size=32, strategy='dp'):
        """Setup model and optimization config (single GPU mode)"""
        if not self.selected_gpus:
            print("No GPU available, using CPU")
            return model.cpu(), torch.device('cpu'), base_batch_size, None
        
        # Single GPU setup
        model, device, is_parallel = setup_multi_gpu(model, self.selected_gpus, strategy)
        
        # Optimize batch size
        avg_memory = sum(
            gpu['memory_free'] for gpu in self.gpu_info['devices'] 
            if gpu['id'] in self.selected_gpus
        ) / len(self.selected_gpus)
        
        optimal_batch_size = get_optimal_batch_size(
            base_batch_size, len(self.selected_gpus), avg_memory
        )
        
        # Setup mixed precision
        scaler = auto_mixed_precision_setup(model, self.enable_amp)
        
        return model, device, optimal_batch_size, scaler
    
    def monitor(self):
        """Monitor GPU status"""
        monitor_gpu_usage()


# Convenience function
def smart_device_setup(model=None, base_batch_size=32, min_memory_gb=2.0, enable_amp=True):
    """
    One-click smart device setup
    
    Returns:
        device, batch_size, scaler, gpu_manager
    """
    manager = SmartGPUManager(min_memory_gb, enable_amp)
    gpus = manager.initialize()
    
    if model is not None:
        model, device, batch_size, scaler = manager.setup_model(model, base_batch_size)
        return model, device, batch_size, scaler, manager
    else:
        device = torch.device(f'cuda:{gpus[0]}' if gpus else 'cpu')
        avg_memory = 8.0  # Default assumption
        if gpus and manager.gpu_info['devices']:
            avg_memory = sum(
                gpu['memory_free'] for gpu in manager.gpu_info['devices'] 
                if gpu['id'] in gpus
            ) / len(gpus)
        
        batch_size = get_optimal_batch_size(base_batch_size, len(gpus), avg_memory)
        scaler = auto_mixed_precision_setup(None, enable_amp) if gpus else None
        
        return device, batch_size, scaler, manager


# Usage example
if __name__ == "__main__":
    # Basic GPU info
    print_gpu_status()
    
    # GPU selection
    selected = select_best_gpus(min_memory_gb=1.0)
    print(f"Selected GPU: {selected}")
    
    # Create test model
    import torchvision.models as models
    model = models.resnet18(num_classes=10)
    
    # One-click setup
    model, device, batch_size, scaler, manager = smart_device_setup(
        model, base_batch_size=32, min_memory_gb=1.0
    )
    
    print(f"Final config: device={device}, batch_size={batch_size}, scaler={scaler is not None}")