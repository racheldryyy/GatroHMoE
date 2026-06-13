import torch
import psutil
import gc
import os
from functools import wraps

def get_memory_info():
    """获取系统和GPU内存信息"""
    # 系统内存
    system_memory = psutil.virtual_memory()
    system_used = system_memory.used / 1024**3  # GB
    system_total = system_memory.total / 1024**3  # GB
    system_percent = system_memory.percent
    
    print(f"🖥️  系统内存: {system_used:.2f}GB / {system_total:.2f}GB ({system_percent:.1f}%)")
    
    # GPU内存
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"🎮 GPU {i} 内存: 已分配 {allocated:.2f}GB, 已缓存 {cached:.2f}GB, 总计 {total:.1f}GB")
    
    return system_used, system_percent

def memory_cleanup():
    """执行内存清理"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def memory_monitor(func):
    """内存监控装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"📊 {func.__name__} 执行前:")
        get_memory_info()
        
        result = func(*args, **kwargs)
        
        print(f"📊 {func.__name__} 执行后:")
        get_memory_info()
        memory_cleanup()
        
        return result
    return wrapper

class MemoryManager:
    """内存管理器"""
    
    def __init__(self, memory_threshold=85):
        self.memory_threshold = memory_threshold
        
    def check_memory(self):
        """检查内存使用情况"""
        system_used, system_percent = get_memory_info()
        
        if system_percent > self.memory_threshold:
            print(f"⚠️  内存使用率过高 ({system_percent:.1f}%)，执行清理...")
            memory_cleanup()
            return False
        return True
    
    def auto_batch_size(self, base_batch_size=8):
        """根据内存情况自动调整批次大小"""
        system_memory = psutil.virtual_memory()
        available_gb = system_memory.available / 1024**3
        
        if available_gb < 4:
            return max(2, base_batch_size // 4)
        elif available_gb < 8:
            return max(4, base_batch_size // 2)
        elif available_gb < 12:
            return base_batch_size
        else:
            return base_batch_size
    
    def gradient_checkpointing_recommendation(self):
        """推荐是否使用梯度检查点"""
        system_memory = psutil.virtual_memory()
        available_gb = system_memory.available / 1024**3
        
        # 16GB系统内存建议使用梯度检查点
        return available_gb < 12

# 内存优化配置生成器
def generate_memory_config():
    """生成内存优化配置"""
    system_memory = psutil.virtual_memory()
    total_gb = system_memory.total / 1024**3
    
    if total_gb <= 16:  # 16GB或更少
        config = {
            'batch_size': 4,
            'num_workers': 2,
            'pin_memory': False,
            'prefetch_factor': 1,
            'persistent_workers': False,
            'gradient_accumulation': 8,
            'use_gradient_checkpointing': True,
            'use_amp': True,
            'max_experts': 3,  # 减少专家数量
        }
        print("🔧 检测到16GB内存，使用极限优化配置")
    elif total_gb <= 32:  # 32GB
        config = {
            'batch_size': 8,
            'num_workers': 4,
            'pin_memory': False,
            'prefetch_factor': 2,
            'persistent_workers': False,
            'gradient_accumulation': 4,
            'use_gradient_checkpointing': True,
            'use_amp': True,
            'max_experts': 4,
        }
        print("🔧 检测到32GB内存，使用标准优化配置")
    else:  # 32GB以上
        config = {
            'batch_size': 16,
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': True,
            'gradient_accumulation': 2,
            'use_gradient_checkpointing': False,
            'use_amp': True,
            'max_experts': 4,
        }
        print("🔧 检测到充足内存，使用性能优化配置")
    
    return config