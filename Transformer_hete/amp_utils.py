import torch
from torch.amp import autocast, GradScaler

class AmpHandler:
    """处理自动混合精度训练的工具类"""
    
    def __init__(self, enabled=True, init_scale=2**16, growth_factor=2.0, 
                 backoff_factor=0.5, growth_interval=1000, max_scale=2**24):
        """
        初始化 AMP 处理器
        
        参数:
            enabled (bool): 是否启用混合精度训练
            init_scale (float): 初始缩放因子
            growth_factor (float): 缩放因子增长率
            backoff_factor (float): 缩放因子回退率
            growth_interval (int): 增长间隔
            max_scale (float): 最大缩放因子
        """
        self.enabled = enabled
        self.scaler = torch.amp.GradScaler(
            device='cuda',
            enabled=enabled,
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval
        )
        
        # 检查CUDA是否可用
        if self.enabled and not torch.cuda.is_available():
            print("警告: CUDA不可用，AMP将被禁用!")
            self.enabled = False
            self.scaler = GradScaler(enabled=False)
    
    def autocast(self):
        """返回自动混合精度上下文管理器"""
        return autocast('cuda', enabled=self.enabled)
    
    def scale_loss(self, loss):
        """缩放损失值"""
        return self.scaler.scale(loss)
    
    def step(self, optimizer):
        """执行优化器步骤"""
        self.scaler.step(optimizer)
    
    def update(self):
        """更新梯度缩放器"""
        self.scaler.update()
        
    def unscale_(self, optimizer):
        """取消梯度缩放"""
        self.scaler.unscale_(optimizer)
        
    def get_scale(self):
        """获取当前缩放值"""
        return self.scaler.get_scale()
    
    def is_enabled(self):
        """检查AMP是否启用"""
        return self.enabled