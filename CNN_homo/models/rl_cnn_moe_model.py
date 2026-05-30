import torch
import torch.nn as nn
import torch.nn.functional as F
from config import HIDDEN_DIM, TOP_K_EXPERTS, EXPERT_DROPOUT, CNN_EXPERT_MODELS
from models.cnn_expert_models import (
    ResidualCNN, DenseNetCNN, MobileNetV3CNN, EfficientNetCNN, 
    RegNetCNN, ResNeXtCNN
)

class RLCNNMixtureOfExperts(nn.Module):
    """集成强化学习的CNN混合专家模型"""
    
    def __init__(self, model_names, num_colon_classes, num_ugi_classes, 
                 num_colon_disease_classes, num_ugi_disease_classes, 
                 drop_rate=EXPERT_DROPOUT, top_k=TOP_K_EXPERTS):
        super(RLCNNMixtureOfExperts, self).__init__()
        
        self.model_names = model_names
        self.num_experts = len(model_names)
        self.top_k = top_k
        
        # 保存类别数量
        self.num_classes = {
            'colon': num_colon_classes,
            'ugi': num_ugi_classes,
            'colon_disease': num_colon_disease_classes,
            'ugi_disease': num_ugi_disease_classes
        }
        
        # 专家模型映射
        self.expert_model_map = {
            'ResidualCNN': ResidualCNN,
            'DenseNetCNN': DenseNetCNN,
            'MobileNetV3CNN': MobileNetV3CNN,
            'EfficientNetCNN': EfficientNetCNN,
            'RegNetCNN': RegNetCNN,
            'ResNeXtCNN': ResNeXtCNN,
        }
        
        # 初始化专家网络
        self.experts = nn.ModuleList()
        self.expert_dims = []
        
        for model_name in model_names:
            if model_name in self.expert_model_map:
                # 创建CNN专家模型
                expert_class = self.expert_model_map[model_name]
                expert = expert_class(num_classes=1000, pretrained=True)  # 使用预训练权重
                
                # 获取特征维度
                feature_dim = expert.get_feature_dim()
                self.expert_dims.append(feature_dim)
                
                # 移除分类头，只保留特征提取
                if hasattr(expert, 'classifier'):
                    expert.classifier = nn.Identity()
                
                self.experts.append(expert)
            else:
                raise ValueError(f"未支持的CNN专家类型: {model_name}")
        
        # 统一特征维度
        self.unified_dim = HIDDEN_DIM
        
        # 特征投影层
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, self.unified_dim),
                nn.ReLU(),
                nn.Dropout(drop_rate)
            ) for dim in self.expert_dims
        ])
        
        # 门控网络 - 每个任务一个门控
        self.gates = nn.ModuleDict()
        for task in self.num_classes.keys():
            self.gates[task] = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(self.expert_dims[0], HIDDEN_DIM),  # 使用第一个专家的维度作为输入
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Linear(HIDDEN_DIM, self.num_experts),
                nn.Softmax(dim=1)
            )
        
        # 任务特定的分类头
        self.task_heads = nn.ModuleDict()
        for task, num_classes in self.num_classes.items():
            self.task_heads[task] = nn.Sequential(
                nn.Linear(self.unified_dim, HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Linear(HIDDEN_DIM, num_classes)
            )
        
        # 路由特征提取器（用于RL）
        self.routing_feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.expert_dims[0], HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(drop_rate)
        )
    
    def forward(self, x, task=None, return_features=False):
        batch_size = x.size(0)
        
        # 并行计算所有专家的特征
        expert_features = []
        raw_features = []
        
        for i, expert in enumerate(self.experts):
            # 获取专家特征
            if hasattr(expert, 'forward') and hasattr(expert, 'get_feature_dim'):
                # 使用专家的forward方法
                output, features = expert(x)
                raw_features.append(features)
            else:
                # 直接使用专家提取特征
                features = expert(x)
                raw_features.append(features)
            
            # 投影到统一维度
            projected_features = self.projections[i](features.view(batch_size, -1))
            expert_features.append(projected_features)
        
        # 堆叠专家特征
        expert_features = torch.stack(expert_features, dim=1)  # [batch_size, num_experts, unified_dim]
        
        # 计算门控权重
        if task is None:
            task = 'colon'  # 默认任务
        
        # 使用第一个专家的原始特征计算门控
        gate_input = raw_features[0]
        if len(gate_input.shape) > 2:
            gate_input = F.adaptive_avg_pool2d(gate_input, (1, 1)).flatten(1)
        
        routing_weights = self.gates[task](gate_input.unsqueeze(-1).unsqueeze(-1) if len(gate_input.shape) == 2 else gate_input)
        
        # Top-K 专家选择
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=1)
        top_k_weights = F.softmax(top_k_weights, dim=1)
        
        # 计算加权特征
        selected_features = torch.zeros(batch_size, self.unified_dim, device=x.device)
        
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            weight = top_k_weights[:, i].unsqueeze(1)
            
            # 选择对应专家的特征
            for b in range(batch_size):
                selected_features[b] += weight[b] * expert_features[b, expert_idx[b]]
        
        # 任务特定分类
        output = self.task_heads[task](selected_features)
        
        if return_features:
            # 提取路由特征用于RL
            routing_features = self.routing_feature_extractor(gate_input.unsqueeze(-1).unsqueeze(-1) if len(gate_input.shape) == 2 else gate_input)
            return output, routing_weights, routing_features
        else:
            return output, routing_weights
    
    def calculate_load_balancing_loss(self, routing_weights):
        """计算负载均衡损失"""
        # 计算每个专家的平均使用率
        expert_usage = routing_weights.mean(dim=0)
        
        # 理想情况下每个专家的使用率应该是 1/num_experts
        target_usage = 1.0 / self.num_experts
        
        # 计算KL散度作为负载均衡损失
        load_balancing_loss = F.kl_div(
            torch.log(expert_usage + 1e-8),
            torch.full_like(expert_usage, target_usage),
            reduction='batchmean'
        )
        
        return load_balancing_loss
    
    def get_expert_utilization(self, routing_weights):
        """获取专家使用率统计"""
        with torch.no_grad():
            expert_usage = routing_weights.mean(dim=0)
            max_usage = expert_usage.max().item()
            min_usage = expert_usage.min().item()
            usage_std = expert_usage.std().item()
            
            return {
                'expert_usage': expert_usage.cpu().numpy(),
                'max_usage': max_usage,
                'min_usage': min_usage,
                'usage_std': usage_std,
                'balance_score': 1.0 - usage_std  # 标准差越小，平衡性越好
            }

class CNNExpertWrapper(nn.Module):
    """CNN专家包装器，统一接口"""
    
    def __init__(self, expert_class, num_classes=1000, pretrained=True):
        super(CNNExpertWrapper, self).__init__()
        self.expert = expert_class(num_classes=num_classes, pretrained=pretrained)
        
    def forward(self, x):
        if hasattr(self.expert, 'forward') and len(self.expert.forward.__code__.co_varnames) > 2:
            # 如果forward方法返回多个值
            try:
                output, features = self.expert(x)
                return features
            except:
                output = self.expert(x)
                return output
        else:
            return self.expert(x)
    
    def get_feature_dim(self):
        if hasattr(self.expert, 'get_feature_dim'):
            return self.expert.get_feature_dim()
        else:
            # 推断特征维度
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                try:
                    output = self.forward(dummy_input)
                    if isinstance(output, tuple):
                        return output[1].shape[1]
                    else:
                        return output.shape[1]
                except:
                    return 512  # 默认维度