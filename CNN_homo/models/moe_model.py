import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base_models import BaseModel
from models.expert_models import ExpertNetwork, GatingNetwork, ExpertAttentionGate
from config import NUM_EXPERTS, TOP_K_EXPERTS, HIDDEN_DIM

class MixtureOfExperts(nn.Module):
    """混合专家模型(MoE)架构"""
    
    def __init__(self, base_model_name, num_colon_classes, num_ugi_classes, num_colon_disease_classes, num_ugi_disease_classes,
                 num_experts=NUM_EXPERTS, top_k=TOP_K_EXPERTS, hidden_dim=HIDDEN_DIM,
                 use_attention_gate=True):
        super(MixtureOfExperts, self).__init__()
        
        # 共享编码器
        self.encoder = BaseModel(model_name=base_model_name, pretrained=True)
        self.feature_dim = self.encoder.get_feature_dim()
        
        # 专家网络
        self.experts = nn.ModuleList([
            ExpertNetwork(self.feature_dim, hidden_dim) 
            for _ in range(num_experts)
        ])
        
        # 门控网络
        if use_attention_gate:
            self.gate = ExpertAttentionGate(self.feature_dim, num_experts)
        else:
            self.gate = GatingNetwork(self.feature_dim, num_experts)
        
        # 任务特定分类头
        self.colon_classifier = nn.Linear(hidden_dim // 2, num_colon_classes)
        self.ugi_classifier = nn.Linear(hidden_dim // 2, num_ugi_classes)
        self.colon_disease_classifier = nn.Linear(hidden_dim // 2, num_colon_disease_classes)
        self.ugi_disease_classifier = nn.Linear(hidden_dim // 2, num_ugi_disease_classes)
        
        # 参数设置
        self.num_experts = num_experts
        self.top_k = top_k
    
    def forward(self, x, task=None, **kwargs):
        """
        前向传播
        
        参数:
            x: 输入图像
            task: 任务名称，用于指定特定任务
            **kwargs: 其他关键字参数
        
        返回:
            如果未指定任务，返回所有任务的预测结果和门控权重
            如果指定了任务，返回该任务的预测结果和门控权重
        """
        # 兼容旧的kwargs传递方式和DataParallel
        if task is None:
            task = kwargs.get('task', None)
        
        # 确保task参数正确传递
        if task is None:
            task = 'colon'  # 默认任务
        # 获取图像特征
        features = self.encoder(x, task=task)
        
        # 计算门控权重
        routing_weights = self.gate(features, task=task)
        
        # 选择 top-k 专家
        if self.training:
            # 训练时使用 Gumbel-Softmax 添加噪声，促进探索
            noise = -torch.log(-torch.log(torch.rand_like(routing_weights)))
            routing_weights = routing_weights + noise
            
        # 获取顶部 k 个专家的索引和权重
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=1)
        top_k_weights = F.softmax(top_k_weights, dim=1)
        
        # 准备批处理索引
        batch_size = x.size(0)
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, self.top_k)
        
        # 运行选中的专家
        combined_features = torch.zeros(batch_size, self.experts[0].get_output_dim(), 
                                       device=x.device)
        
        for k in range(self.top_k):
            # 为当前批次中的每个样本获取第k个专家的索引
            expert_indices = top_k_indices[:, k]
            expert_weights = top_k_weights[:, k].unsqueeze(1)
            
            # 获取专家输出
            for i in range(batch_size):
                expert_idx = expert_indices[i].item()
                expert_output = self.experts[expert_idx](features[i:i+1], task=task)
                combined_features[i:i+1] += expert_output * expert_weights[i]
        
        # 应用任务特定分类器
        colon_output = self.colon_classifier(combined_features)
        ugi_output = self.ugi_classifier(combined_features)
        colon_disease_output = self.colon_disease_classifier(combined_features)
        ugi_disease_output = self.ugi_disease_classifier(combined_features)
        
        # 根据指定任务返回结果
        if task == 'colon':
            return colon_output, routing_weights
        elif task == 'ugi':
            return ugi_output, routing_weights
        elif task == 'colon_disease':
            return colon_disease_output, routing_weights
        elif task == 'ugi_disease':
            return ugi_disease_output, routing_weights
        else:
            return {
                'colon': colon_output,
                'ugi': ugi_output,
                'colon_disease': colon_disease_output,
                'ugi_disease': ugi_disease_output
            }, routing_weights
    
    def calculate_load_balancing_loss(self, routing_weights):
        """计算负载均衡损失，确保专家使用均衡"""
        # 计算每个专家的使用频率
        expert_usage = torch.zeros(self.num_experts, device=routing_weights.device)
        
        for i in range(routing_weights.size(0)):
            _, indices = torch.topk(routing_weights[i], self.top_k)
            for idx in indices:
                expert_usage[idx] += 1
        
        # 归一化使用频率
        expert_usage = expert_usage / torch.sum(expert_usage)
        
        # 理想情况下，每个专家的使用频率应该是均匀的
        target_usage = torch.ones_like(expert_usage) / self.num_experts
        
        # 计算KL散度作为负载均衡损失
        load_balancing_loss = F.kl_div(
            torch.log(expert_usage + 1e-10),
            target_usage,
            reduction='batchmean'
        )
        
        return load_balancing_loss