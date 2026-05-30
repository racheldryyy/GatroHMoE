import torch
import torch.nn as nn
import torch.nn.functional as F
from config import HIDDEN_DIM

class ExpertNetwork(nn.Module):
    """专家网络"""
    
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, output_dim=None):
        super(ExpertNetwork, self).__init__()
        
        self.output_dim = hidden_dim // 2  # 存储输出维度
        
        self.expert_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),  # 使用LayerNorm替代BatchNorm
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, self.output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.output_dim),  # 使用LayerNorm替代BatchNorm
            nn.Dropout(0.2)
        )
        
        # 如果指定了输出维度，添加分类层
        if output_dim is not None:
            self.classifier = nn.Linear(self.output_dim, output_dim)
        else:
            self.classifier = None
    
    def forward(self, x, task=None):  # 修改：添加task参数
        """前向传播"""
        features = self.expert_layers(x)
        
        if self.classifier is not None:
            return self.classifier(features), features
        else:
            return features
            
    def get_output_dim(self):
        """返回特征输出维度"""
        return self.output_dim


class GatingNetwork(nn.Module):
    """门控网络"""
    
    def __init__(self, input_dim, num_experts, hidden_dim=HIDDEN_DIM // 4):
        super(GatingNetwork, self).__init__()
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),  # 使用LayerNorm替代BatchNorm
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_experts)
        )
    
    def forward(self, x, task=None):  # 修改：添加task参数
        """前向传播"""
        return self.gate(x)


class ExpertAttentionGate(nn.Module):
    """基于注意力机制的专家门控网络"""
    
    def __init__(self, input_dim, num_experts, hidden_dim=HIDDEN_DIM // 4):
        super(ExpertAttentionGate, self).__init__()
        
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts)
        )
    
    def forward(self, x, task=None):  # 修改：添加task参数
        """基于自注意力的门控"""
        q = self.query(x).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        k = self.key(x).unsqueeze(1)    # [batch_size, 1, hidden_dim]
        v = self.value(x).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        attn_output, _ = self.attention(q, k, v)
        attn_output = attn_output.squeeze(1)  # [batch_size, hidden_dim]
        
        gates = self.output_layer(attn_output)  # [batch_size, num_experts]
        
        return gates