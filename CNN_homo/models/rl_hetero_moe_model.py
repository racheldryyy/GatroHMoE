import torch
import torch.nn as nn
import torch.nn.functional as F
from config import HIDDEN_DIM, TOP_K_EXPERTS, EXPERT_DROPOUT

# 导入新的CNN专家模型
from models.cnn_expert_models import (
    ResidualCNN, DenseNetCNN, AttentionCNN, 
    DepthwiseCNN, PyramidCNN, DilatedCNN
)

class RLHeterogeneousMixtureOfExperts(nn.Module):
    """集成强化学习的异构混合专家模型"""
    
    def __init__(self, num_colon_classes, num_ugi_classes, 
                 num_colon_disease_classes, num_ugi_disease_classes, 
                 drop_rate=EXPERT_DROPOUT, top_k=TOP_K_EXPERTS):
        super(RLHeterogeneousMixtureOfExperts, self).__init__()
        
        self.expert_names = ['ResNet50', 'DenseNet121', 'AttentionCNN', 'MobileNetV2', 'PyramidCNN', 'DilatedCNN']
        self.num_experts = len(self.expert_names)
        self.top_k = top_k
        
        # 保存类别数量
        self.num_classes = {
            'colon': num_colon_classes,
            'ugi': num_ugi_classes,
            'colon_disease': num_colon_disease_classes,
            'ugi_disease': num_ugi_disease_classes
        }
        
        # 初始化专家网络 - 使用新的CNN专家模型
        self.experts = nn.ModuleList()
        self.expert_dims = []
        
        # 定义六个CNN专家模型
        expert_models = [
            ResidualCNN(num_classes=0, pretrained=True),      # 专家1: ResNet50
            DenseNetCNN(num_classes=0, pretrained=True),      # 专家2: DenseNet121
            AttentionCNN(num_classes=0, pretrained=True),     # 专家3: 多重注意力网络
            DepthwiseCNN(num_classes=0, pretrained=True),     # 专家4: MobileNetV2
            PyramidCNN(num_classes=0, pretrained=True),       # 专家5: 多尺度金字塔网络
            DilatedCNN(num_classes=0, pretrained=True)        # 专家6: 空洞卷积网络
        ]
        
        for i, model in enumerate(expert_models):
            # 获取模型的特征输出维度
            feature_dim = model.get_feature_dim()
            self.expert_dims.append(feature_dim)
            self.experts.append(model)
            
        # 确保专家数量匹配
        self.num_experts = len(self.experts)
        
        print(f"专家模型特征维度: {self.expert_dims}")
        
        # 为每个专家创建特征投影层，以统一特征维度
        self.projections = nn.ModuleList()
        for dim in self.expert_dims:
            self.projections.append(
                nn.Sequential(
                    nn.Linear(dim, HIDDEN_DIM),
                    nn.LayerNorm(HIDDEN_DIM),
                    nn.GELU(),
                    nn.Dropout(drop_rate)
                )
            )
        
        # 为每个任务创建分类头
        self.classifiers = nn.ModuleDict({
            'colon': nn.Linear(HIDDEN_DIM, num_colon_classes),
            'ugi': nn.Linear(HIDDEN_DIM, num_ugi_classes),
            'colon_disease': nn.Linear(HIDDEN_DIM, num_colon_disease_classes),
            'ugi_disease': nn.Linear(HIDDEN_DIM, num_ugi_disease_classes)
        })
        
        # 路由特征提取器
        self.router_feature_size = 3 * HIDDEN_DIM
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(32 * 7 * 7, HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(drop_rate)
            ) for _ in range(3)  # 使用三个特征提取器以捕获不同尺度特征
        ])
        
        # 为每个任务创建路由网络（策略网络）
        self.gates = nn.ModuleDict()
        for task in self.num_classes.keys():
            self.gates[task] = nn.Sequential(
                nn.Linear(self.router_feature_size, 2 * HIDDEN_DIM),
                nn.LayerNorm(2 * HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(2 * HIDDEN_DIM, self.num_experts)
            )
        
        # 任务嵌入
        self.task_embeddings = nn.ParameterDict({
            task: nn.Parameter(torch.randn(HIDDEN_DIM))
            for task in self.num_classes.keys()
        })
        
        # 专家重要性权重
        self.expert_importance = nn.Parameter(torch.ones(self.num_experts))
        
        # 初始化参数
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, task='colon', return_features=False):
        """前向传播"""
        batch_size = x.size(0)
        
        # 获取路由特征
        router_feats = []
        for extractor in self.feature_extractors:
            router_feats.append(extractor(x))
        
        # 构建路由器输入（连接所有特征）
        router_input = torch.cat(router_feats, dim=1)
        
        # 获取策略网络输出
        routing_logits = self.gates[task](router_input)
        
        # 如果是推理模式，直接应用softmax获取路由权重
        if not self.training:
            routing_weights = F.softmax(routing_logits, dim=1)
        else:
            # 训练模式下，添加温度参数使分布更平滑
            temperature = 1.0
            routing_weights = F.softmax(routing_logits / temperature, dim=1)
        
        # 选择前k个专家
        weights, indices = torch.topk(routing_weights, self.top_k, dim=1)
        
        # 确保专家索引在有效范围内
        indices = torch.clamp(indices, 0, self.num_experts - 1)
        
        # 重新归一化权重，使其总和为1
        weights = weights / weights.sum(dim=1, keepdim=True)
        
        # 收集专家输出
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # 提取特征 - CNN专家模型返回输出和特征
            output, features = expert(x)
            # 如果特征是4D张量，进行全局平均池化
            if len(features.shape) == 4:
                features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
            # 投影到统一维度
            features = self.projections[i](features)
            expert_outputs.append(features)
        
        # 将expert_outputs堆叠为张量，形状为[num_experts, batch_size, hidden_dim]
        expert_outputs = torch.stack(expert_outputs)
        
        # 初始化组合特征
        combined_features = torch.zeros(batch_size, HIDDEN_DIM).to(x.device)
        
        # 使用重要性权重调整专家输出
        weighted_expert_outputs = expert_outputs * self.expert_importance.view(-1, 1, 1)
        
        # 对每个样本选择并加权组合前k个专家的输出
        for b in range(batch_size):
            for k in range(self.top_k):
                expert_idx = indices[b, k].item()
                # 添加边界检查，防止索引越界
                if expert_idx >= len(self.experts):
                    print(f"警告：专家索引 {expert_idx} 超出范围 [0, {len(self.experts)-1}]，跳过")
                    continue
                weight = weights[b, k]
                combined_features[b] += weighted_expert_outputs[expert_idx, b] * weight
        
        # 应用任务特定分类头
        outputs = self.classifiers[task](combined_features)
        
        if return_features:
            # 返回路由特征以供强化学习训练使用
            return outputs, routing_weights, router_input
        else:
            return outputs, routing_weights
    
    def calculate_load_balancing_loss(self, routing_weights):
        """计算专家负载均衡损失"""
        # 计算每个专家的期望使用频率（应该是均匀分布）
        expected_frequency = torch.ones(self.num_experts) / self.num_experts
        expected_frequency = expected_frequency.to(routing_weights.device)
        
        # 计算当前批次中每个专家的实际使用频率
        actual_frequency = routing_weights.mean(dim=0)
        
        # 计算KL散度损失，鼓励均匀使用专家
        kl_loss = F.kl_div(
            actual_frequency.log(), 
            expected_frequency.expand_as(actual_frequency),
            reduction='batchmean'
        )
        
        # 添加方差损失
        variance_loss = torch.var(actual_frequency)
        
        # 计算保留率，即每个专家被选中的比例
        importance = routing_weights.sum(dim=0)
        # CV = coefficient of variation = std / mean
        cv_squared = torch.var(importance) / (torch.mean(importance) ** 2)
        
        # 总负载均衡损失
        load_balancing_loss = kl_loss + variance_loss + cv_squared
        
        return load_balancing_loss