import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
from config import HIDDEN_DIM, TOP_K_EXPERTS, EXPERT_DROPOUT

# 设置Hugging Face镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class RLHeterogeneousMixtureOfExperts(nn.Module):
    """集成强化学习的异构混合专家模型"""
    
    def __init__(self, model_names, num_colon_classes, num_ugi_classes, 
                 num_colon_disease_classes, num_ugi_disease_classes, 
                 drop_rate=EXPERT_DROPOUT, top_k=TOP_K_EXPERTS):
        super(RLHeterogeneousMixtureOfExperts, self).__init__()
        
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
        
        # 初始化专家网络
        self.experts = nn.ModuleList()
        self.expert_dims = []
        
        # 本地预训练模型路径（仅用于检查）
        local_model_path = os.path.join(os.path.dirname(__file__), "transformer.pth")
        
        for model_name in model_names:
            # 首先创建模型架构，不加载任何预训练权重
            model = None
            try:
                print(f"正在创建模型架构: {model_name}")
                # 创建未预训练的模型架构
                if 'deit' in model_name:
                    model = timm.create_model(
                        model_name, 
                        pretrained=False,  # 不使用在线预训练权重
                        num_classes=0  # 移除分类头，DeiT使用默认的token池化
                    )
                else:
                    model = timm.create_model(
                        model_name, 
                        pretrained=False,  # 不使用在线预训练权重
                        num_classes=0,  # 移除分类头
                        global_pool='avg'  # 使用全局平均池化
                    )
                
                print(f"✅ 成功创建模型架构: {model_name}")
                
                # 注意：预训练权重将在主程序中统一加载
                # 这里不再单独加载每个专家的权重
                
            except Exception as e:
                print(f"❌ 模型架构 {model_name} 创建失败: {e}")
                continue
            
            if model is None:
                continue
            
            # 获取特征维度
            if hasattr(model, 'num_features'):
                feature_dim = model.num_features
            elif hasattr(model, 'fc'):
                if hasattr(model.fc, 'in_features'):
                    feature_dim = model.fc.in_features
                else:
                    feature_dim = HIDDEN_DIM  # 默认值
            else:
                feature_dim = HIDDEN_DIM  # 默认值
            
            self.expert_dims.append(feature_dim)
            self.experts.append(model)
        
        # 验证至少有一个专家加载成功
        if len(self.experts) == 0:
            raise RuntimeError("没有成功加载任何专家模型，请检查模型名称配置")
        
        # 更新专家数量
        self.num_experts = len(self.experts)
        print(f"成功加载 {self.num_experts} 个专家模型")
        print(f"专家模型特征维度: {self.expert_dims}")
        
        # 确保top_k不超过实际专家数量
        self.top_k = min(self.top_k, self.num_experts)
        
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
        # 注意：必须在确定实际专家数量后创建路由网络
        self.gates = nn.ModuleDict()
        for task in self.num_classes.keys():
            self.gates[task] = nn.Sequential(
                nn.Linear(self.router_feature_size, 2 * HIDDEN_DIM),
                nn.LayerNorm(2 * HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(2 * HIDDEN_DIM, self.num_experts)  # 使用实际专家数量
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
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.weight is not None:
                    nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 检查是否有NaN权重
        self._check_nan_weights()
    
    def _check_nan_weights(self):
        """检查模型权重是否包含NaN"""
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print(f"警告：参数 {name} 包含NaN值，重新初始化")
                if len(param.shape) == 1:  # bias
                    nn.init.constant_(param, 0)
                else:  # weight
                    nn.init.normal_(param, 0, 0.01)
    
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
            # 提取特征
            features = expert(x)
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