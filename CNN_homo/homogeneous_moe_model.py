import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from config import HIDDEN_DIM, TOP_K_EXPERTS, EXPERT_DROPOUT

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入注意力模块
from attention_modules import (
    CBAM, SEModule, NonLocalBlock, RegionAttentionModule, 
    FineGrainedAttention, MultiPathEnhancementModule
)

class ResNet50Expert(nn.Module):
    """ResNet-50专家网络
    
    同构专家模型的基础单元，所有专家都使用相同的ResNet-50架构，
    但具有不同的初始化权重以保持多样性。
    """
    
    def __init__(self, expert_id=0, num_classes=0, pretrained=True):
        super(ResNet50Expert, self).__init__()
        
        self.expert_id = expert_id
        
        # 创建ResNet-50基础架构
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # 获取特征维度（ResNet-50的fc层输入维度）
        self.feature_dim = self.backbone.fc.in_features
        
        # 移除原始分类层
        self.backbone.fc = nn.Identity()
        
        # 为不同专家添加轻微的架构差异以增加多样性
        self._add_expert_specific_modules()
        
        # 如果需要分类输出，添加分类头
        if num_classes > 0:
            self.classifier = nn.Linear(self.feature_dim, num_classes)
        else:
            self.classifier = None
    
    def _add_expert_specific_modules(self):
        """为不同专家添加特定的模块以增加多样性"""
        
        # 为不同专家添加不同的注意力机制
        if self.expert_id % 3 == 0:
            # 专家0,3: 添加CBAM注意力
            self.attention_module = CBAM(self.backbone.layer4[-1].conv3.out_channels)
        elif self.expert_id % 3 == 1:
            # 专家1,4: 添加SE注意力
            self.attention_module = SEModule(self.backbone.layer4[-1].conv3.out_channels)
        else:
            # 专家2,5: 添加非局部注意力
            self.attention_module = NonLocalBlock(self.backbone.layer4[-1].conv3.out_channels)
        
        # 在ResNet最后一层后插入注意力模块
        original_layer4 = self.backbone.layer4
        self.backbone.layer4 = nn.Sequential(
            original_layer4,
            self.attention_module
        )
        
        # 为每个专家添加独特的dropout率以增加多样性
        dropout_rates = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
        self.expert_dropout = nn.Dropout(dropout_rates[self.expert_id % len(dropout_rates)])
    
    def get_feature_dim(self):
        """返回特征维度"""
        return self.feature_dim
    
    def forward(self, x):
        """前向传播"""
        # 通过ResNet骨干网络
        features = self.backbone(x)
        
        # 应用专家特定的dropout
        features = self.expert_dropout(features)
        
        # 如果有分类器，计算分类输出
        if self.classifier is not None:
            outputs = self.classifier(features)
            return outputs, features
        else:
            return None, features

class HomogeneousMixtureOfExperts(nn.Module):
    """
    同构混合专家模型 - ResNet-50版本
    
    使用多个相同架构（ResNet-50）但不同初始化的专家网络，
    通过门控机制动态选择最适合当前输入的专家组合。
    """
    
    def __init__(self, num_experts=6, num_colon_classes=10, num_ugi_classes=10, 
                 num_colon_disease_classes=10, num_ugi_disease_classes=10, 
                 drop_rate=EXPERT_DROPOUT, top_k=TOP_K_EXPERTS):
        super(HomogeneousMixtureOfExperts, self).__init__()
        
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # 各任务的类别数目
        self.num_classes = {
            'colon': num_colon_classes,
            'ugi': num_ugi_classes,
            'colon_disease': num_colon_disease_classes,
            'ugi_disease': num_ugi_disease_classes
        }
        
        # 构建同构专家网络集合
        self.experts = nn.ModuleList()
        self.expert_dims = []
        
        print(f"正在创建 {num_experts} 个ResNet-50同构专家...")
        
        for i in range(num_experts):
            # 创建ResNet-50专家，每个专家有不同的初始化
            expert = ResNet50Expert(expert_id=i, num_classes=0, pretrained=True)
            
            # 为不同专家应用不同的初始化策略以增加多样性
            self._initialize_expert_diversity(expert, i)
            
            feature_dim = expert.get_feature_dim()
            self.expert_dims.append(feature_dim)
            self.experts.append(expert)
            
            print(f"✅ 已创建专家 {i+1}/{num_experts}: ResNet-50 (特征维度: {feature_dim})")
        
        print(f"同构专家模型创建完成，共 {self.num_experts} 个ResNet-50专家")
        
        # 为每个专家创建特征投影层，统一特征维度
        self.projections = nn.ModuleList()
        for i, dim in enumerate(self.expert_dims):
            # 为不同专家使用略有不同的投影层以增加多样性
            if i % 2 == 0:
                # 偶数专家：使用标准投影
                self.projections.append(
                    nn.Sequential(
                        nn.Linear(dim, HIDDEN_DIM),
                        nn.LayerNorm(HIDDEN_DIM),
                        nn.GELU(),
                        nn.Dropout(drop_rate)
                    )
                )
            else:
                # 奇数专家：使用增强投影
                self.projections.append(
                    nn.Sequential(
                        nn.Linear(dim, HIDDEN_DIM),
                        nn.LayerNorm(HIDDEN_DIM),
                        nn.GELU(),
                        nn.Dropout(drop_rate),
                        FineGrainedAttention(HIDDEN_DIM),
                        MultiPathEnhancementModule(HIDDEN_DIM)
                    )
                )
        
        # 为Colon任务添加特殊的增强模块
        self.colon_enhancer = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(drop_rate * 0.5)
        )
        
        # 细粒度区分模块
        self.fine_grained_module = RegionAttentionModule(HIDDEN_DIM, num_regions=4)
        
        # 为每个任务创建分类头
        self.classifiers = nn.ModuleDict({
            'colon': nn.Sequential(
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
                nn.LayerNorm(HIDDEN_DIM // 2),
                nn.ReLU(inplace=True),
                nn.Linear(HIDDEN_DIM // 2, num_colon_classes)
            ),
            'ugi': nn.Linear(HIDDEN_DIM, num_ugi_classes),
            'colon_disease': nn.Linear(HIDDEN_DIM, num_colon_disease_classes),
            'ugi_disease': nn.Linear(HIDDEN_DIM, num_ugi_disease_classes)
        })
        
        # 为每个任务创建路由网络（门控）
        self.gates = nn.ModuleDict()
        for task in self.num_classes.keys():
            if task == 'colon':
                # 为Colon任务使用更复杂的门控网络
                self.gates[task] = nn.Sequential(
                    nn.Linear(3 * HIDDEN_DIM, 2 * HIDDEN_DIM),
                    nn.LayerNorm(2 * HIDDEN_DIM),
                    nn.GELU(),
                    nn.Dropout(drop_rate),
                    nn.Linear(2 * HIDDEN_DIM, HIDDEN_DIM),
                    nn.LayerNorm(HIDDEN_DIM),
                    nn.GELU(),
                    nn.Linear(HIDDEN_DIM, self.num_experts)
                )
            else:
                self.gates[task] = nn.Sequential(
                    nn.Linear(3 * HIDDEN_DIM, 2 * HIDDEN_DIM),
                    nn.LayerNorm(2 * HIDDEN_DIM),
                    nn.GELU(),
                    nn.Dropout(drop_rate),
                    nn.Linear(2 * HIDDEN_DIM, self.num_experts)
                )
        
        # 路由网络的输入特征提取器
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
            ) for _ in range(3)
        ])
        
        # Colon任务特殊特征提取器
        self.colon_feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CBAM(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            NonLocalBlock(128),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(drop_rate * 0.5)
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
    
    def _initialize_expert_diversity(self, expert, expert_id):
        """为不同专家初始化不同的权重以增加多样性"""
        
        # 设置不同的随机种子
        torch.manual_seed(42 + expert_id * 100)
        
        # 对专家的某些层进行重新初始化以增加多样性
        for name, module in expert.named_modules():
            if isinstance(module, nn.Conv2d) and 'layer4' in name:
                # 对最后一层的卷积层应用不同的初始化
                if expert_id % 2 == 0:
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.Linear) and hasattr(module, 'weight'):
                # 对线性层应用不同的初始化
                scale = 1.0 + expert_id * 0.1  # 不同专家使用不同的初始化尺度
                nn.init.normal_(module.weight, 0, 0.01 * scale)
        
        # 恢复全局随机种子
        torch.manual_seed(torch.initial_seed())
    
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
    
    def forward(self, x, task='colon'):
        """前向传播"""
        batch_size = x.size(0)
        
        # 使用特征提取器生成路由输入特征
        router_feats = []
        for extractor in self.feature_extractors:
            router_feats.append(extractor(x))
        
        # 为Colon任务添加额外的细粒度特征
        if task == 'colon':
            colon_feat = self.colon_feature_extractor(x)
            router_feats = router_feats[:2] + [colon_feat]
        
        # 连接所有路由器特征
        router_input = torch.cat(router_feats, dim=1)
        
        # 获取路由权重（专家分配）
        routing_logits = self.gates[task](router_input)
        routing_weights = F.softmax(routing_logits, dim=1)
        
        # 选择前k个专家
        weights, indices = torch.topk(routing_weights, self.top_k, dim=1)
        weights = weights / weights.sum(dim=1, keepdim=True)
        
        # 收集专家输出
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # 提取特征
            output, features = expert(x)
            # 如果特征是4D张量，进行全局平均池化
            if len(features.shape) == 4:
                features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
            # 投影到统一维度
            features = self.projections[i](features)
            expert_outputs.append(features)
        
        # 将expert_outputs堆叠为张量
        expert_outputs = torch.stack(expert_outputs)
        
        # 初始化组合特征
        combined_features = torch.zeros(batch_size, HIDDEN_DIM).to(x.device)
        
        # 使用重要性权重调整专家输出
        weighted_expert_outputs = expert_outputs * self.expert_importance.view(-1, 1, 1)
        
        # 对每个样本选择并加权组合前k个专家的输出
        for b in range(batch_size):
            for k in range(self.top_k):
                expert_idx = indices[b, k].item()
                weight = weights[b, k]
                combined_features[b] += weighted_expert_outputs[expert_idx, b] * weight
        
        # 为Colon任务应用特殊处理
        if task == 'colon':
            # 增强Colon特征
            combined_features = self.colon_enhancer(combined_features)
            
            # 应用区域注意力模块
            batch_size, channels = combined_features.shape
            reshaped_features = combined_features.view(batch_size, channels, 1, 1)
            reshaped_features = F.interpolate(reshaped_features, size=(7, 7), mode='bilinear', align_corners=False)
            enhanced_features = self.fine_grained_module(reshaped_features)
            combined_features = F.adaptive_avg_pool2d(enhanced_features, (1, 1)).view(batch_size, channels)
        
        # 应用任务特定分类头
        outputs = self.classifiers[task](combined_features)
        
        return outputs, routing_weights
    
    def calculate_load_balancing_loss(self, routing_weights):
        """计算专家负载均衡损失"""
        # 计算每个专家的期望使用频率
        expected_frequency = torch.ones(self.num_experts) / self.num_experts
        expected_frequency = expected_frequency.to(routing_weights.device)
        
        # 计算实际使用频率
        actual_frequency = routing_weights.mean(dim=0)
        
        # 计算KL散度损失
        kl_loss = F.kl_div(
            actual_frequency.log(), 
            expected_frequency.expand_as(actual_frequency),
            reduction='batchmean'
        )
        
        # 添加方差损失
        variance_loss = torch.var(actual_frequency)
        
        # 计算变异系数
        importance = routing_weights.sum(dim=0)
        cv_squared = torch.var(importance) / (torch.mean(importance) ** 2)
        
        # 总负载均衡损失
        load_balancing_loss = kl_loss + variance_loss + cv_squared
        
        return load_balancing_loss
    
    def get_expert_usage_stats(self, routing_weights):
        """获取专家使用统计信息"""
        with torch.no_grad():
            usage_frequency = routing_weights.mean(dim=0)
            max_usage = torch.max(usage_frequency)
            min_usage = torch.min(usage_frequency)
            usage_variance = torch.var(usage_frequency)
            
            stats = {
                'usage_frequency': usage_frequency.cpu().numpy(),
                'max_usage': max_usage.item(),
                'min_usage': min_usage.item(),
                'usage_variance': usage_variance.item(),
                'usage_balance_ratio': min_usage.item() / max_usage.item()
            }
            
            return stats