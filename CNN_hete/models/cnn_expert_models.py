import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights, DenseNet121_Weights, MobileNet_V2_Weights
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attention_modules import CBAM, SEModule, RegionAttentionModule


class ResidualCNN(nn.Module):
    """专家1: ResNet50 - 残差CNN网络
    
    参数：
    - 模型：torchvision.models.resnet50(pretrained=True)
    - 架构：[3,4,6,3] Bottleneck blocks
    - 空洞卷积：layer3,4使用dilation=[2,4]
    - 输入尺寸：224×224×3
    - 输出特征维：2048
    - 注意力：CBAM模块
    """
    
    def __init__(self, num_classes=1000, pretrained=True):
        super(ResidualCNN, self).__init__()
        
        # 加载预训练的ResNet50
        if pretrained:
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # 修改layer3和layer4使用空洞卷积
        self._modify_dilated_conv()
        
        # 添加CBAM注意力模块
        self.cbam = CBAM(2048, reduction_ratio=16, kernel_size=7)
        
        # 移除原始分类器
        self.backbone.fc = nn.Identity()
        
        # 添加新的分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4
        )
        
    def _modify_dilated_conv(self):
        """修改layer3和layer4使用空洞卷积"""
        # Layer3: dilation=2
        for n, m in self.backbone.layer3.named_modules():
            if 'conv2' in n:
                m.dilation = (2, 2)
                m.padding = (2, 2)
        
        # Layer4: dilation=4
        for n, m in self.backbone.layer4.named_modules():
            if 'conv2' in n:
                m.dilation = (4, 4)
                m.padding = (4, 4)
    
    def forward(self, x):
        # 特征提取
        features = self.feature_extractor(x)
        
        # 应用CBAM注意力
        features = self.cbam(features)
        
        # 分类
        output = self.classifier(features)
        
        return output, features
    
    def get_feature_dim(self):
        """返回特征维度"""
        return 2048


class DenseNetCNN(nn.Module):
    """专家2: DenseNet121 内存优化版
    
    参数：
    - 模型：DenseNet121
    - 增长率：32
    - 块配置：[6,12,24,16]
    - 初始特征：64
    - Dropout率：0.1
    - 输出特征维：1024
    - 内存优化：启用
    """
    
    def __init__(self, num_classes=1000, pretrained=True, memory_efficient=True):
        super(DenseNetCNN, self).__init__()
        
        # 加载预训练的DenseNet121
        if pretrained:
            self.backbone = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1, 
                                             memory_efficient=memory_efficient)
        else:
            self.backbone = models.densenet121(weights=None, 
                                             memory_efficient=memory_efficient)
        
        # 获取特征维度
        self.feature_dim = self.backbone.classifier.in_features
        
        # 移除原始分类器
        self.backbone.classifier = nn.Identity()
        
        # 添加新的分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # 特征提取
        features = self.backbone.features(x)
        features = F.relu(features, inplace=True)
        
        # 分类
        output = self.classifier(features)
        
        return output, features
    
    def get_feature_dim(self):
        """返回特征维度"""
        return self.feature_dim


class AttentionCNN(nn.Module):
    """专家3: 多重注意力网络
    
    参数：
    - 骨干网络：ResNet34-style
    - 注意力模块：CBAM + SE + RegionAttention
    - 通道注意力比率：16
    - 空间核大小：7
    - 区域数量：4
    - 输出特征维：512
    """
    
    def __init__(self, num_classes=1000, pretrained=True):
        super(AttentionCNN, self).__init__()
        
        # 加载ResNet34作为骨干网络
        if pretrained:
            backbone = models.resnet34(pretrained=True)
        else:
            backbone = models.resnet34(pretrained=False)
        
        # 提取特征层
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # 多重注意力模块
        self.cbam = CBAM(512, reduction_ratio=16, kernel_size=7)
        self.se_module = SEModule(512, reduction=16)
        self.region_attention = RegionAttentionModule(512, num_regions=4)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # 基础特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.layer4(x)
        
        # 多重注意力机制
        features = self.cbam(features)
        features = self.se_module(features)
        features = self.region_attention(features)
        
        # 分类
        output = self.classifier(features)
        
        return output, features
    
    def get_feature_dim(self):
        """返回特征维度"""
        return 512


class DepthwiseCNN(nn.Module):
    """专家4: 轻量化实时网络
    
    参数：
    - 模型：MobileNetV2
    - 宽度倍数：1.0
    - 扩展因子：6
    - 输入分辨率：224×224
    - 输出特征维：1280
    - 优化目标：实时推理
    """
    
    def __init__(self, num_classes=1000, pretrained=True):
        super(DepthwiseCNN, self).__init__()
        
        # 加载预训练的MobileNetV2
        if pretrained:
            self.backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.mobilenet_v2(weights=None)
        
        # 移除原始分类器
        self.backbone.classifier = nn.Identity()
        
        # 添加新的分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, 640),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(640, num_classes)
        )
        
    def forward(self, x):
        # 特征提取
        features = self.backbone.features(x)
        
        # 分类
        output = self.classifier(features)
        
        return output, features
    
    def get_feature_dim(self):
        """返回特征维度"""
        return 1280


class PyramidCNN(nn.Module):
    """专家5: 多尺度金字塔网络
    
    参数：
    - 骨干网络：ResNet34
    - 金字塔尺度：[1,2,3,6]
    - FPN通道数：256
    - FPN层级：[P2,P3,P4,P5]
    - 多核尺寸：[3,5,7]
    - 输出特征维：512
    """
    
    def __init__(self, num_classes=1000, pretrained=True):
        super(PyramidCNN, self).__init__()
        
        # 加载ResNet34作为骨干网络
        if pretrained:
            backbone = models.resnet34(pretrained=True)
        else:
            backbone = models.resnet34(pretrained=False)
        
        # 提取特征层
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # 64 channels
        self.layer2 = backbone.layer2  # 128 channels
        self.layer3 = backbone.layer3  # 256 channels
        self.layer4 = backbone.layer4  # 512 channels
        
        # 金字塔池化模块
        self.pyramid_pool = PyramidPoolingModule(512, scales=[1, 2, 3, 6])
        
        # FPN模块
        self.fpn = FeaturePyramidNetwork([64, 128, 256, 512], 256)
        
        # 多核卷积
        self.multi_kernel_conv = MultiKernelConv(256, 256, kernels=[3, 5, 7])
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # 多层特征提取
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)
        c1 = self.maxpool(c1)
        
        c2 = self.layer1(c1)  # P2
        c3 = self.layer2(c2)  # P3
        c4 = self.layer3(c3)  # P4
        c5 = self.layer4(c4)  # P5
        
        # 金字塔池化
        pooled_features = self.pyramid_pool(c5)
        
        # FPN特征融合
        fpn_features = self.fpn([c2, c3, c4, c5])
        
        # 多核卷积
        final_features = self.multi_kernel_conv(fpn_features[-1])
        
        # 分类
        output = self.classifier(final_features)
        
        return output, final_features
    
    def get_feature_dim(self):
        """返回特征维度"""
        return 256


class DilatedCNN(nn.Module):
    """专家6: 空洞卷积分割网络
    
    参数：
    - 模型：DeepLabV3+ ASPP
    - 骨干网络：ResNet50 (output_stride=16)
    - 空洞率：[1,6,12,18]
    - ASPP通道数：256
    - 解码器通道数：256
    - 输出特征维：256
    """
    
    def __init__(self, num_classes=1000, pretrained=True):
        super(DilatedCNN, self).__init__()
        
        # 加载ResNet50作为骨干网络
        if pretrained:
            backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            backbone = models.resnet50(weights=None)
        
        # 提取特征层
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # 修改layer3和layer4的步长以保持分辨率
        self._modify_backbone_stride()
        
        # ASPP模块
        self.aspp = ASPP(2048, 256, dilations=[1, 6, 12, 18])
        
        # 解码器
        self.decoder = Decoder(256, 256)  # ResNet50 layer1 output has 256 channels
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def _modify_backbone_stride(self):
        """修改骨干网络的步长以保持分辨率"""
        # 修改layer3: stride=1, dilation=2
        for n, m in self.layer3.named_modules():
            if 'conv1' in n and hasattr(m, 'stride'):
                m.stride = (1, 1)
            elif 'conv2' in n and hasattr(m, 'stride'):
                m.stride = (1, 1)
                m.dilation = (2, 2)
                m.padding = (2, 2)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        
        # 修改layer4: stride=1, dilation=4  
        for n, m in self.layer4.named_modules():
            if 'conv1' in n and hasattr(m, 'stride'):
                m.stride = (1, 1)
            elif 'conv2' in n and hasattr(m, 'stride'):
                m.stride = (1, 1)
                m.dilation = (4, 4)
                m.padding = (4, 4)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
    
    def forward(self, x):
        # 特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        low_level_features = x
        
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # ASPP处理
        aspp_features = self.aspp(x)
        
        # 解码器处理
        decoded_features = self.decoder(aspp_features, low_level_features)
        
        # 分类
        output = self.classifier(decoded_features)
        
        return output, decoded_features
    
    def get_feature_dim(self):
        """返回特征维度"""
        return 256


# 辅助模块
class PyramidPoolingModule(nn.Module):
    """金字塔池化模块"""
    
    def __init__(self, in_channels, scales=[1, 2, 3, 6]):
        super(PyramidPoolingModule, self).__init__()
        self.scales = scales
        self.pools = nn.ModuleList()
        self.convs = nn.ModuleList()
        
        for scale in scales:
            self.pools.append(nn.AdaptiveAvgPool2d(scale))
            self.convs.append(nn.Conv2d(in_channels, in_channels // len(scales), 1))
    
    def forward(self, x):
        b, c, h, w = x.size()
        pooled_features = []
        
        for pool, conv in zip(self.pools, self.convs):
            pooled = pool(x)
            pooled = conv(pooled)
            pooled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
            pooled_features.append(pooled)
        
        return torch.cat([x] + pooled_features, dim=1)


class FeaturePyramidNetwork(nn.Module):
    """特征金字塔网络"""
    
    def __init__(self, in_channels_list, out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv2d(in_channels, out_channels, 1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
    
    def forward(self, inputs):
        laterals = [conv(inputs[i]) for i, conv in enumerate(self.lateral_convs)]
        
        # 自顶向下的特征融合
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:], mode='bilinear', align_corners=False
            )
        
        # 应用FPN卷积
        fpn_outs = [conv(laterals[i]) for i, conv in enumerate(self.fpn_convs)]
        
        return fpn_outs


class MultiKernelConv(nn.Module):
    """多核卷积模块"""
    
    def __init__(self, in_channels, out_channels, kernels=[3, 5, 7]):
        super(MultiKernelConv, self).__init__()
        self.convs = nn.ModuleList()
        
        # 确保通道数能被均匀分配
        num_kernels = len(kernels)
        channels_per_kernel = out_channels // num_kernels
        remaining_channels = out_channels % num_kernels
        
        for i, kernel in enumerate(kernels):
            padding = kernel // 2
            # 给最后一个卷积分配剩余的通道数
            current_channels = channels_per_kernel + (remaining_channels if i == num_kernels - 1 else 0)
            self.convs.append(nn.Conv2d(in_channels, current_channels, kernel, padding=padding))
        
        self.fusion = nn.Conv2d(out_channels, out_channels, 1)
    
    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        output = torch.cat(outputs, dim=1)
        return self.fusion(output)


class ASPP(nn.Module):
    """空洞空间金字塔池化模块"""
    
    def __init__(self, in_channels, out_channels, dilations=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.convs = nn.ModuleList()
        
        # 1x1卷积
        self.convs.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # 空洞卷积
        for dilation in dilations[1:]:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        # 全局平均池化
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        h, w = x.size()[-2:]
        
        # 应用不同的空洞卷积
        features = []
        for conv in self.convs:
            features.append(conv(x))
        
        # 全局平均池化
        global_feature = self.global_avg_pool(x)
        global_feature = F.interpolate(global_feature, size=(h, w), mode='bilinear', align_corners=False)
        features.append(global_feature)
        
        # 融合所有特征
        output = torch.cat(features, dim=1)
        output = self.fusion(output)
        
        return output


class Decoder(nn.Module):
    """解码器模块"""
    
    def __init__(self, aspp_channels, low_level_channels):
        super(Decoder, self).__init__()
        
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.last_conv = nn.Sequential(
            nn.Conv2d(aspp_channels + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, aspp_features, low_level_features):
        # 处理低级特征
        low_level_features = self.low_level_conv(low_level_features)
        
        # 上采样ASPP特征
        aspp_features = F.interpolate(aspp_features, size=low_level_features.size()[-2:], 
                                     mode='bilinear', align_corners=False)
        
        # 融合特征
        features = torch.cat([aspp_features, low_level_features], dim=1)
        output = self.last_conv(features)
        
        return output