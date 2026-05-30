import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChannelAttention(nn.Module):
    """
    通道注意力机制
    
    通过全局平均池化和最大池化获取通道间的依赖关系，
    自适应地调整各通道的重要性权重。
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
        
class SpatialAttention(nn.Module):
    """
    空间注意力机制
    
    聚焦于特征图中最具判别性的空间位置，
    增强重要区域的特征表示。
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "卷积核大小必须为3或7"
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 计算空间维度的统计信息
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """
    卷积块注意力模块
    
    结合通道注意力和空间注意力，双重增强特征表示能力，
    在医学图像分析中特别有效。
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

class SEModule(nn.Module):
    """
    Squeeze-and-Excitation 注意力模块
    
    通过压缩-激励操作学习通道间的相互依赖关系，
    提高网络对重要特征通道的敏感性。
    """
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class NonLocalBlock(nn.Module):
    """非局部自注意力块"""
    def __init__(self, in_channels, reduction_ratio=2):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // reduction_ratio
        
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.W = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
        
        self.bn = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W(y)
        z = self.bn(W_y) + x
        
        return z

class RegionAttentionModule(nn.Module):
    """区域关注模块,专注于细粒度特征区分"""
    def __init__(self, in_channels, num_regions=4):
        super(RegionAttentionModule, self).__init__()
        self.num_regions = num_regions
        
        # 生成区域掩码的网络
        self.mask_gen = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_regions, kernel_size=1)
        )
        
        # 区域特征增强
        self.region_enhance = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.LayerNorm([in_channels, 7, 7]),  # 假设特征图为7x7
                nn.ReLU(inplace=True)
            ) for _ in range(num_regions)
        ])
        
    def forward(self, x):
        # 生成区域掩码
        masks = self.mask_gen(x)
        masks = F.softmax(masks, dim=1)  # 每个位置的所有区域权重和为1
        
        # 对每个区域进行特征增强
        enhanced_regions = []
        for i in range(self.num_regions):
            mask = masks[:, i:i+1]
            enhanced = self.region_enhance[i](x)
            enhanced_regions.append(enhanced * mask)
        
        # 合并增强后的区域特征
        output = sum(enhanced_regions)
        
        return output + x  # 残差连接

class FineGrainedAttention(nn.Module):
    """细粒度特征注意力模块,专门针对相似区域的区分"""
    def __init__(self, in_channels, num_features=256):
        super(FineGrainedAttention, self).__init__()
        # 第一阶段:生成注意力图
        self.attention_gen = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 第二阶段:局部特征增强
        self.local_enhancer = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, groups=4),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )
        
        # 初始化特征融合
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        
    def forward(self, x):
        # 生成注意力图
        attention_map = self.attention_gen(x)
        
        # 应用注意力,提取局部特征
        local_features = x * attention_map
        enhanced_features = self.local_enhancer(local_features)
        
        # 融合原始特征和增强特征
        output = torch.cat([x, enhanced_features], dim=1)
        output = self.fusion(output)
        
        return output

class MultiPathEnhancementModule(nn.Module):
    """多路径增强模块,以不同分辨率处理特征"""
    def __init__(self, in_channels):
        super(MultiPathEnhancementModule, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 融合不同分支的特征
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.attention = SEModule(in_channels)
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        # 合并不同分支
        outputs = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        outputs = self.fusion(outputs)
        
        # 应用注意力机制
        outputs = self.attention(outputs)
        
        return outputs + x  # 残差连接