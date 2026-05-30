import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.checkpoint import checkpoint
from config import INPUT_SIZE, USE_GRADIENT_CHECKPOINTING

class BaseModel(nn.Module):
    """基础模型类，支持各种预训练模型"""
    
    def __init__(self, model_name='resnet50', pretrained=True, feature_extract=False, num_classes=None):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.feature_extract = feature_extract
        
        # 获取基础模型和特征维度
        self.model, self.feature_dim = self._initialize_model(model_name, pretrained, feature_extract)
        
        # 如果指定了类别数量，添加分类头
        if num_classes is not None:
            self.classifier = nn.Linear(self.feature_dim, num_classes)
        else:
            self.classifier = None
    
    def _initialize_model(self, model_name, pretrained, feature_extract):
        """初始化所选模型"""
        model = None
        feature_dim = 0
        
        if model_name == 'resnet18':
            if pretrained:
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet18(weights=None)
            feature_dim = model.fc.in_features
            if feature_extract:
                for param in model.parameters():
                    param.requires_grad = False
            model.fc = nn.Identity()
            
        elif model_name == 'resnet50':
            if pretrained:
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet50(weights=None)
            feature_dim = model.fc.in_features
            if feature_extract:
                for param in model.parameters():
                    param.requires_grad = False
            model.fc = nn.Identity()
            
        elif model_name == 'mobilenetv3_small':
            if pretrained:
                model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            else:
                model = models.mobilenet_v3_small(weights=None)
            feature_dim = model.classifier[0].in_features
            if feature_extract:
                for param in model.parameters():
                    param.requires_grad = False
            model.classifier = nn.Identity()
            
        elif model_name == 'efficientnet_b0':
            if pretrained:
                model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                model = models.efficientnet_b0(weights=None)
            feature_dim = model.classifier[1].in_features
            if feature_extract:
                for param in model.parameters():
                    param.requires_grad = False
            model.classifier = nn.Identity()
            
        elif model_name == 'vit_b_16':
            if pretrained:
                model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            else:
                model = models.vit_b_16(weights=None)
            feature_dim = model.heads.head.in_features
            if feature_extract:
                for param in model.parameters():
                    param.requires_grad = False
            model.heads = nn.Identity()
            
        elif model_name == 'shufflenet_v2_x0_5':
            if pretrained:
                model = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
            else:
                model = models.shufflenet_v2_x0_5(weights=None)
            feature_dim = model.fc.in_features
            if feature_extract:
                for param in model.parameters():
                    param.requires_grad = False
            model.fc = nn.Identity()
            
        else:
            raise ValueError(f"不支持的模型名称: {model_name}")
        
        return model, feature_dim
    
    def forward(self, x, task=None):  # 修改：添加task参数
        """前向传播 - 支持梯度检查点"""
        if USE_GRADIENT_CHECKPOINTING and self.training:
            # 使用梯度检查点节省内存
            features = checkpoint(self.model, x, use_reentrant=False)
        else:
            features = self.model(x)
        
        if self.classifier is not None:
            return self.classifier(features), features
        else:
            return features
            
    def get_feature_dim(self):
        """返回特征维度"""
        return self.feature_dim