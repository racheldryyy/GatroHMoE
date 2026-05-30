import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_models import BaseModel
from config import HIDDEN_DIM

class StudentModel(nn.Module):
    """学生模型用于知识蒸馏"""
    
    def __init__(self, base_model_name, num_colon_classes, num_ugi_classes, num_colon_disease_classes, num_ugi_disease_classes,
                hidden_dim=HIDDEN_DIM // 2):
        super(StudentModel, self).__init__()
        
        # 轻量级编码器
        self.encoder = BaseModel(model_name=base_model_name, pretrained=True)
        self.feature_dim = self.encoder.get_feature_dim()
        
        # 共享特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),  # 使用LayerNorm替代BatchNorm
            nn.Dropout(0.3)
        )
        
        # 任务特定分类头
        self.colon_classifier = nn.Linear(hidden_dim, num_colon_classes)
        self.ugi_classifier = nn.Linear(hidden_dim, num_ugi_classes)
        self.colon_disease_classifier = nn.Linear(hidden_dim, num_colon_disease_classes)
        self.ugi_disease_classifier = nn.Linear(hidden_dim, num_ugi_disease_classes)
    
    def forward(self, x, task=None):
        """
        前向传播
        
        参数:
            x: 输入图像
            task: 指定任务 ('colon', 'ugi', 'colon_disease', 'ugi_disease' 或 None)
        
        返回:
            如果未指定任务，返回所有任务的预测结果
            如果指定了任务，返回该任务的预测结果
        """
        # 获取图像特征
        features = self.encoder(x, task=task)
        
        # 共享特征提取
        shared_features = self.shared_layers(features)
        
        # 应用任务特定分类器
        colon_output = self.colon_classifier(shared_features)
        ugi_output = self.ugi_classifier(shared_features)
        colon_disease_output = self.colon_disease_classifier(shared_features)
        ugi_disease_output = self.ugi_disease_classifier(shared_features)
        
        # 根据指定任务返回结果
        if task == 'colon':
            return colon_output
        elif task == 'ugi':
            return ugi_output
        elif task == 'colon_disease':
            return disease_output
        elif task == 'ugi_disease':
            return disease_output
        else:
            return {
                'colon': colon_output,
                'ugi': ugi_output,
                'colon_disease': colon_disease_output,
                'ugi_disease': ugi_disease_output
            }