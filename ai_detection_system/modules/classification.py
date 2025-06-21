#!/usr/bin/env python3
"""
行分类器模块
架构流程第7步：对每行代码进行AI概率预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class LineClassifier(nn.Module):
    """行分类器 - 预测每行代码的AI生成概率"""
    
    def __init__(self, 
                 input_dim: int = 256,  # 手工特征(128) + CodeBERT特征(128)
                 hidden_dims: list = [256, 128, 64],
                 dropout_rate: float = 0.1,
                 activation: str = 'relu'):
        """
        初始化行分类器
        
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            dropout_rate: Dropout比率
            activation: 激活函数类型
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.Dropout(dropout_rate),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        
        self.classifier = nn.Sequential(*layers)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: 输入特征 [batch, input_dim]
            
        Returns:
            torch.Tensor: AI概率 [batch]
        """
        probabilities = self.classifier(features)  # [batch, 1]
        return probabilities.squeeze(-1)  # [batch]
    
    def predict_with_confidence(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        预测并返回置信度信息
        
        Args:
            features: 输入特征
            
        Returns:
            Dict: 包含概率、预测和置信度的字典
        """
        probabilities = self.forward(features)
        predictions = (probabilities > 0.5).float()
        
        # 计算置信度 (距离0.5的距离)
        confidence = torch.abs(probabilities - 0.5) * 2
        
        return {
            'probabilities': probabilities,
            'predictions': predictions,
            'confidence': confidence
        }
    
    def get_feature_importance(self) -> torch.Tensor:
        """
        获取特征重要性（基于第一层权重的L2范数）
        
        Returns:
            torch.Tensor: 特征重要性向量
        """
        first_layer = self.classifier[0]  # 第一个Linear层
        importance = torch.norm(first_layer.weight, dim=0)
        return importance / importance.sum()  # 归一化


class EnsembleClassifier(nn.Module):
    """集成分类器 - 多个分类器的集成"""
    
    def __init__(self, 
                 input_dim: int = 256,
                 num_classifiers: int = 3,
                 hidden_dims: list = [256, 128, 64]):
        """
        初始化集成分类器
        
        Args:
            input_dim: 输入特征维度
            num_classifiers: 分类器数量
            hidden_dims: 隐藏层维度
        """
        super().__init__()
        self.num_classifiers = num_classifiers
        
        # 创建多个分类器
        self.classifiers = nn.ModuleList([
            LineClassifier(input_dim, hidden_dims, dropout_rate=0.1 + i * 0.05)
            for i in range(num_classifiers)
        ])
        
        # 权重融合层
        self.weight_layer = nn.Sequential(
            nn.Linear(num_classifiers, num_classifiers),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        集成前向传播
        
        Args:
            features: 输入特征
            
        Returns:
            torch.Tensor: 集成后的AI概率
        """
        # 获取各个分类器的预测
        predictions = []
        for classifier in self.classifiers:
            pred = classifier(features)
            predictions.append(pred.unsqueeze(-1))
        
        # 拼接预测结果
        all_predictions = torch.cat(predictions, dim=-1)  # [batch, num_classifiers]
        
        # 计算权重
        weights = self.weight_layer(all_predictions)  # [batch, num_classifiers]
        
        # 加权平均
        ensemble_prediction = torch.sum(all_predictions * weights, dim=-1)
        
        return ensemble_prediction
    
    def predict_with_uncertainty(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        预测并返回不确定性信息
        
        Args:
            features: 输入特征
            
        Returns:
            Dict: 包含预测、方差等信息的字典
        """
        # 获取各个分类器的预测
        predictions = []
        for classifier in self.classifiers:
            pred = classifier(features)
            predictions.append(pred)
        
        predictions_tensor = torch.stack(predictions, dim=0)  # [num_classifiers, batch]
        
        # 计算统计量
        mean_prediction = torch.mean(predictions_tensor, dim=0)
        variance = torch.var(predictions_tensor, dim=0)
        std_deviation = torch.sqrt(variance)
        
        return {
            'mean_prediction': mean_prediction,
            'variance': variance,
            'std_deviation': std_deviation,
            'individual_predictions': predictions_tensor
        }


class AdaptiveClassifier(nn.Module):
    """自适应分类器 - 根据输入特征动态调整网络结构"""
    
    def __init__(self, input_dim: int = 256):
        """
        初始化自适应分类器
        
        Args:
            input_dim: 输入特征维度
        """
        super().__init__()
        self.input_dim = input_dim
        
        # 特征分析器
        self.feature_analyzer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 输出3个权重，对应不同的分类路径
        )
        
        # 三个不同的分类路径
        self.simple_path = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.medium_path = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.complex_path = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        自适应前向传播
        
        Args:
            features: 输入特征
            
        Returns:
            torch.Tensor: AI概率
        """
        # 分析特征，决定使用哪个路径
        path_weights = F.softmax(self.feature_analyzer(features), dim=-1)  # [batch, 3]
        
        # 获取三个路径的预测
        simple_pred = self.simple_path(features).squeeze(-1)      # [batch]
        medium_pred = self.medium_path(features).squeeze(-1)      # [batch]
        complex_pred = self.complex_path(features).squeeze(-1)    # [batch]
        
        # 加权融合
        final_prediction = (path_weights[:, 0] * simple_pred + 
                          path_weights[:, 1] * medium_pred + 
                          path_weights[:, 2] * complex_pred)
        
        return final_prediction 