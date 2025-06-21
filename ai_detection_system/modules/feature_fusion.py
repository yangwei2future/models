#!/usr/bin/env python3
"""
特征融合模块
架构流程第4步：融合行级特征和文件级特征
"""

import torch
import torch.nn as nn
from typing import Tuple


class FeatureFusion(nn.Module):
    """特征融合模块 - 融合行级特征和文件级特征"""
    
    def __init__(self, line_feature_dim: int = 19, file_feature_dim: int = 14, output_dim: int = 128):
        """
        初始化特征融合模块
        
        Args:
            line_feature_dim: 行级特征维度
            file_feature_dim: 文件级特征维度
            output_dim: 输出特征维度
        """
        super().__init__()
        self.line_feature_dim = line_feature_dim
        self.file_feature_dim = file_feature_dim
        self.output_dim = output_dim
        
        # 特征投影层
        self.line_projection = nn.Linear(line_feature_dim, output_dim // 2)
        self.file_projection = nn.Linear(file_feature_dim, output_dim // 2)
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, line_features: torch.Tensor, file_features: torch.Tensor) -> torch.Tensor:
        """
        特征融合前向传播
        
        Args:
            line_features: 行级特征张量 [batch, line_feature_dim]
            file_features: 文件级特征张量 [batch, file_feature_dim]
            
        Returns:
            torch.Tensor: 融合后的特征 [batch, output_dim]
        """
        # 投影到相同维度
        line_proj = self.line_projection(line_features)      # [batch, output_dim//2]
        file_proj = self.file_projection(file_features)      # [batch, output_dim//2]
        
        # 拼接融合
        fused = torch.cat([line_proj, file_proj], dim=-1)    # [batch, output_dim]
        
        # 融合处理
        output = self.fusion_layer(fused)                    # [batch, output_dim]
        
        return output
    
    def get_feature_importance(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取特征重要性权重
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (行级特征权重, 文件级特征权重)
        """
        line_weights = torch.norm(self.line_projection.weight, dim=0)
        file_weights = torch.norm(self.file_projection.weight, dim=0)
        
        return line_weights, file_weights 