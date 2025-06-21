#!/usr/bin/env python3
"""
行间关系建模模块
架构流程第5步：建模代码行之间的上下文关系
"""

import torch
import torch.nn as nn
from typing import Optional


class InterLineRelationship(nn.Module):
    """行间关系建模 - 使用自注意力机制建模代码行间关系"""
    
    def __init__(self, feature_dim: int = 128, hidden_dim: int = 64, num_heads: int = 8):
        """
        初始化行间关系建模模块
        
        Args:
            feature_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 位置编码
        self.position_embedding = nn.Embedding(1000, hidden_dim)  # 支持最多1000行
        
        # 自注意力机制
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim + hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(feature_dim + hidden_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)
        
        # 输出投影
        self.output_projection = nn.Linear(feature_dim + hidden_dim, feature_dim)
    
    def forward(self, line_features: torch.Tensor, line_positions: torch.Tensor) -> torch.Tensor:
        """
        建模行间关系的前向传播
        
        Args:
            line_features: 行特征张量 [batch, seq_len, feature_dim]
            line_positions: 行位置张量 [batch, seq_len]
            
        Returns:
            torch.Tensor: 上下文增强的特征 [batch, seq_len, feature_dim]
        """
        batch_size, seq_len, feature_dim = line_features.shape
        
        # 位置编码
        pos_embeddings = self.position_embedding(line_positions)  # [batch, seq_len, hidden_dim]
        
        # 特征与位置融合
        enhanced_features = torch.cat([line_features, pos_embeddings], dim=-1)  # [batch, seq_len, feature_dim + hidden_dim]
        
        # 自注意力 + 残差连接
        attended_features, attention_weights = self.self_attention(
            enhanced_features, enhanced_features, enhanced_features
        )  # [batch, seq_len, feature_dim + hidden_dim]
        
        attended_features = self.layer_norm1(enhanced_features + attended_features)
        
        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(attended_features)  # [batch, seq_len, feature_dim]
        output_features = self.layer_norm2(line_features + ff_output)
        
        return output_features
    
    def get_attention_weights(self, line_features: torch.Tensor, line_positions: torch.Tensor) -> torch.Tensor:
        """
        获取注意力权重矩阵
        
        Args:
            line_features: 行特征张量
            line_positions: 行位置张量
            
        Returns:
            torch.Tensor: 注意力权重矩阵 [batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, feature_dim = line_features.shape
        
        # 位置编码
        pos_embeddings = self.position_embedding(line_positions)
        enhanced_features = torch.cat([line_features, pos_embeddings], dim=-1)
        
        # 获取注意力权重
        _, attention_weights = self.self_attention(
            enhanced_features, enhanced_features, enhanced_features
        )
        
        return attention_weights


class ContextualEncoder(nn.Module):
    """上下文编码器 - 多层行间关系建模"""
    
    def __init__(self, feature_dim: int = 128, num_layers: int = 2, num_heads: int = 8):
        """
        初始化上下文编码器
        
        Args:
            feature_dim: 特征维度
            num_layers: 编码器层数
            num_heads: 注意力头数
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        
        # 多层行间关系建模
        self.layers = nn.ModuleList([
            InterLineRelationship(feature_dim, 64, num_heads)
            for _ in range(num_layers)
        ])
    
    def forward(self, line_features: torch.Tensor, line_positions: torch.Tensor) -> torch.Tensor:
        """
        多层上下文编码
        
        Args:
            line_features: 行特征张量
            line_positions: 行位置张量
            
        Returns:
            torch.Tensor: 上下文编码后的特征
        """
        output = line_features
        
        for layer in self.layers:
            output = layer(output, line_positions)
        
        return output 