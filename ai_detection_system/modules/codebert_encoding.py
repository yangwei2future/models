#!/usr/bin/env python3
"""
CodeBERT编码模块
架构流程第6步：使用CodeBERT对代码进行语义编码
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Optional


class CodeBERTEncoder(nn.Module):
    """CodeBERT编码器 - 使用预训练CodeBERT模型编码代码语义"""
    
    def __init__(self, 
                 model_name: str = "microsoft/codebert-base",
                 output_dim: int = 128,
                 freeze_bert: bool = False):
        """
        初始化CodeBERT编码器
        
        Args:
            model_name: CodeBERT模型名称
            output_dim: 输出特征维度
            freeze_bert: 是否冻结BERT参数
        """
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        
        # 加载CodeBERT
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.codebert = AutoModel.from_pretrained(model_name)
        self.codebert_dim = self.codebert.config.hidden_size  # 768
        
        # 是否冻结BERT参数
        if freeze_bert:
            for param in self.codebert.parameters():
                param.requires_grad = False
        
        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(self.codebert_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def encode_lines(self, code_lines: List[str], max_length: int = 512) -> torch.Tensor:
        """
        编码代码行列表
        
        Args:
            code_lines: 代码行列表
            max_length: 最大序列长度
            
        Returns:
            torch.Tensor: 编码后的特征 [batch, output_dim]
        """
        # 预处理代码行
        processed_lines = []
        for line in code_lines:
            if not line.strip():
                processed_lines.append("[EMPTY_LINE]")
            else:
                processed_lines.append(line)
        
        # Tokenization
        inputs = self.tokenizer(
            processed_lines,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )
        
        # 移动到设备
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # CodeBERT编码
        with torch.no_grad() if hasattr(self, '_freeze_bert') and self._freeze_bert else torch.enable_grad():
            outputs = self.codebert(**inputs)
        
        # 提取[CLS] token表示
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        
        # 投影到目标维度
        projected_embeddings = self.projection(cls_embeddings)  # [batch, output_dim]
        
        return projected_embeddings
    
    def encode_single_line(self, code_line: str) -> torch.Tensor:
        """
        编码单行代码
        
        Args:
            code_line: 单行代码
            
        Returns:
            torch.Tensor: 编码后的特征 [1, output_dim]
        """
        return self.encode_lines([code_line])
    
    def get_attention_weights(self, code_lines: List[str]) -> torch.Tensor:
        """
        获取注意力权重
        
        Args:
            code_lines: 代码行列表
            
        Returns:
            torch.Tensor: 注意力权重
        """
        # 预处理
        processed_lines = []
        for line in code_lines:
            if not line.strip():
                processed_lines.append("[EMPTY_LINE]")
            else:
                processed_lines.append(line)
        
        # Tokenization
        inputs = self.tokenizer(
            processed_lines,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # 移动到设备
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 获取注意力权重
        with torch.no_grad():
            outputs = self.codebert(**inputs, output_attentions=True)
        
        # 返回最后一层的注意力权重
        return outputs.attentions[-1]  # [batch, num_heads, seq_len, seq_len]


class MultiScaleCodeBERTEncoder(nn.Module):
    """多尺度CodeBERT编码器 - 结合不同粒度的代码表示"""
    
    def __init__(self, 
                 model_name: str = "microsoft/codebert-base",
                 output_dim: int = 128):
        """
        初始化多尺度编码器
        
        Args:
            model_name: CodeBERT模型名称
            output_dim: 输出特征维度
        """
        super().__init__()
        self.output_dim = output_dim
        
        # 基础编码器
        self.base_encoder = CodeBERTEncoder(model_name, output_dim // 2)
        
        # 上下文编码器（处理多行）
        self.context_encoder = CodeBERTEncoder(model_name, output_dim // 2)
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, code_lines: List[str], context_window: int = 3) -> torch.Tensor:
        """
        多尺度编码
        
        Args:
            code_lines: 代码行列表
            context_window: 上下文窗口大小
            
        Returns:
            torch.Tensor: 多尺度编码特征
        """
        # 单行编码
        line_features = self.base_encoder.encode_lines(code_lines)
        
        # 上下文编码
        context_lines = []
        for i, line in enumerate(code_lines):
            # 构建上下文窗口
            start_idx = max(0, i - context_window // 2)
            end_idx = min(len(code_lines), i + context_window // 2 + 1)
            context = " ".join(code_lines[start_idx:end_idx])
            context_lines.append(context)
        
        context_features = self.context_encoder.encode_lines(context_lines)
        
        # 特征融合
        combined_features = torch.cat([line_features, context_features], dim=-1)
        output_features = self.fusion_layer(combined_features)
        
        return output_features 