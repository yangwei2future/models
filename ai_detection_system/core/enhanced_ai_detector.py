#!/usr/bin/env python3
"""
增强版AI代码检测器 - 完整架构实现
按照用户要求的流程图实现：文件解析 → 特征提取 → 特征融合 → CodeBERT → 分类器 → 阈值过滤 → 结果聚合

架构流程:
代码文件/文件夹 → 文件解析器 → 行级特征提取 → 文件级特征提取 → 特征融合 → 
行间关系建模 → 行分类器 → 预测概率 → 阈值过滤 → AI生成/人工编码 → 结果聚合 → 输出系统
"""

import os
import sys
import json
import argparse
import logging
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileParser:
    """文件解析器 - 解析代码文件并提取基础信息"""
    
    def __init__(self):
        self.supported_extensions = {
            '.py', '.java', '.js', '.jsx', '.ts', '.tsx',
            '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx',
            '.go', '.rs', '.rb', '.php', '.swift', '.kt'
        }
    
    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """解析单个文件"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # 基础文件信息
            file_info = {
                'filepath': filepath,
                'filename': os.path.basename(filepath),
                'extension': os.path.splitext(filepath)[1],
                'total_lines': len(lines),
                'lines': [],
                'success': True
            }
            
            # 逐行解析
            for i, line in enumerate(lines, 1):
                content = line.rstrip('\n\r')
                line_info = {
                    'line_number': i,
                    'content': content,
                    'original_content': line,
                    'is_empty': not content.strip(),
                    'indent_level': len(content) - len(content.lstrip()) if content else 0
                }
                file_info['lines'].append(line_info)
            
            return file_info
            
        except Exception as e:
            return {
                'filepath': filepath,
                'success': False,
                'error': str(e)
            }
    
    def parse_directory(self, dirpath: str, recursive: bool = False) -> List[Dict[str, Any]]:
        """解析目录中的所有代码文件"""
        files = []
        
        if recursive:
            for root, _, filenames in os.walk(dirpath):
                for filename in filenames:
                    if self.is_code_file(filename):
                        filepath = os.path.join(root, filename)
                        files.append(self.parse_file(filepath))
        else:
            for filename in os.listdir(dirpath):
                filepath = os.path.join(dirpath, filename)
                if os.path.isfile(filepath) and self.is_code_file(filename):
                    files.append(self.parse_file(filepath))
        
        return files
    
    def is_code_file(self, filename: str) -> bool:
        """判断是否为支持的代码文件"""
        _, ext = os.path.splitext(filename.lower())
        return ext in self.supported_extensions


class LineFeatureExtractor:
    """行级特征提取器"""
    
    def __init__(self):
        # AI生成代码的模式
        self.ai_patterns = [
            'comprehensive', 'sophisticated', 'advanced', 'implementation',
            'configuration', 'initialization', 'processing', 'algorithm',
            'calculate', 'compute', 'optimize', 'generate', 'create',
            'implement', 'function', 'method', 'handle', 'manage', 'execute'
        ]
        
        # 代码风格指标
        self.style_indicators = {
            'docstring_start': ['"""', "'''", '/**'],
            'type_annotations': [':', '->', 'List[', 'Dict[', 'Optional[', 'Union['],
            'logging_patterns': ['logger.', 'logging.', 'log.'],
            'exception_patterns': ['raise', 'except', 'try:', 'finally:'],
            'import_patterns': ['from typing import', 'from abc import', 'import logging']
        }
    
    def extract_features(self, line_info: Dict[str, Any], file_context: Dict[str, Any]) -> List[float]:
        """提取单行的特征向量"""
        content = line_info['content']
        line_number = line_info['line_number']
        total_lines = file_context['total_lines']
        
        # 基础特征 (8维)
        length = len(content)
        indent_level = line_info['indent_level']
        relative_position = line_number / total_lines if total_lines > 0 else 0
        
        # 内容特征 (6维)
        has_comment = '#' in content or '//' in content or '/*' in content
        has_string = '"' in content or "'" in content or '`' in content
        has_number = any(c.isdigit() for c in content)
        has_operator = any(op in content for op in ['=', '+', '-', '*', '/', '<', '>', '!', '&', '|'])
        
        # 复杂度特征 (4维)
        bracket_complexity = content.count('(') + content.count('[') + content.count('{')
        word_count = len(content.split())
        word_density = word_count / max(len(content), 1)
        char_diversity = len(set(content)) / max(len(content), 1)
        
        # AI模式特征 (5维)
        has_ai_pattern = any(pattern in content.lower() for pattern in self.ai_patterns)
        has_docstring = any(pattern in content for pattern in self.style_indicators['docstring_start'])
        has_type_annotation = any(pattern in content for pattern in self.style_indicators['type_annotations'])
        has_logging = any(pattern in content for pattern in self.style_indicators['logging_patterns'])
        has_advanced_import = any(pattern in content for pattern in self.style_indicators['import_patterns'])
        
        # 上下文特征 (3维)
        is_first_line = line_number == 1
        is_last_line = line_number == total_lines
        is_middle_section = 0.3 < relative_position < 0.7
        
        # 归一化特征
        features = [
            # 基础特征 (8维)
            min(length / 100.0, 1.0),           # 0. 归一化长度
            min(indent_level / 20.0, 1.0),      # 1. 归一化缩进
            relative_position,                   # 2. 相对位置
            min(word_count / 20.0, 1.0),        # 3. 词汇数量
            
            # 内容特征 (6维)
            float(has_comment),                  # 4. 注释
            float(has_string),                   # 5. 字符串
            float(has_number),                   # 6. 数字
            float(has_operator),                 # 7. 操作符
            
            # 复杂度特征 (4维)
            min(bracket_complexity / 10.0, 1.0), # 8. 括号复杂度
            word_density,                        # 9. 词汇密度
            char_diversity,                      # 10. 字符多样性
            
            # AI模式特征 (5维)
            float(has_ai_pattern),               # 11. AI模式
            float(has_docstring),                # 12. 文档字符串
            float(has_type_annotation),          # 13. 类型注解
            float(has_logging),                  # 14. 日志记录
            float(has_advanced_import),          # 15. 高级导入
            
            # 上下文特征 (3维)
            float(is_first_line),                # 16. 首行
            float(is_last_line),                 # 17. 末行
            float(is_middle_section),            # 18. 中间段
        ]
        
        return features


class FileFeatureExtractor:
    """文件级特征提取器"""
    
    def extract_features(self, file_info: Dict[str, Any]) -> List[float]:
        """提取文件级特征"""
        lines = file_info['lines']
        total_lines = len(lines)
        
        if total_lines == 0:
            return [0.0] * 15  # 返回15维零向量
        
        # 统计特征
        non_empty_lines = [line for line in lines if not line['is_empty']]
        comment_lines = [line for line in non_empty_lines if '#' in line['content'] or '//' in line['content']]
        
        # 基础统计 (5维)
        file_size = total_lines
        code_density = len(non_empty_lines) / total_lines
        comment_ratio = len(comment_lines) / max(len(non_empty_lines), 1)
        avg_line_length = np.mean([len(line['content']) for line in non_empty_lines]) if non_empty_lines else 0
        avg_indent = np.mean([line['indent_level'] for line in non_empty_lines]) if non_empty_lines else 0
        
        # 复杂度特征 (5维)
        total_brackets = sum(line['content'].count('(') + line['content'].count('[') + line['content'].count('{') 
                           for line in non_empty_lines)
        avg_complexity = total_brackets / max(len(non_empty_lines), 1)
        
        function_count = sum(1 for line in non_empty_lines if 'def ' in line['content'] or 'function ' in line['content'])
        class_count = sum(1 for line in non_empty_lines if 'class ' in line['content'])
        import_count = sum(1 for line in non_empty_lines if line['content'].strip().startswith(('import ', 'from ')))
        
        # AI风格特征 (5维)
        docstring_count = sum(1 for line in non_empty_lines if '"""' in line['content'] or "'''" in line['content'])
        type_annotation_count = sum(1 for line in non_empty_lines if '->' in line['content'] or ': ' in line['content'])
        logging_count = sum(1 for line in non_empty_lines if 'logger' in line['content'] or 'logging' in line['content'])
        exception_count = sum(1 for line in non_empty_lines if any(keyword in line['content'] for keyword in ['try:', 'except', 'raise', 'finally:']))
        advanced_pattern_count = sum(1 for line in non_empty_lines if any(pattern in line['content'].lower() 
                                   for pattern in ['comprehensive', 'sophisticated', 'implementation', 'configuration']))
        
        # 归一化特征
        features = [
            # 基础统计 (5维)
            min(file_size / 1000.0, 1.0),           # 0. 文件大小
            code_density,                            # 1. 代码密度
            comment_ratio,                           # 2. 注释比例
            min(avg_line_length / 80.0, 1.0),       # 3. 平均行长度
            min(avg_indent / 10.0, 1.0),            # 4. 平均缩进
            
            # 复杂度特征 (5维)
            min(avg_complexity / 5.0, 1.0),         # 5. 平均复杂度
            min(function_count / 20.0, 1.0),        # 6. 函数数量
            min(class_count / 10.0, 1.0),           # 7. 类数量
            min(import_count / 20.0, 1.0),          # 8. 导入数量
            
            # AI风格特征 (5维)
            min(docstring_count / 10.0, 1.0),       # 9. 文档字符串
            min(type_annotation_count / 20.0, 1.0), # 10. 类型注解
            min(logging_count / 10.0, 1.0),         # 11. 日志记录
            min(exception_count / 10.0, 1.0),       # 12. 异常处理
            min(advanced_pattern_count / 10.0, 1.0) # 13. 高级模式
        ]
        
        return features


class FeatureFusion(nn.Module):
    """特征融合模块"""
    
    def __init__(self, line_feature_dim: int = 19, file_feature_dim: int = 14, output_dim: int = 128):
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
        """特征融合前向传播"""
        # 投影到相同维度
        line_proj = self.line_projection(line_features)      # [batch, output_dim//2]
        file_proj = self.file_projection(file_features)      # [batch, output_dim//2]
        
        # 拼接融合
        fused = torch.cat([line_proj, file_proj], dim=-1)    # [batch, output_dim]
        
        # 融合处理
        output = self.fusion_layer(fused)                    # [batch, output_dim]
        
        return output


class InterLineRelationship(nn.Module):
    """行间关系建模"""
    
    def __init__(self, feature_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 位置编码
        self.position_embedding = nn.Embedding(1000, hidden_dim)  # 支持最多1000行
        
        # 自注意力机制
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim + hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 输出投影
        self.output_projection = nn.Linear(feature_dim + hidden_dim, feature_dim)
    
    def forward(self, line_features: torch.Tensor, line_positions: torch.Tensor) -> torch.Tensor:
        """建模行间关系"""
        batch_size, seq_len, feature_dim = line_features.shape
        
        # 位置编码
        pos_embeddings = self.position_embedding(line_positions)  # [batch, seq_len, hidden_dim]
        
        # 特征与位置融合
        enhanced_features = torch.cat([line_features, pos_embeddings], dim=-1)  # [batch, seq_len, feature_dim + hidden_dim]
        
        # 自注意力
        attended_features, _ = self.self_attention(
            enhanced_features, enhanced_features, enhanced_features
        )  # [batch, seq_len, feature_dim + hidden_dim]
        
        # 输出投影
        output = self.output_projection(attended_features)  # [batch, seq_len, feature_dim]
        
        return output


class EnhancedAIDetector(nn.Module):
    """增强版AI检测器 - 完整架构"""
    
    def __init__(self, 
                 codebert_model: str = "microsoft/codebert-base",
                 line_feature_dim: int = 19,
                 file_feature_dim: int = 14,
                 fusion_dim: int = 128,
                 hidden_dim: int = 256):
        super().__init__()
        
        # CodeBERT
        self.tokenizer = AutoTokenizer.from_pretrained(codebert_model)
        self.codebert = AutoModel.from_pretrained(codebert_model)
        self.codebert_dim = self.codebert.config.hidden_size  # 768
        
        # 特征融合
        self.feature_fusion = FeatureFusion(line_feature_dim, file_feature_dim, fusion_dim)
        
        # 行间关系建模
        self.inter_line_relationship = InterLineRelationship(fusion_dim, 64)
        
        # CodeBERT特征投影
        self.codebert_projection = nn.Linear(self.codebert_dim, fusion_dim)
        
        # 最终特征融合
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, hidden_dim),  # 手工特征 + CodeBERT特征
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim)
        )
        
        # 行分类器
        self.line_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def encode_with_codebert(self, lines: List[str]) -> torch.Tensor:
        """使用CodeBERT编码代码行"""
        # 批量处理
        processed_lines = []
        for line in lines:
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
        
        # CodeBERT编码
        with torch.no_grad():
            outputs = self.codebert(**inputs)
        
        # 提取[CLS] token表示
        code_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        
        return code_embeddings
    
    def forward(self, 
                line_features: torch.Tensor,
                file_features: torch.Tensor,
                code_lines: List[str],
                line_positions: torch.Tensor) -> torch.Tensor:
        """完整的前向传播"""
        
        # 1. 特征融合
        fused_features = self.feature_fusion(line_features, file_features)  # [batch, fusion_dim]
        
        # 2. 行间关系建模
        if len(fused_features.shape) == 2:
            fused_features = fused_features.unsqueeze(0)  # [1, batch, fusion_dim]
        
        contextual_features = self.inter_line_relationship(fused_features, line_positions)  # [1, batch, fusion_dim]
        contextual_features = contextual_features.squeeze(0)  # [batch, fusion_dim]
        
        # 3. CodeBERT编码
        codebert_features = self.encode_with_codebert(code_lines)  # [batch, 768]
        codebert_projected = self.codebert_projection(codebert_features)  # [batch, fusion_dim]
        
        # 4. 最终特征融合
        final_features = torch.cat([contextual_features, codebert_projected], dim=-1)  # [batch, fusion_dim*2]
        final_features = self.final_fusion(final_features)  # [batch, hidden_dim]
        
        # 5. 行分类器
        ai_probabilities = self.line_classifier(final_features)  # [batch, 1]
        
        return ai_probabilities.squeeze(-1)  # [batch]


class ThresholdFilter:
    """阈值过滤模块"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def filter(self, probabilities: torch.Tensor) -> torch.Tensor:
        """应用阈值过滤"""
        return (probabilities > self.threshold).float()
    
    def set_threshold(self, threshold: float):
        """动态设置阈值"""
        self.threshold = threshold


class ResultAggregator:
    """结果聚合模块"""
    
    def aggregate_file_results(self, 
                             file_info: Dict[str, Any],
                             ai_probabilities: List[float],
                             ai_predictions: List[bool]) -> Dict[str, Any]:
        """聚合单个文件的结果"""
        lines = file_info['lines']
        
        # 逐行结果
        line_results = []
        for i, (line_info, ai_prob, is_ai) in enumerate(zip(lines, ai_probabilities, ai_predictions)):
            if not line_info['is_empty']:  # 只处理非空行
                line_results.append({
                    "line_number": line_info['line_number'],
                    "content": line_info['content'],
                    "ai_prob": round(float(ai_prob), 3),
                    "is_ai": bool(is_ai)
                })
        
        # 文件级统计
        code_lines = [r for r in line_results if r['content'].strip()]
        ai_lines = [r for r in line_results if r['is_ai']]
        
        file_result = {
            "file_path": file_info['filepath'],
            "success": True,
            "lines": line_results,
            "summary": {
                "total_lines": file_info['total_lines'],
                "code_lines": len(code_lines),
                "ai_lines": len(ai_lines),
                "ai_percentage": round((len(ai_lines) / len(code_lines) * 100) if code_lines else 0, 1),
                "average_ai_prob": round(np.mean([r['ai_prob'] for r in code_lines]), 3) if code_lines else 0.0
            }
        }
        
        return file_result
    
    def aggregate_batch_results(self, file_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合批量文件结果"""
        successful_results = [r for r in file_results if r.get('success', False)]
        
        # 总体统计
        total_files = len(file_results)
        successful_files = len(successful_results)
        failed_files = total_files - successful_files
        
        total_lines = sum(r['summary']['total_lines'] for r in successful_results)
        total_code_lines = sum(r['summary']['code_lines'] for r in successful_results)
        total_ai_lines = sum(r['summary']['ai_lines'] for r in successful_results)
        
        # 构建最终结果
        batch_result = {
            "results": file_results,
            "statistics": {
                "total_files": total_files,
                "successful_files": successful_files,
                "failed_files": failed_files,
                "total_lines": total_lines,
                "total_code_lines": total_code_lines,
                "total_ai_lines": total_ai_lines,
                "overall_ai_percentage": round((total_ai_lines / total_code_lines * 100) if total_code_lines > 0 else 0, 1),
                "average_file_ai_percentage": round(np.mean([r['summary']['ai_percentage'] for r in successful_results]), 1) if successful_results else 0
            },
            "metadata": {
                "model_type": "Enhanced CodeBERT-based AI Detector",
                "architecture": "File Parser → Feature Extraction → Feature Fusion → Inter-line Modeling → CodeBERT → Classifier → Threshold Filter → Result Aggregation",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return batch_result


class EnhancedAIDetectionSystem:
    """增强版AI检测系统 - 完整流程"""
    
    def __init__(self, model_path: Optional[str] = None, threshold: float = 0.5):
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        
        # 初始化各个模块
        self.file_parser = FileParser()
        self.line_feature_extractor = LineFeatureExtractor()
        self.file_feature_extractor = FileFeatureExtractor()
        self.threshold_filter = ThresholdFilter(threshold)
        self.result_aggregator = ResultAggregator()
        
        # 初始化模型
        self.model = EnhancedAIDetector()
        
        # 加载预训练权重
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            self.logger.warning("No model file provided or file not found, using random weights")
        
        self.model.eval()
    
    def _load_model(self, model_path: str):
        """加载模型权重"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.logger.info(f"✅ Enhanced model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
    
    def detect_file(self, filepath: str) -> Dict[str, Any]:
        """检测单个文件"""
        # 1. 文件解析
        file_info = self.file_parser.parse_file(filepath)
        if not file_info['success']:
            return file_info
        
        # 2. 特征提取
        line_features_list = []
        code_lines = []
        line_positions = []
        
        for line_info in file_info['lines']:
            if not line_info['is_empty']:  # 只处理非空行
                # 行级特征
                line_features = self.line_feature_extractor.extract_features(line_info, file_info)
                line_features_list.append(line_features)
                
                # 代码内容
                code_lines.append(line_info['content'])
                
                # 行位置
                line_positions.append(line_info['line_number'] - 1)  # 0-indexed
        
        if not line_features_list:
            return {
                "file_path": filepath,
                "success": False,
                "error": "No code lines found"
            }
        
        # 文件级特征
        file_features = self.file_feature_extractor.extract_features(file_info)
        
        # 3. 转换为张量
        line_features_tensor = torch.tensor(line_features_list, dtype=torch.float32)
        file_features_tensor = torch.tensor(file_features, dtype=torch.float32).unsqueeze(0).repeat(len(line_features_list), 1)
        line_positions_tensor = torch.tensor(line_positions, dtype=torch.long).unsqueeze(0)
        
        # 4. 模型推理
        with torch.no_grad():
            ai_probabilities = self.model(
                line_features_tensor,
                file_features_tensor,
                code_lines,
                line_positions_tensor
            )
        
        # 5. 阈值过滤
        ai_predictions = self.threshold_filter.filter(ai_probabilities)
        
        # 6. 结果聚合
        # 为空行填充默认值
        full_probabilities = []
        full_predictions = []
        code_idx = 0
        
        for line_info in file_info['lines']:
            if line_info['is_empty']:
                full_probabilities.append(0.0)
                full_predictions.append(False)
            else:
                full_probabilities.append(float(ai_probabilities[code_idx]))
                full_predictions.append(bool(ai_predictions[code_idx]))
                code_idx += 1
        
        result = self.result_aggregator.aggregate_file_results(
            file_info, full_probabilities, full_predictions
        )
        
        return result
    
    def detect_batch(self, input_paths: List[str], recursive: bool = False) -> Dict[str, Any]:
        """批量检测"""
        all_files = []
        
        for input_path in input_paths:
            if os.path.isfile(input_path):
                all_files.append(input_path)
            elif os.path.isdir(input_path):
                dir_files = self.file_parser.parse_directory(input_path, recursive)
                all_files.extend([f['filepath'] for f in dir_files if f['success']])
        
        # 批量检测
        results = []
        for filepath in all_files:
            self.logger.info(f"Processing: {filepath}")
            result = self.detect_file(filepath)
            results.append(result)
        
        # 聚合结果
        batch_result = self.result_aggregator.aggregate_batch_results(results)
        
        return batch_result


def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(
        description="Enhanced AI Code Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Architecture Flow:
代码文件/文件夹 → 文件解析器 → 行级特征提取 → 文件级特征提取 → 特征融合 → 
行间关系建模 → CodeBERT编码 → 最终融合 → 行分类器 → 阈值过滤 → 结果聚合 → JSON输出

Examples:
  %(prog)s --input file.py --output results.json
  %(prog)s --input src/ --recursive --threshold 0.7 --output results.json
        """
    )
    
    parser.add_argument("--input", type=str, nargs='+', required=True,
                       help="Input files, directories, or patterns")
    parser.add_argument("--output", type=str, default="detection_results.json",
                       help="Output JSON file")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to trained model file")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="AI detection threshold (0.0-1.0)")
    parser.add_argument("--recursive", "-r", action="store_true",
                       help="Recursively process directories")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🚀 Enhanced AI Code Detection System")
    print("=" * 60)
    print("Architecture: File Parser → Feature Extraction → Feature Fusion →")
    print("              Inter-line Modeling → CodeBERT → Classifier →")
    print("              Threshold Filter → Result Aggregation")
    print("=" * 60)
    
    # 初始化系统
    detection_system = EnhancedAIDetectionSystem(
        model_path=args.model,
        threshold=args.threshold
    )
    
    # 执行检测
    print(f"📁 Processing: {args.input}")
    print(f"🎯 Threshold: {args.threshold}")
    print(f"🔄 Recursive: {args.recursive}")
    
    results = detection_system.detect_batch(args.input, args.recursive)
    
    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 显示统计
    stats = results['statistics']
    print(f"\n📊 Detection Results:")
    print(f"   Files processed: {stats['successful_files']}/{stats['total_files']}")
    print(f"   Total lines: {stats['total_lines']}")
    print(f"   Code lines: {stats['total_code_lines']}")
    print(f"   AI-generated lines: {stats['total_ai_lines']}")
    print(f"   Overall AI percentage: {stats['overall_ai_percentage']}%")
    print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main() 