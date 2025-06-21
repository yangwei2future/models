#!/usr/bin/env python3
"""
模块化AI代码检测器
按照架构流程图实现的完整检测系统

架构流程:
代码文件/文件夹 → 文件解析器 → 行级特征提取 → 文件级特征提取 → 特征融合 → 
行间关系建模 → CodeBERT编码 → 行分类器 → 阈值过滤 → 结果聚合 → 输出系统
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入所有模块
from modules.file_parser import FileParser
from modules.feature_extraction import LineFeatureExtractor, FileFeatureExtractor
from modules.feature_fusion import FeatureFusion
from modules.inter_line_modeling import InterLineRelationship
from modules.codebert_encoding import CodeBERTEncoder
from modules.classification import LineClassifier
from modules.threshold_filter import ThresholdFilter
from modules.result_aggregation import ResultAggregator
from modules.output_system import OutputSystem

import torch
import torch.nn as nn
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModularAIDetector(nn.Module):
    """模块化AI检测器 - 完整的架构实现"""
    
    def __init__(self, 
                 codebert_model: str = "microsoft/codebert-base",
                 threshold: float = 0.5,
                 output_dir: str = "./output"):
        super().__init__()
        
        # 初始化所有模块
        self.file_parser = FileParser()
        self.line_feature_extractor = LineFeatureExtractor()
        self.file_feature_extractor = FileFeatureExtractor()
        
        # 神经网络模块
        self.feature_fusion = FeatureFusion(line_feature_dim=19, file_feature_dim=14, output_dim=128)
        self.inter_line_modeling = InterLineRelationship(feature_dim=128, hidden_dim=64)
        self.codebert_encoder = CodeBERTEncoder(model_name=codebert_model, output_dim=128)
        self.line_classifier = LineClassifier(input_dim=256, hidden_dims=[256, 128, 64])
        
        # 后处理模块
        self.threshold_filter = ThresholdFilter(threshold=threshold)
        self.result_aggregator = ResultAggregator()
        self.output_system = OutputSystem(output_dir=output_dir)
        
        # 设置评估模式
        self.eval()
    
    def detect_file(self, filepath: str) -> Dict[str, Any]:
        """
        检测单个文件
        按照完整的架构流程处理
        
        Args:
            filepath: 文件路径
            
        Returns:
            Dict: 检测结果
        """
        logger.info(f"🔍 开始检测文件: {filepath}")
        
        # 步骤1: 文件解析
        logger.debug("📄 步骤1: 文件解析")
        file_info = self.file_parser.parse_file(filepath)
        if not file_info['success']:
            return file_info
        
        # 筛选非空行
        non_empty_lines = [line for line in file_info['lines'] if not line['is_empty']]
        if not non_empty_lines:
            return {
                "file_path": filepath,
                "success": False,
                "error": "No code lines found"
            }
        
        # 步骤2: 行级特征提取
        logger.debug("🔧 步骤2: 行级特征提取")
        line_features_list = []
        for line_info in non_empty_lines:
            features = self.line_feature_extractor.extract_features(line_info, file_info)
            line_features_list.append(features)
        
        # 步骤3: 文件级特征提取
        logger.debug("📊 步骤3: 文件级特征提取")
        file_features = self.file_feature_extractor.extract_features(file_info)
        
        # 转换为张量
        line_features_tensor = torch.tensor(line_features_list, dtype=torch.float32)
        file_features_tensor = torch.tensor(file_features, dtype=torch.float32).unsqueeze(0).repeat(len(line_features_list), 1)
        
        # 步骤4: 特征融合
        logger.debug("🔗 步骤4: 特征融合")
        fused_features = self.feature_fusion(line_features_tensor, file_features_tensor)
        
        # 步骤5: 行间关系建模
        logger.debug("🧠 步骤5: 行间关系建模")
        line_positions = torch.tensor([line['line_number'] - 1 for line in non_empty_lines], dtype=torch.long).unsqueeze(0)
        fused_features_expanded = fused_features.unsqueeze(0)  # [1, num_lines, feature_dim]
        contextual_features = self.inter_line_modeling(fused_features_expanded, line_positions)
        contextual_features = contextual_features.squeeze(0)  # [num_lines, feature_dim]
        
        # 步骤6: CodeBERT编码
        logger.debug("🤖 步骤6: CodeBERT编码")
        code_lines = [line['content'] for line in non_empty_lines]
        codebert_features = self.codebert_encoder.encode_lines(code_lines)
        
        # 最终特征融合
        final_features = torch.cat([contextual_features, codebert_features], dim=-1)
        
        # 步骤7: 行分类器
        logger.debug("🎯 步骤7: 行分类器预测")
        with torch.no_grad():
            ai_probabilities = self.line_classifier(final_features)
        
        # 步骤8: 阈值过滤
        logger.debug("⚖️ 步骤8: 阈值过滤")
        ai_predictions = self.threshold_filter.filter(ai_probabilities)
        
        # 为所有行（包括空行）填充结果
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
        
        # 步骤9: 结果聚合
        logger.debug("📋 步骤9: 结果聚合")
        result = self.result_aggregator.aggregate_file_results(
            file_info, full_probabilities, full_predictions
        )
        
        logger.info(f"✅ 文件检测完成: AI比例 {result['summary']['ai_percentage']}%")
        return result
    
    def detect_batch(self, input_paths: List[str], recursive: bool = False) -> Dict[str, Any]:
        """
        批量检测
        
        Args:
            input_paths: 输入路径列表
            recursive: 是否递归处理目录
            
        Returns:
            Dict: 批量检测结果
        """
        logger.info(f"🚀 开始批量检测: {len(input_paths)} 个路径")
        
        # 收集所有文件
        all_files = []
        for input_path in input_paths:
            if os.path.isfile(input_path):
                if self.file_parser.is_code_file(os.path.basename(input_path)):
                    all_files.append(input_path)
            elif os.path.isdir(input_path):
                dir_files = self.file_parser.parse_directory(input_path, recursive)
                all_files.extend([f['filepath'] for f in dir_files if f['success']])
        
        logger.info(f"📁 找到 {len(all_files)} 个代码文件")
        
        # 批量检测
        results = []
        for i, filepath in enumerate(all_files, 1):
            logger.info(f"📝 处理 {i}/{len(all_files)}: {os.path.basename(filepath)}")
            result = self.detect_file(filepath)
            results.append(result)
        
        # 步骤9: 结果聚合（批量）
        logger.info("📊 聚合批量结果")
        batch_result = self.result_aggregator.aggregate_batch_results(results)
        
        return batch_result
    
    def detect_and_output(self, 
                         input_paths: List[str],
                         output_formats: List[str] = ['json'],
                         output_filename: str = None,
                         recursive: bool = False) -> Dict[str, str]:
        """
        检测并输出结果
        
        Args:
            input_paths: 输入路径列表
            output_formats: 输出格式列表
            output_filename: 输出文件名
            recursive: 是否递归处理
            
        Returns:
            Dict: 输出文件路径映射
        """
        # 执行检测
        results = self.detect_batch(input_paths, recursive)
        
        # 步骤10: 输出系统
        logger.info("💾 步骤10: 输出结果")
        output_files = self.output_system.output_multiple_formats(
            results, output_formats, output_filename
        )
        
        # 显示统计信息
        stats = results['statistics']
        logger.info("📈 检测统计:")
        logger.info(f"   文件总数: {stats['total_files']}")
        logger.info(f"   成功处理: {stats['successful_files']}")
        logger.info(f"   代码行数: {stats['total_code_lines']}")
        logger.info(f"   AI行数: {stats['total_ai_lines']}")
        logger.info(f"   AI比例: {stats['overall_ai_percentage']}%")
        
        return output_files
    
    def set_threshold(self, threshold: float):
        """设置检测阈值"""
        self.threshold_filter.set_threshold(threshold)
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """获取架构信息"""
        return {
            "architecture_flow": [
                "文件解析器",
                "行级特征提取",
                "文件级特征提取", 
                "特征融合",
                "行间关系建模",
                "CodeBERT编码",
                "行分类器",
                "阈值过滤",
                "结果聚合",
                "输出系统"
            ],
            "modules": {
                "file_parser": "FileParser",
                "line_feature_extractor": "LineFeatureExtractor (19维特征)",
                "file_feature_extractor": "FileFeatureExtractor (14维特征)",
                "feature_fusion": "FeatureFusion (19+14 → 128)",
                "inter_line_modeling": "InterLineRelationship (自注意力)",
                "codebert_encoder": "CodeBERTEncoder (microsoft/codebert-base)",
                "line_classifier": "LineClassifier (256 → 1)",
                "threshold_filter": f"ThresholdFilter (阈值: {self.threshold_filter.get_threshold()})",
                "result_aggregator": "ResultAggregator",
                "output_system": "OutputSystem (多格式输出)"
            },
            "supported_languages": list(self.file_parser.supported_extensions),
            "output_formats": self.output_system.get_supported_formats()
        }


def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(
        description="模块化AI代码检测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
架构流程:
代码文件/文件夹 → 文件解析器 → 行级特征提取 → 文件级特征提取 → 特征融合 → 
行间关系建模 → CodeBERT编码 → 行分类器 → 阈值过滤 → 结果聚合 → 输出系统

示例:
  %(prog)s --input file.py --output results
  %(prog)s --input src/ --recursive --formats json csv html
  %(prog)s --input *.py --threshold 0.7 --output-dir ./reports
        """
    )
    
    parser.add_argument("--input", type=str, nargs='+',
                       help="输入文件、目录或模式")
    parser.add_argument("--output", type=str, default="ai_detection_results",
                       help="输出文件名（不含扩展名）")
    parser.add_argument("--output-dir", type=str, default="./output",
                       help="输出目录")
    parser.add_argument("--formats", type=str, nargs='+', 
                       default=['json'], choices=['json', 'csv', 'xml', 'txt', 'html'],
                       help="输出格式")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="AI检测阈值 (0.0-1.0)")
    parser.add_argument("--recursive", "-r", action="store_true",
                       help="递归处理目录")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="详细输出")
    parser.add_argument("--info", action="store_true",
                       help="显示架构信息")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🚀 模块化AI代码检测系统")
    print("=" * 80)
    print("架构流程: 文件解析器 → 特征提取 → 特征融合 → 行间建模 → CodeBERT → 分类器 → 过滤 → 聚合 → 输出")
    print("=" * 80)
    
    # 初始化检测器
    detector = ModularAIDetector(threshold=args.threshold, output_dir=args.output_dir)
    
    # 显示架构信息
    if args.info:
        arch_info = detector.get_architecture_info()
        print("\n📋 架构信息:")
        print("流程步骤:")
        for i, step in enumerate(arch_info['architecture_flow'], 1):
            print(f"  {i}. {step}")
        print("\n模块详情:")
        for name, desc in arch_info['modules'].items():
            print(f"  {name}: {desc}")
        print(f"\n支持语言: {', '.join(arch_info['supported_languages'])}")
        print(f"输出格式: {', '.join(arch_info['output_formats'])}")
        return
    
    # 检查必需的参数
    if not args.input:
        parser.error("--input 是必需的参数（除非使用 --info）")
    
    # 执行检测
    print(f"\n📁 输入: {args.input}")
    print(f"🎯 阈值: {args.threshold}")
    print(f"📊 格式: {', '.join(args.formats)}")
    print(f"🔄 递归: {args.recursive}")
    
    try:
        output_files = detector.detect_and_output(
            args.input, 
            args.formats, 
            args.output,
            args.recursive
        )
        
        print(f"\n💾 输出文件:")
        for format_type, filepath in output_files.items():
            if not filepath.startswith("Error:"):
                print(f"  {format_type}: {filepath}")
            else:
                print(f"  {format_type}: ❌ {filepath}")
        
        print(f"\n✨ 检测完成!")
        
    except Exception as e:
        logger.error(f"检测失败: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()


if __name__ == "__main__":
    main() 