#!/usr/bin/env python3
"""
模块化AI检测系统测试脚本
验证每个模块是否正常工作
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# 添加模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_modules():
    """测试所有模块"""
    print("🚀 测试模块化AI检测系统")
    print("=" * 60)
    
    # 测试1: 文件解析器
    print("📄 测试1: 文件解析器")
    try:
        from modules.file_parser import FileParser
        parser = FileParser()
        
        # 创建测试文件
        test_code = '''def hello():
    print("Hello World")
    return True
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            test_file = f.name
        
        result = parser.parse_file(test_file)
        print(f"   ✅ 文件解析成功: {len(result['lines'])} 行")
        os.unlink(test_file)
    except Exception as e:
        print(f"   ❌ 文件解析失败: {e}")
    
    # 测试2: 特征提取
    print("🔧 测试2: 特征提取")
    try:
        from modules.feature_extraction import LineFeatureExtractor, FileFeatureExtractor
        line_extractor = LineFeatureExtractor()
        file_extractor = FileFeatureExtractor()
        
        # 模拟行信息
        line_info = {
            'content': 'def calculate_sum(a: int, b: int) -> int:',
            'line_number': 1,
            'is_empty': False,
            'indent_level': 0
        }
        file_context = {'total_lines': 3}
        
        line_features = line_extractor.extract_features(line_info, file_context)
        print(f"   ✅ 行级特征提取: {len(line_features)} 维")
        
        # 模拟文件信息
        file_info = {
            'lines': [
                {'content': 'def hello():', 'is_empty': False, 'indent_level': 0},
                {'content': '    return True', 'is_empty': False, 'indent_level': 4}
            ]
        }
        file_features = file_extractor.extract_features(file_info)
        print(f"   ✅ 文件级特征提取: {len(file_features)} 维")
    except Exception as e:
        print(f"   ❌ 特征提取失败: {e}")
    
    # 测试3: 特征融合
    print("🔗 测试3: 特征融合")
    try:
        from modules.feature_fusion import FeatureFusion
        import torch
        
        fusion = FeatureFusion(line_feature_dim=19, file_feature_dim=14, output_dim=128)
        
        line_features = torch.randn(5, 19)  # 5行，19维特征
        file_features = torch.randn(5, 14)  # 5行，14维特征
        
        fused = fusion(line_features, file_features)
        print(f"   ✅ 特征融合成功: {fused.shape}")
    except Exception as e:
        print(f"   ❌ 特征融合失败: {e}")
    
    # 测试4: 行间关系建模
    print("🧠 测试4: 行间关系建模")
    try:
        from modules.inter_line_modeling import InterLineRelationship
        import torch
        
        inter_line = InterLineRelationship(feature_dim=128, hidden_dim=64)
        
        features = torch.randn(1, 5, 128)  # 1个文件，5行，128维特征
        positions = torch.tensor([[0, 1, 2, 3, 4]])  # 行位置
        
        contextual = inter_line(features, positions)
        print(f"   ✅ 行间关系建模成功: {contextual.shape}")
    except Exception as e:
        print(f"   ❌ 行间关系建模失败: {e}")
    
    # 测试5: CodeBERT编码（可能需要网络）
    print("🤖 测试5: CodeBERT编码")
    try:
        from modules.codebert_encoding import CodeBERTEncoder
        
        encoder = CodeBERTEncoder(output_dim=128)
        
        code_lines = [
            "def hello():",
            "    return True",
            "print('Hello')"
        ]
        
        embeddings = encoder.encode_lines(code_lines)
        print(f"   ✅ CodeBERT编码成功: {embeddings.shape}")
    except Exception as e:
        print(f"   ⚠️ CodeBERT编码跳过: {e}")
    
    # 测试6: 分类器
    print("🎯 测试6: 分类器")
    try:
        from modules.classification import LineClassifier
        import torch
        
        classifier = LineClassifier(input_dim=256, hidden_dims=[128, 64])
        
        features = torch.randn(5, 256)  # 5行，256维特征
        probabilities = classifier(features)
        print(f"   ✅ 分类器预测成功: {probabilities.shape}")
    except Exception as e:
        print(f"   ❌ 分类器失败: {e}")
    
    # 测试7: 阈值过滤
    print("⚖️ 测试7: 阈值过滤")
    try:
        from modules.threshold_filter import ThresholdFilter
        import numpy as np
        
        filter_module = ThresholdFilter(threshold=0.5)
        
        probabilities = np.array([0.2, 0.6, 0.8, 0.3, 0.9])
        predictions = filter_module.filter(probabilities)
        print(f"   ✅ 阈值过滤成功: {predictions}")
    except Exception as e:
        print(f"   ❌ 阈值过滤失败: {e}")
    
    # 测试8: 结果聚合
    print("📋 测试8: 结果聚合")
    try:
        from modules.result_aggregation import ResultAggregator
        
        aggregator = ResultAggregator()
        
        # 模拟文件信息
        file_info = {
            'filepath': 'test.py',
            'total_lines': 3,
            'lines': [
                {'line_number': 1, 'content': 'def hello():', 'is_empty': False},
                {'line_number': 2, 'content': '    return True', 'is_empty': False},
                {'line_number': 3, 'content': '', 'is_empty': True}
            ]
        }
        
        ai_probabilities = [0.3, 0.7, 0.0]
        ai_predictions = [False, True, False]
        
        result = aggregator.aggregate_file_results(file_info, ai_probabilities, ai_predictions)
        print(f"   ✅ 结果聚合成功: AI比例 {result['summary']['ai_percentage']}%")
    except Exception as e:
        print(f"   ❌ 结果聚合失败: {e}")
    
    # 测试9: 输出系统
    print("💾 测试9: 输出系统")
    try:
        from modules.output_system import OutputSystem
        
        output_system = OutputSystem(output_dir="./test_output")
        
        # 模拟检测结果
        test_data = {
            "results": [
                {
                    "file_path": "test.py",
                    "success": True,
                    "lines": [
                        {"line_number": 1, "content": "def hello():", "ai_prob": 0.3, "is_ai": False}
                    ],
                    "summary": {
                        "total_lines": 1,
                        "code_lines": 1,
                        "ai_lines": 0,
                        "ai_percentage": 0.0,
                        "average_ai_prob": 0.3
                    }
                }
            ],
            "statistics": {
                "total_files": 1,
                "successful_files": 1,
                "total_lines": 1,
                "total_code_lines": 1,
                "total_ai_lines": 0,
                "overall_ai_percentage": 0.0
            },
            "metadata": {
                "timestamp": "2024-01-01T00:00:00"
            }
        }
        
        json_file = output_system.output_json(test_data, "test_result.json")
        print(f"   ✅ JSON输出成功: {json_file}")
        
        # 清理测试文件
        if os.path.exists(json_file):
            os.unlink(json_file)
        if os.path.exists("./test_output"):
            import shutil
            shutil.rmtree("./test_output")
    except Exception as e:
        print(f"   ❌ 输出系统失败: {e}")
    
    print("\n✨ 模块测试完成!")


def test_integration():
    """测试集成系统"""
    print("\n🔧 测试集成系统")
    print("-" * 40)
    
    try:
        # 导入模块化检测器
        from modular_ai_detector import ModularAIDetector
        
        # 创建检测器实例（使用随机权重）
        detector = ModularAIDetector()
        
        # 获取架构信息
        arch_info = detector.get_architecture_info()
        print("📋 架构信息:")
        for i, step in enumerate(arch_info['architecture_flow'], 1):
            print(f"  {i}. {step}")
        
        # 创建测试文件
        test_code = '''def calculate_fibonacci(n: int) -> int:
    """Calculate fibonacci number with comprehensive error handling."""
    if not isinstance(n, int):
        raise TypeError("Input must be an integer")
    
    if n < 0:
        raise ValueError("Input must be non-negative")
    
    if n <= 1:
        return n
    
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def simple_add(a, b):
    return a + b
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            test_file = f.name
        
        # 测试文件检测
        print(f"\n🔍 测试文件检测: {os.path.basename(test_file)}")
        result = detector.detect_file(test_file)
        
        if result['success']:
            summary = result['summary']
            print(f"   ✅ 检测成功!")
            print(f"   📊 统计信息:")
            print(f"      总行数: {summary['total_lines']}")
            print(f"      代码行数: {summary['code_lines']}")
            print(f"      AI行数: {summary['ai_lines']}")
            print(f"      AI比例: {summary['ai_percentage']}%")
            print(f"      平均AI概率: {summary['average_ai_prob']}")
            
            # 显示部分检测结果
            print(f"   📝 部分检测结果:")
            for line in result['lines'][:5]:
                indicator = "🤖" if line['is_ai'] else "👤"
                print(f"      {indicator} 行{line['line_number']}: {line['ai_prob']:.3f} - {line['content'][:50]}...")
        else:
            print(f"   ❌ 检测失败: {result.get('error', 'Unknown error')}")
        
        # 清理测试文件
        os.unlink(test_file)
        
        print("   ✅ 集成测试完成!")
        
    except Exception as e:
        print(f"   ❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_modules()
    test_integration() 