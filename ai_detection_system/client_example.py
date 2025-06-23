#!/usr/bin/env python3
"""
AI代码检测系统 - 客户端示例
展示如何调用API接口进行代码检测
"""

import requests
import json
from pathlib import Path

# API服务器配置
API_BASE_URL = "http://localhost:8000/v1"

def test_health_check():
    """测试健康检查接口"""
    print("🏥 测试健康检查接口...")
    
    response = requests.get(f"{API_BASE_URL}/health")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 服务状态: {result['status']}")
        print(f"📖 API版本: {result['version']}")
        print(f"🤖 模型加载: {result['model_loaded']}")
    else:
        print(f"❌ 健康检查失败: {response.status_code}")

def test_code_snippet_detection():
    """测试代码片段检测"""
    print("\n🔍 测试代码片段检测...")
    
    # 示例代码片段
    code_snippets = [
        {
            "content": """def calculate_comprehensive_metrics(data: List[float]) -> Dict[str, float]:
    \"\"\"
    Calculate comprehensive statistical metrics for the given data.
    
    Args:
        data: List of numerical values for analysis
        
    Returns:
        Dictionary containing various statistical metrics
    \"\"\"
    if not data:
        logger.warning("Empty data provided for metrics calculation")
        return {}
    
    try:
        metrics = {
            'mean': sum(data) / len(data),
            'median': sorted(data)[len(data) // 2],
            'std_dev': (sum((x - sum(data)/len(data))**2 for x in data) / len(data))**0.5
        }
        logger.info(f"Successfully calculated metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise""",
            "filename": "ai_generated.py",
            "language": "python"
        },
        {
            "content": """def add(a, b):
    return a + b

def subtract(x, y):
    return x - y""",
            "filename": "human_written.py",
            "language": "python"
        }
    ]
    
    # 发送检测请求
    payload = {
        "code_snippets": code_snippets,
        "threshold": 0.5,
        "output_format": "json"
    }
    
    response = requests.post(f"{API_BASE_URL}/detect/code", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 检测成功!")
        print(f"📊 处理时间: {result['processing_time']:.2f}秒")
        print(f"📈 统计信息:")
        stats = result['statistics']
        print(f"   - 总代码片段: {stats['total_snippets']}")
        print(f"   - 成功检测: {stats['successful_detections']}")
        print(f"   - 总代码行数: {stats['total_code_lines']}")
        print(f"   - AI生成行数: {stats['total_ai_lines']}")
        print(f"   - AI比例: {stats['overall_ai_percentage']:.1f}%")
        
        # 显示每个片段的结果
        for i, snippet_result in enumerate(result['results']):
            if snippet_result['success']:
                summary = snippet_result['summary']
                print(f"\n📝 代码片段 {i+1} ({snippet_result['filename']}):")
                print(f"   - 总行数: {summary['total_lines']}")
                print(f"   - AI行数: {summary['ai_lines']}")
                print(f"   - AI比例: {summary['ai_percentage']:.1f}%")
                print(f"   - 平均AI概率: {summary['average_ai_prob']:.3f}")
    else:
        print(f"❌ 检测失败: {response.status_code}")
        print(f"错误信息: {response.text}")

def test_file_upload():
    """测试文件上传检测"""
    print("\n📁 测试文件上传检测...")
    
    # 创建测试文件
    test_file_content = """#!/usr/bin/env python3
\"\"\"
Advanced data processing module with comprehensive functionality.
\"\"\"

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class DataProcessor:
    \"\"\"
    Comprehensive data processing class with advanced algorithms.
    \"\"\"
    
    def __init__(self, configuration: Dict[str, Any]):
        \"\"\"
        Initialize the data processor with configuration.
        
        Args:
            configuration: Processing configuration parameters
        \"\"\"
        self.config = configuration
        logger.info("DataProcessor initialized successfully")
    
    def process_data(self, input_data: List[Dict]) -> Dict[str, Any]:
        \"\"\"
        Process input data using advanced algorithms.
        
        Args:
            input_data: List of data dictionaries to process
            
        Returns:
            Processed data results
        \"\"\"
        try:
            logger.info(f"Processing {len(input_data)} data items")
            
            # Implementation of comprehensive processing logic
            processed_results = []
            for item in input_data:
                processed_item = self._process_single_item(item)
                processed_results.append(processed_item)
            
            logger.info("Data processing completed successfully")
            return {"results": processed_results, "status": "success"}
            
        except Exception as e:
            logger.error(f"Error during data processing: {str(e)}")
            raise ProcessingException(f"Failed to process data: {str(e)}")
    
    def _process_single_item(self, item: Dict) -> Dict:
        \"\"\"Process a single data item.\"\"\"
        return {"processed": True, "original": item}
"""
    
    # 保存测试文件
    test_file_path = Path("test_file.py")
    test_file_path.write_text(test_file_content)
    
    try:
        # 上传文件检测
        with open(test_file_path, 'rb') as f:
            files = {'file': ('test_file.py', f, 'text/plain')}
            data = {'threshold': 0.6, 'output_format': 'json'}
            
            response = requests.post(f"{API_BASE_URL}/detect/file", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 文件检测成功!")
            print(f"📊 处理时间: {result['processing_time']:.2f}秒")
            
            file_result = result['result']
            if file_result['success']:
                summary = file_result['summary']
                print(f"📝 文件: {file_result['original_filename']}")
                print(f"   - 文件大小: {file_result['file_size']} 字节")
                print(f"   - 总行数: {summary['total_lines']}")
                print(f"   - 代码行数: {summary['code_lines']}")
                print(f"   - AI行数: {summary['ai_lines']}")
                print(f"   - AI比例: {summary['ai_percentage']:.1f}%")
                
                # 显示部分检测结果
                print(f"\n🔍 部分行检测结果:")
                for line_result in file_result['lines'][:5]:  # 显示前5行
                    print(f"   行{line_result['line_number']}: AI概率={line_result['ai_prob']:.3f}, "
                          f"AI生成={'是' if line_result['is_ai'] else '否'}")
                    print(f"      内容: {line_result['content'][:50]}...")
        else:
            print(f"❌ 文件检测失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    finally:
        # 清理测试文件
        if test_file_path.exists():
            test_file_path.unlink()

def test_architecture_info():
    """测试获取架构信息"""
    print("\n🏗️ 测试获取架构信息...")
    
    response = requests.get(f"{API_BASE_URL}/info/architecture")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 架构信息获取成功!")
        arch_info = result['architecture']
        print(f"📋 架构流程:")
        for step in arch_info['pipeline']:
            print(f"   {step['step']}: {step['name']} - {step['description']}")
    else:
        print(f"❌ 获取架构信息失败: {response.status_code}")

def test_threshold_update():
    """测试更新阈值"""
    print("\n⚙️ 测试更新检测阈值...")
    
    # 更新阈值为0.7
    data = {'threshold': 0.7}
    response = requests.post(f"{API_BASE_URL}/config/threshold", data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 阈值更新成功!")
        print(f"📝 {result['message']}")
    else:
        print(f"❌ 阈值更新失败: {response.status_code}")

def main():
    """运行所有测试"""
    print("🧪 AI代码检测API客户端测试")
    print("=" * 50)
    
    try:
        # 测试各个接口
        test_health_check()
        test_code_snippet_detection()
        test_file_upload()
        test_architecture_info()
        test_threshold_update()
        
        print("\n🎉 所有测试完成!")
        
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到API服务器!")
        print("请确保服务器正在运行: python api_server.py")
    except Exception as e:
        print(f"❌ 测试过程中出错: {str(e)}")

if __name__ == "__main__":
    main() 