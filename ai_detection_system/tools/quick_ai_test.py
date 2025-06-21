#!/usr/bin/env python3
"""
快速AI检测测试脚本 - 简化版本
直接调用主检测器进行功能验证
"""

import os
import sys
import json
import subprocess
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_test_file():
    """创建测试代码文件"""
    test_code = '''# Test file for AI detection
import os
import sys
from typing import List, Dict

def fibonacci(n: int) -> int:
    """Calculate fibonacci number using recursion."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers with comprehensive logging."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, x, y):
        return x * y

if __name__ == "__main__":
    calc = Calculator()
    result = calc.add(10, 5)
    print(f"Result: {result}")
'''
    
    with open('test_sample.py', 'w', encoding='utf-8') as f:
        f.write(test_code)
    
    logger.info("✅ Created test file: test_sample.py")

def test_ai_detector():
    """测试AI检测器功能"""
    logger.info("🤖 Testing AI Code Detector...")
    
    try:
        # 使用subprocess调用主检测器
        cmd = [
            sys.executable, 
            'core/ai_code_detector.py',
            '--model', 'models/ai_detector.pt',
            '--input', 'test_sample.py',
            '--output', 'test_results.json',
            '--quiet'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            logger.info("✅ Detection successful!")
            
            # 读取结果
            if os.path.exists('test_results.json'):
                with open('test_results.json', 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'results' in data and data['results']:
                    file_result = data['results'][0]
                    if 'lines' in file_result:
                        lines = file_result['lines']
                        code_lines = [line for line in lines if line['type'] == 'code']
                        
                        logger.info(f"📊 Detected {len(code_lines)} code lines")
                        
                        # 创建您要求的格式
                        output_format = []
                        for line in code_lines:
                            output_format.append({
                                "line_number": line["line_number"],
                                "content": line["content"],
                                "ai_prob": line["ai_prob"],
                                "is_ai": line["is_ai"]
                            })
                        
                        # 保存为您要求的格式
                        with open('line_detection_result.json', 'w', encoding='utf-8') as f:
                            json.dump(output_format, f, indent=2, ensure_ascii=False)
                        
                        logger.info("💾 Results saved in your requested format: line_detection_result.json")
                        
                        # 显示前3行示例
                        logger.info("📋 Sample output format:")
                        for i, item in enumerate(output_format[:3]):
                            logger.info(f"  Line {item['line_number']}: AI={item['ai_prob']:.3f} | {item['content'][:50]}")
                        
                        return True
        
        logger.error(f"❌ Detection failed: {result.stderr}")
        return False
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

def cleanup():
    """清理测试文件"""
    files_to_remove = ['test_sample.py', 'test_results.json', 'line_detection_result.json']
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
    logger.info("🧹 Cleanup completed")

def main():
    logger.info("🚀 Quick AI Detection Test")
    logger.info("=" * 40)
    
    try:
        # 1. 创建测试文件
        logger.info("📝 Step 1: Creating test file...")
        create_test_file()
        
        # 2. 测试检测器
        logger.info("🔍 Step 2: Testing AI detector...")
        success = test_ai_detector()
        
        if success:
            logger.info("✅ All tests passed!")
            logger.info("\n🎯 Usage:")
            logger.info("python core/ai_code_detector.py --model models/ai_detector.pt --input your_file.py --output results.json")
        else:
            logger.error("❌ Tests failed!")
            return 1
        
    except Exception as e:
        logger.error(f"Test error: {e}")
        return 1
    
    finally:
        # 3. 清理
        cleanup()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 