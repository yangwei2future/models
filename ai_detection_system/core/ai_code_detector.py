#!/usr/bin/env python3
"""
AI代码检测器 - 基于CodeBERT的逐行检测
使用CodeBERT编码器 + 特征融合 + 分类器架构

使用方法:
    python ai_code_detector.py --model models/ai_detector.pt --input file.py
    python ai_code_detector.py --model models/ai_detector.pt --input src/ --recursive
"""

import os
import sys
import json
import argparse
import logging
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel

# 添加路径
sys.path.append('parsers/fileParser')
sys.path.append('.')

def setup_logging(verbose: bool = False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

class CodeBERTAIDetector(nn.Module):
    """基于CodeBERT的AI代码检测器 - 逐行检测"""
    
    def __init__(self, model_name: str = "microsoft/codebert-base", feature_dim: int = 10, hidden_dim: int = 256):
        super().__init__()
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # CodeBERT tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.codebert = AutoModel.from_pretrained(model_name)
        
        # 冻结CodeBERT的部分参数（可选）
        # for param in self.codebert.parameters():
        #     param.requires_grad = False
        
        # 特征维度
        self.codebert_dim = self.codebert.config.hidden_size  # 768 for base model
        
        # 特征融合层
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        self.code_projection = nn.Linear(self.codebert_dim, hidden_dim)
        
        # 融合后的分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 特征 + CodeBERT输出
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),  # 输出AI概率
            nn.Sigmoid()
        )
    
    def extract_line_features(self, line: str, line_number: int, total_lines: int) -> List[float]:
        """提取行级特征"""
        line = line.strip()
        
        # 基础特征
        length = len(line)
        indent_level = len(line) - len(line.lstrip()) if line else 0
        
        # 内容特征
        has_comment = '#' in line or '//' in line or '/*' in line
        has_string = '"' in line or "'" in line or '`' in line
        has_number = any(c.isdigit() for c in line)
        has_operator = any(op in line for op in ['=', '+', '-', '*', '/', '<', '>', '!', '&', '|'])
        
        # 位置特征
        relative_position = line_number / total_lines if total_lines > 0 else 0
        
        # AI生成代码的常见模式
        ai_patterns = [
            'generate', 'create', 'implement', 'function', 'method',
            'algorithm', 'process', 'handle', 'manage', 'execute',
            'calculate', 'compute', 'optimize', 'initialize', 'configure'
        ]
        has_ai_pattern = any(pattern in line.lower() for pattern in ai_patterns)
        
        # 复杂度和风格特征
        complexity = line.count('(') + line.count('[') + line.count('{')
        word_density = len(line.split()) / max(len(line), 1)
        
        return [
            length / 100.0,  # 归一化长度
            indent_level / 20.0,  # 归一化缩进
            float(has_comment),
            float(has_string),
            float(has_number),
            float(has_operator),
            relative_position,
            float(has_ai_pattern),
            complexity / 10.0,
            word_density
        ]
    
    def encode_with_codebert(self, line: str) -> torch.Tensor:
        """使用CodeBERT编码代码行"""
        # 处理空行
        if not line.strip():
            line = "[EMPTY_LINE]"
        
        # Tokenize
        inputs = self.tokenizer(
            line,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # 移动到正确的设备
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # CodeBERT编码
        with torch.no_grad():
            outputs = self.codebert(**inputs)
        
        # 使用[CLS] token的表示
        code_embedding = outputs.last_hidden_state[:, 0, :]  # [1, hidden_size]
        
        return code_embedding
    
    def forward(self, line: str, line_number: int, total_lines: int) -> float:
        """前向传播 - 预测单行AI概率"""
        device = next(self.parameters()).device
        
        # 1. 提取行级特征
        line_features = self.extract_line_features(line, line_number, total_lines)
        line_features_tensor = torch.tensor(line_features, device=device, dtype=torch.float32).unsqueeze(0)
        
        # 2. CodeBERT编码
        code_embedding = self.encode_with_codebert(line)
        
        # 3. 特征投影
        projected_features = self.feature_projection(line_features_tensor)  # [1, hidden_dim]
        projected_code = self.code_projection(code_embedding)  # [1, hidden_dim]
        
        # 4. 特征融合
        fused_features = torch.cat([projected_features, projected_code], dim=1)  # [1, hidden_dim*2]
        
        # 5. 分类预测
        ai_prob = self.classifier(fused_features).item()
        
        return ai_prob

class AICodeDetector:
    """AI代码检测器主类"""
    
    def __init__(self, model_path: str, threshold: float = 0.5):
        self.model_path = model_path
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        
        # 支持的文件扩展名
        self.supported_extensions = {
            '.py', '.java', '.js', '.jsx', '.ts', '.tsx',
            '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx',
            '.go', '.rs', '.rb', '.php', '.swift', '.kt'
        }
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            self.model = CodeBERTAIDetector()
            
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                self.logger.info(f"✅ CodeBERT model loaded from {self.model_path}")
            else:
                self.logger.warning(f"Model file not found: {self.model_path}, using random weights")
            
            self.model.eval()
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def detect_file(self, filepath: str) -> Dict[str, Any]:
        """检测单个文件的每一行"""
        try:
            # 读取文件
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            if not lines:
                return {
                    "file_path": filepath,
                    "success": False,
                    "error": "Empty file"
                }
            
            # 逐行检测
            line_results = []
            total_lines = len(lines)
            
            for i, line in enumerate(lines, 1):
                content = line.rstrip('\n\r')
                
                # 跳过空行
                if not content.strip():
                    line_results.append({
                        "line_number": i,
                        "content": content,
                        "ai_prob": 0.0,
                        "is_ai": False,
                        "type": "empty"
                    })
                    continue
                
                # AI概率预测
                with torch.no_grad():
                    ai_prob = self.model(content, i, total_lines)
                
                is_ai = ai_prob > self.threshold
                
                line_results.append({
                    "line_number": i,
                    "content": content,
                    "ai_prob": round(ai_prob, 3),
                    "is_ai": is_ai,
                    "type": "code" if content.strip() else "empty"
                })
            
            # 统计信息
            ai_lines = sum(1 for r in line_results if r["is_ai"])
            total_code_lines = sum(1 for r in line_results if r["type"] == "code")
            
            return {
                "file_path": filepath,
                "success": True,
                "lines": line_results,
                "summary": {
                    "total_lines": total_lines,
                    "code_lines": total_code_lines,
                    "ai_lines": ai_lines,
                    "ai_percentage": round((ai_lines / total_code_lines * 100) if total_code_lines > 0 else 0, 1),
                    "average_ai_prob": round(np.mean([r["ai_prob"] for r in line_results if r["type"] == "code"]), 3) if total_code_lines > 0 else 0.0
                }
            }
            
        except Exception as e:
            return {
                "file_path": filepath,
                "success": False,
                "error": str(e)
            }
    
    def is_code_file(self, filepath: str) -> bool:
        """判断是否为代码文件"""
        _, ext = os.path.splitext(filepath.lower())
        return ext in self.supported_extensions
    
    def collect_files(self, input_paths: List[str], recursive: bool = False) -> List[str]:
        """收集要处理的文件"""
        files = []
        
        for input_path in input_paths:
            # 处理通配符
            if '*' in input_path or '?' in input_path:
                matched_files = glob.glob(input_path)
                files.extend([f for f in matched_files if os.path.isfile(f) and self.is_code_file(f)])
                continue
            
            if os.path.isfile(input_path):
                if self.is_code_file(input_path):
                    files.append(input_path)
                else:
                    self.logger.warning(f"Skipping unsupported file: {input_path}")
            
            elif os.path.isdir(input_path):
                if recursive:
                    for root, dirs, filenames in os.walk(input_path):
                        for filename in filenames:
                            filepath = os.path.join(root, filename)
                            if self.is_code_file(filepath):
                                files.append(filepath)
                else:
                    for filename in os.listdir(input_path):
                        filepath = os.path.join(input_path, filename)
                        if os.path.isfile(filepath) and self.is_code_file(filepath):
                            files.append(filepath)
            else:
                self.logger.warning(f"Path not found: {input_path}")
        
        return sorted(list(set(files)))
    
    def detect_batch(self, filepaths: List[str]) -> Dict[str, Any]:
        """批量检测多个文件"""
        results = []
        stats = {
            "total_files": len(filepaths),
            "successful_files": 0,
            "failed_files": 0,
            "total_lines": 0,
            "total_ai_lines": 0,
            "file_ai_percentages": []
        }
        
        self.logger.info(f"Processing {len(filepaths)} files...")
        
        for i, filepath in enumerate(filepaths, 1):
            self.logger.debug(f"Processing {i}/{len(filepaths)}: {filepath}")
            
            result = self.detect_file(filepath)
            results.append(result)
            
            if result["success"]:
                stats["successful_files"] += 1
                summary = result["summary"]
                stats["total_lines"] += summary["total_lines"]
                stats["total_ai_lines"] += summary["ai_lines"]
                stats["file_ai_percentages"].append(summary["ai_percentage"])
                
                self.logger.info(f"✅ {os.path.basename(filepath)} -> {summary['ai_percentage']}% AI ({summary['ai_lines']}/{summary['code_lines']} lines)")
            else:
                stats["failed_files"] += 1
                self.logger.warning(f"❌ {os.path.basename(filepath)} -> {result['error']}")
        
        # 计算总体统计
        if stats["successful_files"] > 0:
            stats["overall_ai_percentage"] = round((stats["total_ai_lines"] / (stats["total_lines"] - sum(r["summary"]["total_lines"] - r["summary"]["code_lines"] for r in results if r["success"])) * 100) if stats["total_lines"] > 0 else 0, 1)
            stats["average_file_ai_percentage"] = round(np.mean(stats["file_ai_percentages"]), 1)
        else:
            stats["overall_ai_percentage"] = 0.0
            stats["average_file_ai_percentage"] = 0.0
        
        return {
            "results": results,
            "statistics": stats,
            "metadata": {
                "model_path": self.model_path,
                "model_type": "CodeBERT-based",
                "threshold": self.threshold,
                "timestamp": datetime.now().isoformat()
            }
        }

def print_results(detection_results: Dict[str, Any], verbose: bool = False, show_lines: bool = False):
    """打印检测结果"""
    stats = detection_results["statistics"]
    
    print("\n" + "=" * 80)
    print("🤖 AI CODE DETECTION RESULTS (CodeBERT-based)")
    print("=" * 80)
    
    # 总体统计
    print(f"\n📊 Summary:")
    print(f"   Total files: {stats['total_files']}")
    print(f"   Successful: {stats['successful_files']}")
    print(f"   Failed: {stats['failed_files']}")
    print(f"   Total lines: {stats['total_lines']}")
    print(f"   AI-generated lines: {stats['total_ai_lines']}")
    print(f"   Overall AI percentage: {stats['overall_ai_percentage']}%")
    print(f"   Average per file: {stats['average_file_ai_percentage']}%")
    
    # 详细结果
    if verbose or show_lines:
        print(f"\n📄 Detailed Results:")
        print("-" * 80)
        
        for result in detection_results["results"]:
            if not result["success"]:
                print(f"\n❌ {result['file_path']}")
                print(f"   Error: {result['error']}")
                continue
            
            summary = result["summary"]
            print(f"\n📁 {result['file_path']}")
            print(f"   Lines: {summary['total_lines']} total, {summary['code_lines']} code")
            print(f"   AI detection: {summary['ai_lines']} lines ({summary['ai_percentage']}%)")
            print(f"   Average AI probability: {summary['average_ai_prob']}")
            
            # 显示逐行结果
            if show_lines:
                print("   Line-by-line analysis:")
                for line_result in result["lines"]:
                    if line_result["type"] == "code" and line_result["ai_prob"] > 0.3:  # 只显示可能的AI行
                        status = "🤖 AI" if line_result["is_ai"] else "👤 Human"
                        print(f"     {line_result['line_number']:3d}: {status} ({line_result['ai_prob']:.3f}) | {line_result['content'][:60]}")

def main():
    parser = argparse.ArgumentParser(
        description="AI Code Detection Tool - CodeBERT-based line-by-line detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model models/ai_detector.pt --input file.py
  %(prog)s --model models/ai_detector.pt --input src/ --recursive
  %(prog)s --model models/ai_detector.pt --input "*.py" --output results.json --show-lines
        """
    )
    
    # 必需参数
    parser.add_argument("--model", type=str, required=True,
                       help="Path to AI detection model file (.pt)")
    parser.add_argument("--input", type=str, nargs='+', required=True,
                       help="Input files, directories, or patterns to process")
    
    # 可选参数
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file to save results")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="AI detection threshold (0.0-1.0, default: 0.5)")
    parser.add_argument("--recursive", "-r", action="store_true",
                       help="Recursively process subdirectories")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed results")
    parser.add_argument("--show-lines", action="store_true",
                       help="Show line-by-line AI detection results")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress progress messages")
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.verbose and not args.quiet)
    
    if not args.quiet:
        print("🤖 AI Code Detection Tool (CodeBERT-based)")
        print("-" * 50)
        print(f"Model: {args.model}")
        print(f"Input: {args.input}")
        print(f"Threshold: {args.threshold}")
        print(f"Recursive: {args.recursive}")
    
    try:
        # 初始化检测器
        detector = AICodeDetector(args.model, args.threshold)
        
        # 收集文件
        files = detector.collect_files(args.input, args.recursive)
        
        if not files:
            logger.error("No supported code files found to process")
            return 1
        
        if not args.quiet:
            logger.info(f"Found {len(files)} code files to process")
        
        # 执行检测
        results = detector.detect_batch(files)
        
        # 显示结果
        if not args.quiet:
            print_results(results, args.verbose, args.show_lines)
        
        # 保存结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Results saved to {args.output}")
        
        # 返回状态码
        stats = results["statistics"]
        if stats["failed_files"] > 0 and stats["successful_files"] == 0:
            return 1  # 全部失败
        else:
            return 0  # 成功
        
    except KeyboardInterrupt:
        logger.info("⏹️  Processing interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 