#!/usr/bin/env python3
"""
数据集加载器 - AI代码检测训练数据
用于加载和处理Human vs AI代码样本数据集
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeDatasetLoader:
    """AI代码检测数据集加载器"""
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化数据集加载器
        
        Args:
            data_dir: 数据集目录路径
        """
        self.data_dir = Path(data_dir)
        self.labels_file = self.data_dir / "dataset_labels.json"
        self.dataset_info = None
        self.labels_data = None
        
        # 加载标注数据
        self._load_labels()
    
    def _load_labels(self):
        """加载数据集标注信息"""
        try:
            if self.labels_file.exists():
                with open(self.labels_file, 'r', encoding='utf-8') as f:
                    self.labels_data = json.load(f)
                    self.dataset_info = self.labels_data.get('dataset_info', {})
                logger.info(f"Loaded dataset labels: {self.dataset_info.get('description', 'Unknown')}")
            else:
                logger.warning(f"Labels file not found: {self.labels_file}")
                self.labels_data = {}
                self.dataset_info = {}
        except Exception as e:
            logger.error(f"Failed to load labels: {e}")
            self.labels_data = {}
            self.dataset_info = {}
    
    def load_file_samples(self, filename: str) -> List[Dict[str, Any]]:
        """
        加载单个文件的代码样本
        
        Args:
            filename: 文件名
            
        Returns:
            代码行样本列表
        """
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        try:
            # 读取代码文件
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            samples = []
            file_labels = {}
            file_type = 'unknown'
            
            if self.labels_data and 'files' in self.labels_data:
                file_info = self.labels_data['files'].get(filename, {})
                file_labels = file_info.get('line_labels', {})
                file_type = file_info.get('file_type', 'unknown')
            
            for line_num, line_content in enumerate(lines, 1):
                line_content = line_content.rstrip('\n\r')
                
                # 跳过空行
                if not line_content.strip():
                    continue
                
                # 获取标注信息
                line_key = str(line_num)
                if line_key in file_labels:
                    # 使用标注的标签
                    label_info = file_labels[line_key]
                    is_ai = label_info.get('is_ai', False)
                    confidence = label_info.get('confidence', 0.5)
                else:
                    # 使用文件类型作为默认标签
                    if file_type == 'human':
                        is_ai = False
                        confidence = 0.8
                    elif file_type == 'ai':
                        is_ai = True
                        confidence = 0.8
                    else:
                        # 混合文件，使用启发式规则
                        is_ai = self._heuristic_ai_detection(line_content)
                        confidence = 0.6
                
                sample = {
                    'line': line_content,
                    'line_number': line_num,
                    'total_lines': len(lines),
                    'is_ai': is_ai,
                    'confidence': confidence,
                    'file_type': file_type,
                    'filename': filename
                }
                
                samples.append(sample)
            
            logger.info(f"Loaded {len(samples)} samples from {filename}")
            return samples
            
        except Exception as e:
            logger.error(f"Failed to load file {filename}: {e}")
            return []
    
    def _heuristic_ai_detection(self, line: str) -> bool:
        """
        启发式AI代码检测（用于未标注的行）
        
        Args:
            line: 代码行内容
            
        Returns:
            是否为AI生成的代码
        """
        ai_indicators = [
            '"""',  # 详细文档字符串
            'Args:',  # 参数文档
            'Returns:',  # 返回值文档
            'Raises:',  # 异常文档
            'typing import',  # 类型注解导入
            'Optional[',  # 可选类型
            'Dict[str, Any]',  # 复杂类型注解
            'comprehensive',  # AI常用词汇
            'sophisticated',
            'advanced',
            'implementation',
            'configuration',
            'initialization',
            'processing',
            'logging.getLogger',  # 详细日志配置
            'abstractmethod',  # 抽象方法
            '@dataclass',  # 数据类装饰器
        ]
        
        line_lower = line.lower()
        ai_score = sum(1 for indicator in ai_indicators if indicator.lower() in line_lower)
        
        # 如果包含2个或以上AI指标，认为是AI生成
        return ai_score >= 2
    
    def load_all_samples(self) -> List[Dict[str, Any]]:
        """
        加载所有代码样本
        
        Returns:
            所有代码行样本列表
        """
        all_samples = []
        
        # 获取所有Python文件
        python_files = [
            'human_code_samples.py',
            'ai_code_samples.py',
            'mixed_code_samples.py'
        ]
        
        for filename in python_files:
            samples = self.load_file_samples(filename)
            all_samples.extend(samples)
        
        logger.info(f"Total samples loaded: {len(all_samples)}")
        return all_samples
    
    def get_balanced_dataset(self, max_samples_per_class: Optional[int] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        获取平衡的数据集
        
        Args:
            max_samples_per_class: 每个类别的最大样本数
            
        Returns:
            (human_samples, ai_samples) 元组
        """
        all_samples = self.load_all_samples()
        
        # 分离Human和AI样本
        human_samples = [s for s in all_samples if not s['is_ai']]
        ai_samples = [s for s in all_samples if s['is_ai']]
        
        logger.info(f"Human samples: {len(human_samples)}, AI samples: {len(ai_samples)}")
        
        # 平衡数据集
        if max_samples_per_class:
            human_samples = human_samples[:max_samples_per_class]
            ai_samples = ai_samples[:max_samples_per_class]
        else:
            # 使用较小类别的大小
            min_size = min(len(human_samples), len(ai_samples))
            human_samples = human_samples[:min_size]
            ai_samples = ai_samples[:min_size]
        
        logger.info(f"Balanced dataset - Human: {len(human_samples)}, AI: {len(ai_samples)}")
        
        return human_samples, ai_samples
    
    def get_training_data(self, test_split: float = 0.2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        获取训练和测试数据
        
        Args:
            test_split: 测试集比例
            
        Returns:
            (train_samples, test_samples) 元组
        """
        human_samples, ai_samples = self.get_balanced_dataset()
        
        # 计算分割点
        human_split = int(len(human_samples) * (1 - test_split))
        ai_split = int(len(ai_samples) * (1 - test_split))
        
        # 分割数据
        train_samples = human_samples[:human_split] + ai_samples[:ai_split]
        test_samples = human_samples[human_split:] + ai_samples[ai_split:]
        
        # 打乱顺序
        import random
        random.shuffle(train_samples)
        random.shuffle(test_samples)
        
        logger.info(f"Training samples: {len(train_samples)}, Test samples: {len(test_samples)}")
        
        return train_samples, test_samples
    
    def export_training_data(self, output_file: str = "training_data.json"):
        """
        导出训练数据为JSON格式
        
        Args:
            output_file: 输出文件名
        """
        train_samples, test_samples = self.get_training_data()
        
        export_data = {
            'dataset_info': self.dataset_info,
            'training_data': train_samples,
            'test_data': test_samples,
            'statistics': {
                'total_training_samples': len(train_samples),
                'total_test_samples': len(test_samples),
                'training_human_samples': sum(1 for s in train_samples if not s['is_ai']),
                'training_ai_samples': sum(1 for s in train_samples if s['is_ai']),
                'test_human_samples': sum(1 for s in test_samples if not s['is_ai']),
                'test_ai_samples': sum(1 for s in test_samples if s['is_ai'])
            }
        }
        
        output_path = self.data_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Training data exported to {output_path}")
        
        return export_data
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        获取数据集统计信息
        
        Returns:
            数据集统计信息字典
        """
        all_samples = self.load_all_samples()
        
        human_samples = [s for s in all_samples if not s['is_ai']]
        ai_samples = [s for s in all_samples if s['is_ai']]
        
        # 按文件类型统计
        file_stats = {}
        for sample in all_samples:
            filename = sample['filename']
            if filename not in file_stats:
                file_stats[filename] = {'total': 0, 'human': 0, 'ai': 0}
            
            file_stats[filename]['total'] += 1
            if sample['is_ai']:
                file_stats[filename]['ai'] += 1
            else:
                file_stats[filename]['human'] += 1
        
        # 置信度统计
        confidence_levels = {
            'high': sum(1 for s in all_samples if s['confidence'] >= 0.8),
            'medium': sum(1 for s in all_samples if 0.6 <= s['confidence'] < 0.8),
            'low': sum(1 for s in all_samples if s['confidence'] < 0.6)
        }
        
        statistics = {
            'total_samples': len(all_samples),
            'human_samples': len(human_samples),
            'ai_samples': len(ai_samples),
            'human_percentage': len(human_samples) / len(all_samples) * 100 if all_samples else 0,
            'ai_percentage': len(ai_samples) / len(all_samples) * 100 if all_samples else 0,
            'file_statistics': file_stats,
            'confidence_distribution': confidence_levels,
            'average_confidence': sum(s['confidence'] for s in all_samples) / len(all_samples) if all_samples else 0
        }
        
        return statistics


def main():
    """演示数据集加载器的使用"""
    logger.info("🗂️  AI Code Detection Dataset Loader Demo")
    logger.info("=" * 50)
    
    # 初始化数据集加载器
    loader = CodeDatasetLoader()
    
    # 显示数据集统计信息
    stats = loader.get_dataset_statistics()
    logger.info("📊 Dataset Statistics:")
    logger.info(f"  Total samples: {stats['total_samples']}")
    logger.info(f"  Human samples: {stats['human_samples']} ({stats['human_percentage']:.1f}%)")
    logger.info(f"  AI samples: {stats['ai_samples']} ({stats['ai_percentage']:.1f}%)")
    logger.info(f"  Average confidence: {stats['average_confidence']:.3f}")
    
    logger.info("\n📁 File Statistics:")
    for filename, file_stat in stats['file_statistics'].items():
        logger.info(f"  {filename}: {file_stat['total']} total ({file_stat['human']} human, {file_stat['ai']} AI)")
    
    # 获取平衡数据集
    human_samples, ai_samples = loader.get_balanced_dataset(max_samples_per_class=50)
    logger.info(f"\n⚖️  Balanced Dataset: {len(human_samples)} human, {len(ai_samples)} AI")
    
    # 获取训练数据
    train_samples, test_samples = loader.get_training_data(test_split=0.2)
    logger.info(f"\n🚂 Training Data: {len(train_samples)} train, {len(test_samples)} test")
    
    # 导出训练数据
    export_data = loader.export_training_data()
    logger.info(f"\n💾 Exported training data with {export_data['statistics']['total_training_samples']} training samples")
    
    # 显示一些样本
    logger.info("\n📝 Sample Data:")
    for i, sample in enumerate(train_samples[:3]):
        label = "AI" if sample['is_ai'] else "Human"
        logger.info(f"  Sample {i+1} ({label}, conf: {sample['confidence']:.2f}): {sample['line'][:60]}...")
    
    logger.info("\n✅ Dataset loader demo completed!")


if __name__ == "__main__":
    main() 