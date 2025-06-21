#!/usr/bin/env python3
"""
增强版AI检测器训练脚本
训练基于完整架构的AI代码检测模型
"""

import os
import sys
import json
import argparse
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_ai_detector import (
    EnhancedAIDetector, FileParser, LineFeatureExtractor, 
    FileFeatureExtractor, EnhancedAIDetectionSystem
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AICodeDataset(Dataset):
    """AI代码检测数据集"""
    
    def __init__(self, data_path: str, test_split: float = 0.2):
        """
        初始化数据集
        
        Args:
            data_path: 数据集路径
            test_split: 测试集比例
        """
        self.data_path = data_path
        self.test_split = test_split
        
        # 初始化特征提取器
        self.line_extractor = LineFeatureExtractor()
        self.file_extractor = FileFeatureExtractor()
        
        # 加载数据
        self.samples = self._load_data()
        
        # 划分训练/测试集
        self.train_samples, self.test_samples = self._split_data()
        
        logger.info(f"Dataset loaded: {len(self.train_samples)} train, {len(self.test_samples)} test")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """加载训练数据"""
        samples = []
        
        if os.path.isfile(self.data_path):
            # 从JSON文件加载
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 处理数据格式
            if 'samples' in data:
                samples = data['samples']
            elif isinstance(data, list):
                samples = data
            else:
                raise ValueError("Invalid data format")
        
        elif os.path.isdir(self.data_path):
            # 从目录加载代码文件
            samples = self._load_from_directory()
        
        else:
            raise ValueError(f"Data path not found: {self.data_path}")
        
        return samples
    
    def _load_from_directory(self) -> List[Dict[str, Any]]:
        """从目录加载代码文件"""
        samples = []
        
        # 假设目录结构：human/ 和 ai/ 子目录
        human_dir = os.path.join(self.data_path, 'human')
        ai_dir = os.path.join(self.data_path, 'ai')
        
        file_parser = FileParser()
        
        # 加载人类代码
        if os.path.exists(human_dir):
            for filename in os.listdir(human_dir):
                filepath = os.path.join(human_dir, filename)
                if file_parser.is_code_file(filename):
                    file_info = file_parser.parse_file(filepath)
                    if file_info['success']:
                        for line_info in file_info['lines']:
                            if not line_info['is_empty']:
                                samples.append({
                                    'content': line_info['content'],
                                    'line_number': line_info['line_number'],
                                    'file_info': file_info,
                                    'is_ai': False
                                })
        
        # 加载AI代码
        if os.path.exists(ai_dir):
            for filename in os.listdir(ai_dir):
                filepath = os.path.join(ai_dir, filename)
                if file_parser.is_code_file(filename):
                    file_info = file_parser.parse_file(filepath)
                    if file_info['success']:
                        for line_info in file_info['lines']:
                            if not line_info['is_empty']:
                                samples.append({
                                    'content': line_info['content'],
                                    'line_number': line_info['line_number'],
                                    'file_info': file_info,
                                    'is_ai': True
                                })
        
        return samples
    
    def _split_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """划分训练和测试集"""
        random.shuffle(self.samples)
        
        split_idx = int(len(self.samples) * (1 - self.test_split))
        train_samples = self.samples[:split_idx]
        test_samples = self.samples[split_idx:]
        
        return train_samples, test_samples
    
    def get_train_loader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """获取训练数据加载器"""
        train_dataset = AICodeSubset(self.train_samples, self.line_extractor, self.file_extractor)
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    
    def get_test_loader(self, batch_size: int = 32, shuffle: bool = False) -> DataLoader:
        """获取测试数据加载器"""
        test_dataset = AICodeSubset(self.test_samples, self.line_extractor, self.file_extractor)
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


class AICodeSubset(Dataset):
    """AI代码数据子集"""
    
    def __init__(self, samples: List[Dict[str, Any]], line_extractor: LineFeatureExtractor, file_extractor: FileFeatureExtractor):
        self.samples = samples
        self.line_extractor = line_extractor
        self.file_extractor = file_extractor
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 构造行信息
        line_info = {
            'content': sample['content'],
            'line_number': sample['line_number'],
            'is_empty': False,
            'indent_level': len(sample['content']) - len(sample['content'].lstrip())
        }
        
        # 提取特征
        line_features = self.line_extractor.extract_features(line_info, sample['file_info'])
        file_features = self.file_extractor.extract_features(sample['file_info'])
        
        return {
            'line_features': torch.tensor(line_features, dtype=torch.float32),
            'file_features': torch.tensor(file_features, dtype=torch.float32),
            'content': sample['content'],
            'line_position': sample['line_number'] - 1,  # 0-indexed
            'label': float(sample['is_ai'])
        }


def collate_fn(batch):
    """数据批处理函数"""
    line_features = torch.stack([item['line_features'] for item in batch])
    file_features = torch.stack([item['file_features'] for item in batch])
    contents = [item['content'] for item in batch]
    line_positions = torch.tensor([item['line_position'] for item in batch], dtype=torch.long)
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    
    return {
        'line_features': line_features,
        'file_features': file_features,
        'contents': contents,
        'line_positions': line_positions.unsqueeze(0),  # Add batch dimension
        'labels': labels
    }


class EnhancedTrainer:
    """增强版训练器"""
    
    def __init__(self, model: EnhancedAIDetector, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # 优化器 - 对不同部分使用不同学习率
        self.optimizer = self._setup_optimizer()
        
        # 损失函数
        self.criterion = nn.BCELoss()
        
        # 训练历史
        self.train_history = {'loss': [], 'accuracy': []}
        self.val_history = {'loss': [], 'accuracy': []}
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """设置优化器 - 对CodeBERT使用较小学习率"""
        # 分离参数
        codebert_params = list(self.model.codebert.parameters())
        other_params = [p for p in self.model.parameters() if p not in codebert_params]
        
        # 使用不同学习率
        optimizer = optim.AdamW([
            {'params': codebert_params, 'lr': 2e-5},      # CodeBERT较小学习率
            {'params': other_params, 'lr': 2e-4}          # 其他层较大学习率
        ], weight_decay=0.01)
        
        return optimizer
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        predictions = []
        targets = []
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            # 移动到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 前向传播
            outputs = self.model(
                batch['line_features'],
                batch['file_features'], 
                batch['contents'],
                batch['line_positions']
            )
            
            # 计算损失
            loss = self.criterion(outputs, batch['labels'])
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            predictions.extend((outputs > 0.5).cpu().numpy())
            targets.extend(batch['labels'].cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(targets, predictions)
        
        return float(avg_loss), float(accuracy)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, Dict[str, float]]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # 移动到设备
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(
                    batch['line_features'],
                    batch['file_features'],
                    batch['contents'],
                    batch['line_positions']
                )
                
                # 计算损失
                loss = self.criterion(outputs, batch['labels'])
                
                # 统计
                total_loss += loss.item()
                predictions.extend((outputs > 0.5).cpu().numpy())
                targets.extend(batch['labels'].cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(targets, predictions)
        
        # 详细指标
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision_score(targets, predictions, zero_division='warn')),
            'recall': float(recall_score(targets, predictions, zero_division='warn')),
            'f1': float(f1_score(targets, predictions, zero_division='warn'))
        }
        
        return avg_loss, float(accuracy), metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 10, save_path: Optional[str] = None) -> Dict[str, Any]:
        """完整训练流程"""
        logger.info(f"Starting training for {epochs} epochs")
        
        best_val_accuracy = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            
            # 验证
            val_loss, val_acc, val_metrics = self.validate(val_loader)
            self.val_history['loss'].append(val_loss)
            self.val_history['accuracy'].append(val_acc)
            
            # 日志
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            logger.info(f"Val Metrics: {val_metrics}")
            
            # 保存最佳模型
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_model_state = self.model.state_dict().copy()
                
                if save_path:
                    self.save_model(save_path, epoch, val_metrics)
        
        # 恢复最佳模型
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        training_result = {
            'best_val_accuracy': best_val_accuracy,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'final_metrics': val_metrics
        }
        
        return training_result
    
    def save_model(self, save_path: str, epoch: int, metrics: Dict[str, float]):
        """保存模型"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"✅ Model saved to {save_path}")


def create_sample_data():
    """创建示例训练数据"""
    sample_data = {
        'samples': [
            # 人类代码样本
            {
                'content': 'def add(a, b):',
                'line_number': 1,
                'file_info': {'total_lines': 3, 'lines': []},
                'is_ai': False
            },
            {
                'content': '    return a + b',
                'line_number': 2,
                'file_info': {'total_lines': 3, 'lines': []},
                'is_ai': False
            },
            # AI代码样本
            {
                'content': 'def calculate_sum(first_number: int, second_number: int) -> int:',
                'line_number': 1,
                'file_info': {'total_lines': 5, 'lines': []},
                'is_ai': True
            },
            {
                'content': '    """Calculate the sum of two integers with comprehensive error handling."""',
                'line_number': 2,
                'file_info': {'total_lines': 5, 'lines': []},
                'is_ai': True
            },
            {
                'content': '    if not isinstance(first_number, int) or not isinstance(second_number, int):',
                'line_number': 3,
                'file_info': {'total_lines': 5, 'lines': []},
                'is_ai': True
            },
            {
                'content': '        raise TypeError("Both arguments must be integers")',
                'line_number': 4,
                'file_info': {'total_lines': 5, 'lines': []},
                'is_ai': True
            },
            {
                'content': '    return first_number + second_number',
                'line_number': 5,
                'file_info': {'total_lines': 5, 'lines': []},
                'is_ai': True
            }
        ]
    }
    
    return sample_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Train Enhanced AI Code Detector")
    parser.add_argument("--data", type=str, required=True, help="Training data path")
    parser.add_argument("--output", type=str, default="enhanced_ai_detector.pth", help="Output model path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--create-sample", action="store_true", help="Create sample data")
    
    args = parser.parse_args()
    
    # 创建示例数据
    if args.create_sample:
        sample_data = create_sample_data()
        sample_path = "sample_training_data.json"
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Sample data created: {sample_path}")
        return
    
    # 设备选择
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"🚀 Enhanced AI Code Detector Training")
    print(f"📁 Data: {args.data}")
    print(f"💾 Output: {args.output}")
    print(f"🔧 Device: {device}")
    print(f"📊 Epochs: {args.epochs}, Batch Size: {args.batch_size}")
    
    # 加载数据集
    try:
        dataset = AICodeDataset(args.data, args.test_split)
        train_loader = dataset.get_train_loader(args.batch_size)
        test_loader = dataset.get_test_loader(args.batch_size)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # 初始化模型
    model = EnhancedAIDetector()
    trainer = EnhancedTrainer(model, device)
    
    # 训练模型
    try:
        results = trainer.train(train_loader, test_loader, args.epochs, args.output)
        
        print(f"\n🎯 Training completed!")
        print(f"   Best validation accuracy: {results['best_val_accuracy']:.4f}")
        print(f"   Final metrics: {results['final_metrics']}")
        print(f"💾 Model saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return


if __name__ == "__main__":
    main() 