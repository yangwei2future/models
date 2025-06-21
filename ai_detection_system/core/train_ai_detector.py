#!/usr/bin/env python3
"""
AI代码检测模型训练脚本 - 基于CodeBERT
生成训练数据并训练CodeBERT + 特征融合的AI检测模型
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple
import logging
from ai_code_detector import CodeBERTAIDetector

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIDetectionDataset(Dataset):
    """AI检测数据集"""
    
    def __init__(self, samples: List[Dict]):
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'line': sample['line'],
            'line_number': sample['line_number'],
            'total_lines': sample['total_lines'],
            'label': float(sample['is_ai'])
        }

def generate_training_data() -> List[Dict]:
    """生成训练数据 - AI vs Human代码样本"""
    
    # Human代码样本 (标签: 0) - 更真实的人类编程风格
    human_samples = [
        # 基础Python代码
        "def calc_sum(a, b):",
        "    return a + b",
        "x = 10",
        "y = 20",
        "result = calc_sum(x, y)",
        "print(result)",
        
        # 实际开发中的代码
        "import os",
        "import sys",
        "from datetime import datetime",
        "class FileManager:",
        "    def __init__(self, path):",
        "        self.path = path",
        "        self.files = []",
        "    def load_files(self):",
        "        for f in os.listdir(self.path):",
        "            self.files.append(f)",
        "if __name__ == '__main__':",
        "    fm = FileManager('/tmp')",
        "    fm.load_files()",
        
        # 简单逻辑和常见模式
        "for i in range(10):",
        "    if i % 2 == 0:",
        "        print(i)",
        "data = [1, 2, 3, 4, 5]",
        "filtered = [x for x in data if x > 2]",
        "total = sum(filtered)",
        "print(f'Total: {total}')",
        
        # 错误处理
        "try:",
        "    file = open('test.txt', 'r')",
        "    content = file.read()",
        "except FileNotFoundError:",
        "    print('File not found')",
        "finally:",
        "    file.close()",
        
        # 常见变量名和简单函数
        "count = 0",
        "index = 0",
        "temp = None",
        "flag = True",
        "items = []",
        "config = {}",
        "def get_data():",
        "    return data",
        "def process():",
        "    pass"
    ]
    
    # AI生成代码样本 (标签: 1) - AI典型的详细、规范风格
    ai_samples = [
        # 详细注释和文档字符串风格
        "# This function calculates the factorial of a given number using recursion",
        "def calculate_factorial(n: int) -> int:",
        '    """Calculate the factorial of a given number using recursion."""',
        "    # Base case: factorial of 0 or 1 is 1",
        "    if n <= 1:",
        "        return 1",
        "    # Recursive case: n! = n * (n-1)!",
        "    return n * calculate_factorial(n - 1)",
        
        # 完整的类实现，规范的类型注解
        "class DataProcessor:",
        '    """A comprehensive class for processing and analyzing data."""',
        "    def __init__(self, data: List[Any]):",
        '        """Initialize the DataProcessor with input data."""',
        "        self.data = data",
        "        self.processed_data = []",
        "        self.logger = logging.getLogger(__name__)",
        "    def process_data(self) -> List[Any]:",
        '        """Process the input data and return processed results."""',
        "        # Implement comprehensive data processing logic here",
        "        for item in self.data:",
        "            processed_item = self._process_item(item)",
        "            self.processed_data.append(processed_item)",
        "        return self.processed_data",
        
        # 算法实现，详细的步骤说明
        "def binary_search(arr: List[int], target: int) -> int:",
        '    """Implement binary search algorithm to find target in sorted array."""',
        "    left, right = 0, len(arr) - 1",
        "    while left <= right:",
        "        # Calculate middle index to avoid overflow",
        "        mid = left + (right - left) // 2",
        "        if arr[mid] == target:",
        "            return mid",
        "        elif arr[mid] < target:",
        "            left = mid + 1",
        "        else:",
        "            right = mid - 1",
        "    return -1",
        
        # 规范的异常处理和类型检查
        "def safe_divide(numerator: float, denominator: float) -> Optional[float]:",
        '    """Safely divide two numbers, handling division by zero gracefully."""',
        "    try:",
        "        if denominator == 0:",
        "            raise ValueError('Division by zero is not allowed')",
        "        result = numerator / denominator",
        "        return result",
        "    except ValueError as e:",
        "        logger.error(f'Error in division operation: {e}')",
        "        return None",
        
        # 生成器函数和高级特性
        "def fibonacci_generator(n: int) -> Generator[int, None, None]:",
        '    """Generate Fibonacci sequence up to n terms using generator."""',
        "    a, b = 0, 1",
        "    for _ in range(n):",
        "        yield a",
        "        a, b = b, a + b",
        
        # 装饰器和高级模式
        "def performance_monitor(func):",
        '    """Decorator to monitor function performance and execution time."""',
        "    def wrapper(*args, **kwargs):",
        "        start_time = time.time()",
        "        result = func(*args, **kwargs)",
        "        end_time = time.time()",
        "        logger.info(f'Function {func.__name__} executed in {end_time - start_time:.4f} seconds')",
        "        return result",
        "    return wrapper"
    ]
    
    # 构建训练样本
    training_samples = []
    
    # Human样本
    for i, line in enumerate(human_samples):
        training_samples.append({
            'line': line,
            'line_number': i + 1,
            'total_lines': len(human_samples),
            'is_ai': False
        })
    
    # AI样本
    for i, line in enumerate(ai_samples):
        training_samples.append({
            'line': line,
            'line_number': i + 1,
            'total_lines': len(ai_samples),
            'is_ai': True
        })
    
    return training_samples

def train_model(samples: List[Dict], epochs: int = 30, batch_size: int = 8, learning_rate: float = 2e-5) -> CodeBERTAIDetector:
    """训练CodeBERT AI检测模型"""
    
    # 创建数据集和数据加载器
    dataset = AIDetectionDataset(samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = CodeBERTAIDetector()
    
    # 设置优化器 - 对CodeBERT使用较小的学习率
    optimizer = optim.AdamW([
        {'params': model.codebert.parameters(), 'lr': learning_rate},  # CodeBERT较小学习率
        {'params': model.feature_projection.parameters(), 'lr': learning_rate * 10},  # 新层较大学习率
        {'params': model.code_projection.parameters(), 'lr': learning_rate * 10},
        {'params': model.classifier.parameters(), 'lr': learning_rate * 10}
    ])
    
    criterion = nn.BCELoss()
    
    logger.info(f"Starting training with {len(samples)} samples for {epochs} epochs")
    logger.info(f"Model: CodeBERT + Feature Fusion + Classifier")
    logger.info(f"Learning rate: {learning_rate} (CodeBERT), {learning_rate * 10} (new layers)")
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            batch_losses = []
            batch_correct = 0
            
            for i in range(len(batch['line'])):
                line = batch['line'][i]
                line_number = batch['line_number'][i].item()
                total_lines = batch['total_lines'][i].item()
                label = batch['label'][i].item()
                
                # 前向传播
                ai_prob = model(line, line_number, total_lines)
                
                # 计算损失
                loss = criterion(torch.tensor([ai_prob]), torch.tensor([label]))
                batch_losses.append(loss)
                
                # 计算准确率
                prediction = 1 if ai_prob > 0.5 else 0
                if prediction == int(label):
                    batch_correct += 1
                
                total_predictions += 1
            
            # 合并批次损失
            if batch_losses:
                batch_loss = torch.stack(batch_losses).mean()
                
                # 反向传播
                batch_loss.backward()
                optimizer.step()
                
                total_loss += batch_loss.item()
                correct_predictions += batch_correct
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return model

def evaluate_model(model: CodeBERTAIDetector, samples: List[Dict]) -> Dict:
    """评估模型性能"""
    model.eval()
    
    correct = 0
    total = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    with torch.no_grad():
        for sample in samples:
            line = sample['line']
            line_number = sample['line_number']
            total_lines = sample['total_lines']
            true_label = sample['is_ai']
            
            ai_prob = model(line, line_number, total_lines)
            predicted_label = ai_prob > 0.5
            
            total += 1
            if predicted_label == true_label:
                correct += 1
            
            # 混淆矩阵
            if true_label and predicted_label:
                true_positives += 1
            elif not true_label and predicted_label:
                false_positives += 1
            elif not true_label and not predicted_label:
                true_negatives += 1
            else:
                false_negatives += 1
    
    accuracy = correct / total
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }
    }

def main():
    logger.info("🤖 CodeBERT-based AI Code Detection Model Training")
    logger.info("=" * 60)
    
    # 1. 生成训练数据
    logger.info("📊 Generating training data...")
    training_samples = generate_training_data()
    
    human_count = sum(1 for s in training_samples if not s['is_ai'])
    ai_count = sum(1 for s in training_samples if s['is_ai'])
    
    logger.info(f"Generated {len(training_samples)} training samples:")
    logger.info(f"  - Human code: {human_count} samples")
    logger.info(f"  - AI code: {ai_count} samples")
    
    # 2. 训练模型
    logger.info("\n🚀 Training CodeBERT-based model...")
    logger.info("Note: First run will download CodeBERT model (~500MB)")
    
    try:
        model = train_model(training_samples, epochs=20, batch_size=4, learning_rate=2e-5)
        logger.info("✅ Model training completed successfully")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.info("This might be due to missing transformers library or network issues")
        logger.info("Please install: pip install transformers")
        return
    
    # 3. 评估模型
    logger.info("\n📈 Evaluating model...")
    metrics = evaluate_model(model, training_samples)
    
    logger.info("CodeBERT Model Performance:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
    
    # 4. 保存模型
    os.makedirs('models', exist_ok=True)
    model_path = 'models/ai_detector.pt'
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': 'CodeBERT-based',
        'codebert_model': model.model_name,
        'feature_dim': model.feature_dim,
        'hidden_dim': model.hidden_dim,
        'metrics': metrics,
        'training_samples': len(training_samples)
    }, model_path)
    
    logger.info(f"\n💾 CodeBERT model saved to {model_path}")
    
    # 5. 保存训练数据用于参考
    data_path = 'models/training_data.json'
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(training_samples, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 Training data saved to {data_path}")
    
    # 6. 测试一些样本
    logger.info("\n🔍 Testing sample predictions:")
    test_samples = [
        "def hello_world():",
        "# This function implements a sophisticated algorithm for data processing",
        "x = 5",
        '"""Calculate the optimal solution using dynamic programming approach."""',
        "for i in range(10):",
        "def process_data(input_data: List[Any]) -> Dict[str, Any]:",
        "    return result",
        "logger.info(f'Processing completed successfully with {count} items')"
    ]
    
    model.eval()
    with torch.no_grad():
        for i, line in enumerate(test_samples, 1):
            ai_prob = model(line, i, len(test_samples))
            prediction = "AI" if ai_prob > 0.5 else "Human"
            logger.info(f"  '{line[:50]}...' -> {prediction} ({ai_prob:.3f})")
    
    logger.info(f"\n✅ CodeBERT training completed! Model ready for use.")
    logger.info(f"Usage: python line_ai_detector.py --model {model_path} --input your_file.py")
    logger.info(f"Note: The model uses CodeBERT + feature fusion for improved accuracy")

if __name__ == "__main__":
    main() 