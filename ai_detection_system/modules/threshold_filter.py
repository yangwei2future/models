#!/usr/bin/env python3
"""
阈值过滤模块
架构流程第8步：根据阈值将概率转换为二分类结果
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class ThresholdFilter:
    """阈值过滤器 - 将概率转换为二分类结果"""
    
    def __init__(self, threshold: float = 0.5):
        """
        初始化阈值过滤器
        
        Args:
            threshold: 分类阈值，默认0.5
        """
        self.threshold = threshold
        self.history = []  # 记录历史决策
    
    def filter(self, probabilities: Union[torch.Tensor, np.ndarray, List[float]]) -> Union[torch.Tensor, np.ndarray]:
        """
        应用阈值过滤
        
        Args:
            probabilities: AI概率数组
            
        Returns:
            Union[torch.Tensor, np.ndarray]: 二分类结果
        """
        if isinstance(probabilities, torch.Tensor):
            predictions = (probabilities > self.threshold).float()
            # 转换为numpy进行统计
            predictions_np = predictions.cpu().numpy()
        elif isinstance(probabilities, np.ndarray):
            predictions = (probabilities > self.threshold).astype(float)
            predictions_np = predictions
        else:
            predictions = np.array([1.0 if p > self.threshold else 0.0 for p in probabilities])
            predictions_np = predictions
        
        # 记录历史
        self.history.append({
            'threshold': self.threshold,
            'num_positive': int(predictions_np.sum()),
            'num_total': len(predictions_np),
            'positive_rate': float(predictions_np.mean())
        })
        
        return predictions
    
    def set_threshold(self, threshold: float):
        """
        设置新的阈值
        
        Args:
            threshold: 新阈值
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.threshold = threshold
    
    def get_threshold(self) -> float:
        """获取当前阈值"""
        return self.threshold
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取过滤统计信息
        
        Returns:
            Dict: 统计信息
        """
        if not self.history:
            return {}
        
        recent_stats = self.history[-1]
        all_positive_rates = [h['positive_rate'] for h in self.history]
        
        return {
            'current_threshold': self.threshold,
            'recent_positive_rate': recent_stats['positive_rate'],
            'recent_positive_count': recent_stats['num_positive'],
            'recent_total_count': recent_stats['num_total'],
            'avg_positive_rate': float(np.mean(all_positive_rates)),
            'std_positive_rate': float(np.std(all_positive_rates)),
            'total_decisions': len(self.history)
        }


class AdaptiveThresholdFilter:
    """自适应阈值过滤器 - 根据数据分布动态调整阈值"""
    
    def __init__(self, 
                 initial_threshold: float = 0.5,
                 adaptation_rate: float = 0.1,
                 target_positive_rate: Optional[float] = None):
        """
        初始化自适应阈值过滤器
        
        Args:
            initial_threshold: 初始阈值
            adaptation_rate: 适应速率
            target_positive_rate: 目标正例率
        """
        self.threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.target_positive_rate = target_positive_rate
        self.probability_history = []
        self.threshold_history = [initial_threshold]
    
    def filter(self, probabilities: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        自适应阈值过滤
        
        Args:
            probabilities: AI概率数组
            
        Returns:
            Union[torch.Tensor, np.ndarray]: 二分类结果
        """
        # 转换为numpy数组便于处理
        if isinstance(probabilities, torch.Tensor):
            probs_np = probabilities.cpu().numpy()
            is_tensor = True
        else:
            probs_np = np.array(probabilities)
            is_tensor = False
        
        # 记录概率历史
        self.probability_history.extend(probs_np.tolist())
        
        # 更新阈值
        self._update_threshold(probs_np)
        
        # 应用阈值
        predictions = (probs_np > self.threshold).astype(float)
        
        # 转换回原始类型
        if is_tensor:
            return torch.tensor(predictions, dtype=torch.float32)
        else:
            return predictions
    
    def _update_threshold(self, probabilities: np.ndarray):
        """
        更新阈值
        
        Args:
            probabilities: 当前批次的概率
        """
        if self.target_positive_rate is not None:
            # 基于目标正例率调整
            current_rate = np.mean(probabilities > self.threshold)
            error = current_rate - self.target_positive_rate
            adjustment = -self.adaptation_rate * error
            self.threshold = np.clip(self.threshold + adjustment, 0.0, 1.0)
        else:
            # 基于概率分布调整
            if len(self.probability_history) > 100:  # 有足够历史数据
                recent_probs = np.array(self.probability_history[-100:])
                
                # 使用概率分布的统计信息
                mean_prob = np.mean(recent_probs)
                std_prob = np.std(recent_probs)
                
                # 动态调整阈值
                if std_prob > 0.2:  # 概率分布较分散
                    self.threshold = mean_prob
                else:  # 概率分布较集中
                    self.threshold = mean_prob + 0.1 * std_prob
                
                self.threshold = np.clip(self.threshold, 0.1, 0.9)
        
        # 记录阈值历史
        self.threshold_history.append(float(self.threshold))
    
    def get_threshold_evolution(self) -> List[float]:
        """获取阈值演化历史"""
        return self.threshold_history.copy()


class MultiThresholdFilter:
    """多阈值过滤器 - 使用多个阈值进行分层分类"""
    
    def __init__(self, thresholds: List[float] = [0.3, 0.5, 0.7]):
        """
        初始化多阈值过滤器
        
        Args:
            thresholds: 阈值列表，必须是递增的
        """
        self.thresholds = sorted(thresholds)
        if len(self.thresholds) < 2:
            raise ValueError("At least 2 thresholds are required")
    
    def filter(self, probabilities: Union[torch.Tensor, np.ndarray]) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """
        多阈值过滤
        
        Args:
            probabilities: AI概率数组
            
        Returns:
            Dict: 包含不同阈值结果的字典
        """
        if isinstance(probabilities, torch.Tensor):
            probs = probabilities.cpu().numpy()
            is_tensor = True
        else:
            probs = np.array(probabilities)
            is_tensor = False
        
        results = {}
        
        # 对每个阈值进行分类
        for threshold in self.thresholds:
            predictions = (probs > threshold).astype(float)
            if is_tensor:
                predictions = torch.tensor(predictions, dtype=torch.float32)
            results[f'threshold_{threshold}'] = predictions
        
        # 分层分类
        confidence_levels = self._classify_by_confidence(probs)
        if is_tensor:
            confidence_levels = torch.tensor(confidence_levels, dtype=torch.long)
        results['confidence_levels'] = confidence_levels
        
        return results
    
    def _classify_by_confidence(self, probabilities: np.ndarray) -> np.ndarray:
        """
        基于置信度分层分类
        
        Args:
            probabilities: 概率数组
            
        Returns:
            np.ndarray: 置信度等级 (0: 低置信度人工, 1: 中等置信度, 2: 高置信度AI)
        """
        levels = np.zeros(len(probabilities), dtype=int)
        
        for i, prob in enumerate(probabilities):
            if prob < self.thresholds[0]:
                levels[i] = 0  # 低置信度，倾向于人工
            elif prob > self.thresholds[-1]:
                levels[i] = 2  # 高置信度，倾向于AI
            else:
                levels[i] = 1  # 中等置信度，不确定
        
        return levels


class ROCBasedThresholdSelector:
    """基于ROC曲线的阈值选择器"""
    
    def __init__(self):
        self.optimal_threshold = 0.5
        self.roc_data = None
    
    def find_optimal_threshold(self, 
                             probabilities: np.ndarray, 
                             true_labels: np.ndarray,
                             criterion: str = 'youden') -> float:
        """
        基于ROC曲线找到最优阈值
        
        Args:
            probabilities: 预测概率
            true_labels: 真实标签
            criterion: 优化准则 ('youden', 'f1', 'precision', 'recall')
            
        Returns:
            float: 最优阈值
        """
        thresholds = np.linspace(0, 1, 101)
        scores = []
        
        for threshold in thresholds:
            predictions = (probabilities > threshold).astype(int)
            
            # 计算混淆矩阵
            tp = np.sum((predictions == 1) & (true_labels == 1))
            fp = np.sum((predictions == 1) & (true_labels == 0))
            tn = np.sum((predictions == 0) & (true_labels == 0))
            fn = np.sum((predictions == 0) & (true_labels == 1))
            
            # 计算指标
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = sensitivity
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # 根据准则计算分数
            if criterion == 'youden':
                score = sensitivity + specificity - 1  # Youden's J statistic
            elif criterion == 'f1':
                score = f1
            elif criterion == 'precision':
                score = precision
            elif criterion == 'recall':
                score = recall
            else:
                score = f1  # 默认使用F1
            
            scores.append(score)
        
        # 找到最优阈值
        optimal_idx = np.argmax(scores)
        self.optimal_threshold = thresholds[optimal_idx]
        
        # 保存ROC数据
        self.roc_data = {
            'thresholds': thresholds,
            'scores': scores,
            'optimal_threshold': self.optimal_threshold,
            'optimal_score': scores[optimal_idx],
            'criterion': criterion
        }
        
        return self.optimal_threshold
    
    def get_roc_data(self) -> Optional[Dict]:
        """获取ROC分析数据"""
        return self.roc_data 