#!/usr/bin/env python3
"""
结果聚合模块
架构流程第9步：聚合检测结果并生成统计信息
"""

import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Union


class ResultAggregator:
    """结果聚合器 - 聚合检测结果并生成统计信息"""
    
    def __init__(self):
        self.aggregation_history = []
    
    def aggregate_file_results(self, 
                             file_info: Dict[str, Any],
                             ai_probabilities: List[float],
                             ai_predictions: List[bool]) -> Dict[str, Any]:
        """
        聚合单个文件的结果
        
        Args:
            file_info: 文件信息字典
            ai_probabilities: AI概率列表
            ai_predictions: AI预测结果列表
            
        Returns:
            Dict: 聚合后的文件结果
        """
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
        """
        聚合批量文件结果
        
        Args:
            file_results: 文件结果列表
            
        Returns:
            Dict: 聚合后的批量结果
        """
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
        
        # 记录聚合历史
        self.aggregation_history.append({
            'timestamp': datetime.now().isoformat(),
            'total_files': total_files,
            'successful_files': successful_files,
            'total_ai_lines': total_ai_lines,
            'total_code_lines': total_code_lines
        })
        
        return batch_result
    
    def generate_detailed_report(self, batch_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成详细分析报告
        
        Args:
            batch_result: 批量结果
            
        Returns:
            Dict: 详细报告
        """
        successful_results = [r for r in batch_result['results'] if r.get('success', False)]
        
        if not successful_results:
            return {"error": "No successful results to analyze"}
        
        # 文件级分析
        file_analysis = []
        ai_prob_distribution = []
        
        for result in successful_results:
            file_stats = result['summary']
            file_analysis.append({
                'file_path': result['file_path'],
                'ai_percentage': file_stats['ai_percentage'],
                'average_ai_prob': file_stats['average_ai_prob'],
                'code_lines': file_stats['code_lines'],
                'ai_lines': file_stats['ai_lines']
            })
            
            # 收集概率分布数据
            for line in result['lines']:
                ai_prob_distribution.append(line['ai_prob'])
        
        # 统计分析
        ai_percentages = [f['ai_percentage'] for f in file_analysis]
        avg_probs = [f['average_ai_prob'] for f in file_analysis]
        
        detailed_report = {
            "file_analysis": file_analysis,
            "distribution_analysis": {
                "ai_probability_stats": {
                    "mean": round(np.mean(ai_prob_distribution), 3),
                    "median": round(np.median(ai_prob_distribution), 3),
                    "std": round(np.std(ai_prob_distribution), 3),
                    "min": round(np.min(ai_prob_distribution), 3),
                    "max": round(np.max(ai_prob_distribution), 3),
                    "quartiles": [
                        round(np.percentile(ai_prob_distribution, 25), 3),
                        round(np.percentile(ai_prob_distribution, 50), 3),
                        round(np.percentile(ai_prob_distribution, 75), 3)
                    ]
                },
                "file_ai_percentage_stats": {
                    "mean": round(np.mean(ai_percentages), 1),
                    "median": round(np.median(ai_percentages), 1),
                    "std": round(np.std(ai_percentages), 1),
                    "min": round(np.min(ai_percentages), 1),
                    "max": round(np.max(ai_percentages), 1)
                }
            },
            "pattern_analysis": self._analyze_patterns(successful_results),
            "recommendations": self._generate_recommendations(successful_results)
        }
        
        return detailed_report
    
    def _analyze_patterns(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析AI代码模式
        
        Args:
            results: 检测结果列表
            
        Returns:
            Dict: 模式分析结果
        """
        ai_lines = []
        human_lines = []
        
        for result in results:
            for line in result['lines']:
                if line['is_ai']:
                    ai_lines.append(line['content'])
                else:
                    human_lines.append(line['content'])
        
        # 分析AI代码特征
        ai_patterns = {
            'avg_length': np.mean([len(line) for line in ai_lines]) if ai_lines else 0,
            'common_keywords': self._extract_common_keywords(ai_lines),
            'avg_complexity': self._calculate_avg_complexity(ai_lines)
        }
        
        # 分析人类代码特征
        human_patterns = {
            'avg_length': np.mean([len(line) for line in human_lines]) if human_lines else 0,
            'common_keywords': self._extract_common_keywords(human_lines),
            'avg_complexity': self._calculate_avg_complexity(human_lines)
        }
        
        return {
            'ai_patterns': ai_patterns,
            'human_patterns': human_patterns,
            'total_ai_lines': len(ai_lines),
            'total_human_lines': len(human_lines)
        }
    
    def _extract_common_keywords(self, lines: List[str], top_k: int = 10) -> List[Dict[str, Union[str, int]]]:
        """
        提取常见关键词
        
        Args:
            lines: 代码行列表
            top_k: 返回前k个关键词
            
        Returns:
            List: 关键词统计
        """
        if not lines:
            return []
        
        # 简单的关键词提取
        keywords = {}
        for line in lines:
            words = line.split()
            for word in words:
                if len(word) > 2 and word.isalpha():  # 过滤短词和非字母词
                    keywords[word] = keywords.get(word, 0) + 1
        
        # 排序并返回前k个
        sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        return [{'keyword': word, 'count': count} for word, count in sorted_keywords[:top_k]]
    
    def _calculate_avg_complexity(self, lines: List[str]) -> float:
        """
        计算平均复杂度
        
        Args:
            lines: 代码行列表
            
        Returns:
            float: 平均复杂度
        """
        if not lines:
            return 0.0
        
        total_complexity = 0
        for line in lines:
            # 简单的复杂度指标：括号、操作符、关键词数量
            complexity = (line.count('(') + line.count('[') + line.count('{') + 
                         line.count('=') + line.count('+') + line.count('-') +
                         len([w for w in line.split() if w in ['if', 'for', 'while', 'try', 'def', 'class']]))
            total_complexity += complexity
        
        return round(total_complexity / len(lines), 2)
    
    def _generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """
        生成优化建议
        
        Args:
            results: 检测结果列表
            
        Returns:
            List: 建议列表
        """
        recommendations = []
        
        # 计算总体AI比例
        total_code_lines = sum(r['summary']['code_lines'] for r in results)
        total_ai_lines = sum(r['summary']['ai_lines'] for r in results)
        overall_ai_ratio = total_ai_lines / total_code_lines if total_code_lines > 0 else 0
        
        if overall_ai_ratio > 0.7:
            recommendations.append("检测到大量AI生成代码，建议进行人工审查以确保代码质量")
        elif overall_ai_ratio > 0.3:
            recommendations.append("检测到中等比例的AI生成代码，建议重点关注关键功能模块")
        else:
            recommendations.append("AI生成代码比例较低，代码质量相对可控")
        
        # 基于概率分布的建议
        all_probs = []
        for result in results:
            all_probs.extend([line['ai_prob'] for line in result['lines']])
        
        if all_probs:
            high_uncertainty_ratio = sum(1 for p in all_probs if 0.4 <= p <= 0.6) / len(all_probs)
            if high_uncertainty_ratio > 0.2:
                recommendations.append("检测到较多不确定性高的代码行，建议调整检测阈值或进行人工验证")
        
        return recommendations
    
    def get_aggregation_history(self) -> List[Dict[str, Any]]:
        """获取聚合历史记录"""
        return self.aggregation_history.copy()


class AdvancedResultAggregator(ResultAggregator):
    """高级结果聚合器 - 提供更多分析功能"""
    
    def __init__(self):
        super().__init__()
        self.comparison_cache = {}
    
    def compare_detection_results(self, 
                                results_a: Dict[str, Any], 
                                results_b: Dict[str, Any],
                                label_a: str = "Detection A",
                                label_b: str = "Detection B") -> Dict[str, Any]:
        """
        比较两次检测结果
        
        Args:
            results_a: 第一次检测结果
            results_b: 第二次检测结果
            label_a: 第一次检测标签
            label_b: 第二次检测标签
            
        Returns:
            Dict: 比较结果
        """
        stats_a = results_a['statistics']
        stats_b = results_b['statistics']
        
        comparison = {
            'labels': [label_a, label_b],
            'statistics_comparison': {
                'total_files': [stats_a['total_files'], stats_b['total_files']],
                'total_ai_lines': [stats_a['total_ai_lines'], stats_b['total_ai_lines']],
                'overall_ai_percentage': [stats_a['overall_ai_percentage'], stats_b['overall_ai_percentage']]
            },
            'difference_analysis': {
                'ai_lines_diff': stats_b['total_ai_lines'] - stats_a['total_ai_lines'],
                'ai_percentage_diff': stats_b['overall_ai_percentage'] - stats_a['overall_ai_percentage']
            }
        }
        
        # 缓存比较结果
        cache_key = f"{label_a}_vs_{label_b}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.comparison_cache[cache_key] = comparison
        
        return comparison
    
    def generate_trend_analysis(self, historical_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        生成趋势分析
        
        Args:
            historical_results: 历史检测结果列表
            
        Returns:
            Dict: 趋势分析结果
        """
        if len(historical_results) < 2:
            return {"error": "Need at least 2 historical results for trend analysis"}
        
        # 提取时间序列数据
        timestamps = []
        ai_percentages = []
        total_lines = []
        
        for result in historical_results:
            if 'metadata' in result and 'timestamp' in result['metadata']:
                timestamps.append(result['metadata']['timestamp'])
                ai_percentages.append(result['statistics']['overall_ai_percentage'])
                total_lines.append(result['statistics']['total_code_lines'])
        
        # 计算趋势
        if len(ai_percentages) >= 2:
            trend_slope = (ai_percentages[-1] - ai_percentages[0]) / (len(ai_percentages) - 1)
            
            trend_analysis = {
                'time_series': {
                    'timestamps': timestamps,
                    'ai_percentages': ai_percentages,
                    'total_lines': total_lines
                },
                'trend_metrics': {
                    'slope': round(trend_slope, 3),
                    'direction': 'increasing' if trend_slope > 0.1 else 'decreasing' if trend_slope < -0.1 else 'stable',
                    'volatility': round(np.std(ai_percentages), 2),
                    'range': [min(ai_percentages), max(ai_percentages)]
                },
                'insights': self._generate_trend_insights(ai_percentages, trend_slope)
            }
            
            return trend_analysis
        
        return {"error": "Insufficient data for trend analysis"}
    
    def _generate_trend_insights(self, ai_percentages: List[float], slope: float) -> List[str]:
        """
        生成趋势洞察
        
        Args:
            ai_percentages: AI百分比序列
            slope: 趋势斜率
            
        Returns:
            List: 洞察列表
        """
        insights = []
        
        if slope > 0.5:
            insights.append("AI代码比例呈明显上升趋势，需要关注代码质量控制")
        elif slope < -0.5:
            insights.append("AI代码比例呈下降趋势，可能表明开发模式的改变")
        else:
            insights.append("AI代码比例相对稳定")
        
        volatility = np.std(ai_percentages)
        if volatility > 10:
            insights.append("AI代码比例波动较大，建议建立更稳定的开发流程")
        
        return insights 