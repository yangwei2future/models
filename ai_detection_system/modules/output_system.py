#!/usr/bin/env python3
"""
输出系统模块
架构流程第10步：格式化输出结果，支持多种输出格式
"""

import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import logging


class OutputSystem:
    """输出系统 - 支持多种格式的结果输出"""
    
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # 支持的输出格式
        self.supported_formats = ['json', 'csv', 'xml', 'txt', 'html']
        
        # 输出历史
        self.output_history = []
    
    def output_json(self, data: Dict[str, Any], filename: str = None) -> str:
        """
        输出JSON格式
        
        Args:
            data: 要输出的数据
            filename: 输出文件名
            
        Returns:
            str: 输出文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_detection_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self._record_output('json', str(filepath), data)
            self.logger.info(f"JSON output saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save JSON output: {e}")
            raise
    
    def output_csv(self, data: Dict[str, Any], filename: str = None) -> str:
        """
        输出CSV格式（逐行结果）
        
        Args:
            data: 要输出的数据
            filename: 输出文件名
            
        Returns:
            str: 输出文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_detection_results_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # 写入表头
                writer.writerow(['file_path', 'line_number', 'content', 'ai_prob', 'is_ai'])
                
                # 写入数据
                if 'results' in data:
                    for file_result in data['results']:
                        if file_result.get('success', False):
                            file_path = file_result['file_path']
                            for line in file_result['lines']:
                                writer.writerow([
                                    file_path,
                                    line['line_number'],
                                    line['content'],
                                    line['ai_prob'],
                                    line['is_ai']
                                ])
            
            self._record_output('csv', str(filepath), data)
            self.logger.info(f"CSV output saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save CSV output: {e}")
            raise
    
    def output_xml(self, data: Dict[str, Any], filename: str = None) -> str:
        """
        输出XML格式
        
        Args:
            data: 要输出的数据
            filename: 输出文件名
            
        Returns:
            str: 输出文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_detection_results_{timestamp}.xml"
        
        filepath = self.output_dir / filename
        
        try:
            # 创建XML根元素
            root = ET.Element("ai_detection_results")
            
            # 添加元数据
            if 'metadata' in data:
                metadata_elem = ET.SubElement(root, "metadata")
                for key, value in data['metadata'].items():
                    elem = ET.SubElement(metadata_elem, key)
                    elem.text = str(value)
            
            # 添加统计信息
            if 'statistics' in data:
                stats_elem = ET.SubElement(root, "statistics")
                for key, value in data['statistics'].items():
                    elem = ET.SubElement(stats_elem, key)
                    elem.text = str(value)
            
            # 添加结果
            if 'results' in data:
                results_elem = ET.SubElement(root, "results")
                for file_result in data['results']:
                    if file_result.get('success', False):
                        file_elem = ET.SubElement(results_elem, "file")
                        file_elem.set("path", file_result['file_path'])
                        
                        # 文件摘要
                        summary_elem = ET.SubElement(file_elem, "summary")
                        for key, value in file_result['summary'].items():
                            elem = ET.SubElement(summary_elem, key)
                            elem.text = str(value)
                        
                        # 行结果
                        lines_elem = ET.SubElement(file_elem, "lines")
                        for line in file_result['lines']:
                            line_elem = ET.SubElement(lines_elem, "line")
                            line_elem.set("number", str(line['line_number']))
                            line_elem.set("ai_prob", str(line['ai_prob']))
                            line_elem.set("is_ai", str(line['is_ai']))
                            line_elem.text = line['content']
            
            # 保存XML文件
            tree = ET.ElementTree(root)
            tree.write(filepath, encoding='utf-8', xml_declaration=True)
            
            self._record_output('xml', str(filepath), data)
            self.logger.info(f"XML output saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save XML output: {e}")
            raise
    
    def output_txt(self, data: Dict[str, Any], filename: str = None) -> str:
        """
        输出文本格式（可读性强的报告）
        
        Args:
            data: 要输出的数据
            filename: 输出文件名
            
        Returns:
            str: 输出文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_detection_report_{timestamp}.txt"
        
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # 标题
                f.write("AI代码检测报告\n")
                f.write("=" * 50 + "\n\n")
                
                # 元数据
                if 'metadata' in data:
                    f.write("检测信息:\n")
                    f.write("-" * 20 + "\n")
                    for key, value in data['metadata'].items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
                
                # 统计信息
                if 'statistics' in data:
                    f.write("统计摘要:\n")
                    f.write("-" * 20 + "\n")
                    stats = data['statistics']
                    f.write(f"总文件数: {stats.get('total_files', 0)}\n")
                    f.write(f"成功处理: {stats.get('successful_files', 0)}\n")
                    f.write(f"总代码行数: {stats.get('total_code_lines', 0)}\n")
                    f.write(f"AI生成行数: {stats.get('total_ai_lines', 0)}\n")
                    f.write(f"AI代码比例: {stats.get('overall_ai_percentage', 0)}%\n")
                    f.write("\n")
                
                # 详细结果
                if 'results' in data:
                    f.write("详细结果:\n")
                    f.write("-" * 20 + "\n")
                    
                    for file_result in data['results']:
                        if file_result.get('success', False):
                            f.write(f"\n文件: {file_result['file_path']}\n")
                            summary = file_result['summary']
                            f.write(f"  代码行数: {summary['code_lines']}\n")
                            f.write(f"  AI行数: {summary['ai_lines']}\n")
                            f.write(f"  AI比例: {summary['ai_percentage']}%\n")
                            f.write(f"  平均AI概率: {summary['average_ai_prob']}\n")
                            
                            # 显示AI概率较高的行
                            high_prob_lines = [line for line in file_result['lines'] 
                                             if line['ai_prob'] > 0.7]
                            if high_prob_lines:
                                f.write("  高AI概率行:\n")
                                for line in high_prob_lines[:5]:  # 只显示前5行
                                    f.write(f"    行{line['line_number']}: {line['ai_prob']:.3f} - {line['content'][:60]}...\n")
            
            self._record_output('txt', str(filepath), data)
            self.logger.info(f"Text report saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save text report: {e}")
            raise
    
    def output_html(self, data: Dict[str, Any], filename: str = None) -> str:
        """
        输出HTML格式（可视化报告）
        
        Args:
            data: 要输出的数据
            filename: 输出文件名
            
        Returns:
            str: 输出文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_detection_report_{timestamp}.html"
        
        filepath = self.output_dir / filename
        
        try:
            html_content = self._generate_html_report(data)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self._record_output('html', str(filepath), data)
            self.logger.info(f"HTML report saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save HTML report: {e}")
            raise
    
    def _generate_html_report(self, data: Dict[str, Any]) -> str:
        """生成HTML报告内容"""
        html = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AI代码检测报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .stats { display: flex; justify-content: space-around; margin: 20px 0; }
                .stat-box { background-color: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }
                .file-result { margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }
                .file-header { background-color: #f8f9fa; padding: 10px; font-weight: bold; }
                .line-result { padding: 5px 10px; border-bottom: 1px solid #eee; }
                .ai-line { background-color: #ffe6e6; }
                .human-line { background-color: #e6ffe6; }
                .prob-high { color: #d32f2f; font-weight: bold; }
                .prob-medium { color: #ff9800; }
                .prob-low { color: #4caf50; }
            </style>
        </head>
        <body>
        """
        
        # 标题和元数据
        html += "<div class='header'><h1>AI代码检测报告</h1>"
        if 'metadata' in data:
            html += f"<p><strong>检测时间:</strong> {data['metadata'].get('timestamp', 'N/A')}</p>"
            html += f"<p><strong>模型类型:</strong> {data['metadata'].get('model_type', 'N/A')}</p>"
        html += "</div>"
        
        # 统计信息
        if 'statistics' in data:
            stats = data['statistics']
            html += "<div class='stats'>"
            html += f"<div class='stat-box'><h3>{stats.get('total_files', 0)}</h3><p>总文件数</p></div>"
            html += f"<div class='stat-box'><h3>{stats.get('total_code_lines', 0)}</h3><p>总代码行数</p></div>"
            html += f"<div class='stat-box'><h3>{stats.get('total_ai_lines', 0)}</h3><p>AI生成行数</p></div>"
            html += f"<div class='stat-box'><h3>{stats.get('overall_ai_percentage', 0)}%</h3><p>AI代码比例</p></div>"
            html += "</div>"
        
        # 详细结果
        if 'results' in data:
            html += "<h2>详细结果</h2>"
            for file_result in data['results']:
                if file_result.get('success', False):
                    html += f"<div class='file-result'>"
                    html += f"<div class='file-header'>{file_result['file_path']}</div>"
                    
                    summary = file_result['summary']
                    html += f"<p>代码行数: {summary['code_lines']} | AI行数: {summary['ai_lines']} | AI比例: {summary['ai_percentage']}%</p>"
                    
                    # 显示部分行结果
                    for line in file_result['lines'][:20]:  # 只显示前20行
                        css_class = 'ai-line' if line['is_ai'] else 'human-line'
                        prob_class = 'prob-high' if line['ai_prob'] > 0.7 else 'prob-medium' if line['ai_prob'] > 0.3 else 'prob-low'
                        
                        html += f"<div class='line-result {css_class}'>"
                        html += f"<span>行{line['line_number']}</span> "
                        html += f"<span class='{prob_class}'>概率: {line['ai_prob']:.3f}</span> "
                        html += f"<code>{self._escape_html(line['content'][:80])}</code>"
                        html += "</div>"
                    
                    if len(file_result['lines']) > 20:
                        html += f"<p><em>... 还有 {len(file_result['lines']) - 20} 行</em></p>"
                    
                    html += "</div>"
        
        html += "</body></html>"
        return html
    
    def _escape_html(self, text: str) -> str:
        """HTML转义"""
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&#x27;'))
    
    def output_multiple_formats(self, 
                               data: Dict[str, Any], 
                               formats: List[str],
                               base_filename: str = None) -> Dict[str, str]:
        """
        同时输出多种格式
        
        Args:
            data: 要输出的数据
            formats: 输出格式列表
            base_filename: 基础文件名
            
        Returns:
            Dict: 格式到文件路径的映射
        """
        results = {}
        
        for format_type in formats:
            if format_type not in self.supported_formats:
                self.logger.warning(f"Unsupported format: {format_type}")
                continue
            
            try:
                if base_filename:
                    filename = f"{base_filename}.{format_type}"
                else:
                    filename = None
                
                if format_type == 'json':
                    filepath = self.output_json(data, filename)
                elif format_type == 'csv':
                    filepath = self.output_csv(data, filename)
                elif format_type == 'xml':
                    filepath = self.output_xml(data, filename)
                elif format_type == 'txt':
                    filepath = self.output_txt(data, filename)
                elif format_type == 'html':
                    filepath = self.output_html(data, filename)
                
                results[format_type] = filepath
                
            except Exception as e:
                self.logger.error(f"Failed to output {format_type} format: {e}")
                results[format_type] = f"Error: {e}"
        
        return results
    
    def _record_output(self, format_type: str, filepath: str, data: Dict[str, Any]):
        """记录输出历史"""
        self.output_history.append({
            'timestamp': datetime.now().isoformat(),
            'format': format_type,
            'filepath': filepath,
            'data_size': len(str(data)),
            'num_files': data.get('statistics', {}).get('total_files', 0),
            'num_lines': data.get('statistics', {}).get('total_code_lines', 0)
        })
    
    def get_output_history(self) -> List[Dict[str, Any]]:
        """获取输出历史"""
        return self.output_history.copy()
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的输出格式"""
        return self.supported_formats.copy()


class StreamingOutputSystem:
    """流式输出系统 - 支持实时输出结果"""
    
    def __init__(self, output_stream=None):
        self.output_stream = output_stream or print
        self.buffer = []
        self.buffer_size = 100
    
    def stream_result(self, file_result: Dict[str, Any]):
        """流式输出单个文件结果"""
        if file_result.get('success', False):
            summary = file_result['summary']
            self.output_stream(f"📁 {file_result['file_path']}")
            self.output_stream(f"   AI比例: {summary['ai_percentage']}% ({summary['ai_lines']}/{summary['code_lines']})")
            
            # 输出高AI概率的行
            high_prob_lines = [line for line in file_result['lines'] if line['ai_prob'] > 0.7]
            if high_prob_lines:
                self.output_stream(f"   🤖 高AI概率行:")
                for line in high_prob_lines[:3]:
                    self.output_stream(f"      行{line['line_number']}: {line['ai_prob']:.3f}")
    
    def stream_progress(self, current: int, total: int, message: str = ""):
        """流式输出进度信息"""
        percentage = (current / total) * 100 if total > 0 else 0
        progress_bar = "█" * int(percentage // 5) + "░" * (20 - int(percentage // 5))
        self.output_stream(f"\r[{progress_bar}] {percentage:.1f}% {message}", end="")
    
    def flush_buffer(self):
        """清空缓冲区"""
        self.buffer.clear() 