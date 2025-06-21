#!/usr/bin/env python3
"""
文件解析器模块
架构流程第1步：解析代码文件并提取基础信息
"""

import os
from typing import List, Dict, Any


class FileParser:
    """文件解析器 - 解析代码文件并提取基础信息"""
    
    def __init__(self):
        self.supported_extensions = {
            '.py', '.java', '.js', '.jsx', '.ts', '.tsx',
            '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx',
            '.go', '.rs', '.rb', '.php', '.swift', '.kt'
        }
    
    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """
        解析单个文件
        
        Args:
            filepath: 文件路径
            
        Returns:
            Dict: 包含文件信息和逐行数据的字典
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # 基础文件信息
            file_info = {
                'filepath': filepath,
                'filename': os.path.basename(filepath),
                'extension': os.path.splitext(filepath)[1],
                'total_lines': len(lines),
                'lines': [],
                'success': True
            }
            
            # 逐行解析
            for i, line in enumerate(lines, 1):
                content = line.rstrip('\n\r')
                line_info = {
                    'line_number': i,
                    'content': content,
                    'original_content': line,
                    'is_empty': not content.strip(),
                    'indent_level': len(content) - len(content.lstrip()) if content else 0
                }
                file_info['lines'].append(line_info)
            
            return file_info
            
        except Exception as e:
            return {
                'filepath': filepath,
                'success': False,
                'error': str(e)
            }
    
    def parse_directory(self, dirpath: str, recursive: bool = False) -> List[Dict[str, Any]]:
        """
        解析目录中的所有代码文件
        
        Args:
            dirpath: 目录路径
            recursive: 是否递归处理子目录
            
        Returns:
            List[Dict]: 文件解析结果列表
        """
        files = []
        
        if recursive:
            for root, _, filenames in os.walk(dirpath):
                for filename in filenames:
                    if self.is_code_file(filename):
                        filepath = os.path.join(root, filename)
                        files.append(self.parse_file(filepath))
        else:
            for filename in os.listdir(dirpath):
                filepath = os.path.join(dirpath, filename)
                if os.path.isfile(filepath) and self.is_code_file(filename):
                    files.append(self.parse_file(filepath))
        
        return files
    
    def is_code_file(self, filename: str) -> bool:
        """
        判断是否为支持的代码文件
        
        Args:
            filename: 文件名
            
        Returns:
            bool: 是否为代码文件
        """
        _, ext = os.path.splitext(filename.lower())
        return ext in self.supported_extensions 