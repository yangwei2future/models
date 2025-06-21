#!/usr/bin/env python3
"""
特征提取模块
架构流程第2-3步：文件级特征提取和行级特征提取
"""

import numpy as np
from typing import List, Dict, Any


class LineFeatureExtractor:
    """行级特征提取器 - 提取每行代码的特征向量(19维)"""
    
    def __init__(self):
        # AI生成代码的模式
        self.ai_patterns = [
            'comprehensive', 'sophisticated', 'advanced', 'implementation',
            'configuration', 'initialization', 'processing', 'algorithm',
            'calculate', 'compute', 'optimize', 'generate', 'create',
            'implement', 'function', 'method', 'handle', 'manage', 'execute'
        ]
        
        # 代码风格指标
        self.style_indicators = {
            'docstring_start': ['"""', "'''", '/**'],
            'type_annotations': [':', '->', 'List[', 'Dict[', 'Optional[', 'Union['],
            'logging_patterns': ['logger.', 'logging.', 'log.'],
            'exception_patterns': ['raise', 'except', 'try:', 'finally:'],
            'import_patterns': ['from typing import', 'from abc import', 'import logging']
        }
    
    def extract_features(self, line_info: Dict[str, Any], file_context: Dict[str, Any]) -> List[float]:
        """
        提取单行的特征向量
        
        Args:
            line_info: 行信息字典
            file_context: 文件上下文信息
            
        Returns:
            List[float]: 19维特征向量
        """
        content = line_info['content']
        line_number = line_info['line_number']
        total_lines = file_context['total_lines']
        
        # 基础特征 (4维)
        '''计算方式: 代码行字符长度，除以100归一化，上限1.0, 意义: 反映单行代码的复杂程度和信息量, 特点: AI代码可能有更长的行，包含更多信息'''
        length = len(content)
        '''计算方式: 行缩进级别，除以20归一化，上限1.0, 意义: 反映代码的嵌套深度和结构层次, 特点: AI代码可能有更深的嵌套结构'''
        indent_level = line_info['indent_level']
        '''计算方式: 行号 ÷ 总行数，归一化到0-1, 意义: 反映代码在文件中的位置, 特点: AI代码可能更集中于某些区域'''
        relative_position = line_number / total_lines if total_lines > 0 else 0
        '''计算方式: 代码行字符串分割后的单词数量, 意义: 反映代码的复杂度和信息量, 特点: AI代码可能有更长的行，包含更多信息'''
        word_count = len(content.split())
        
        # 内容特征 (4维)
        '''计算方式: 代码行是否包含注释, 意义: 反映代码的文档完整性, 特点: AI代码通常注释更充分'''
        has_comment = '#' in content or '//' in content or '/*' in content
        '''计算方式: 代码行是否包含字符串, 意义: 反映代码的复杂度和信息量, 特点: AI代码可能有更长的行，包含更多信息'''
        has_string = '"' in content or "'" in content or '`' in content
        '''计算方式: 代码行是否包含数字, 意义: 反映代码的复杂度和信息量, 特点: AI代码可能有更长的行，包含更多信息'''
        has_number = any(c.isdigit() for c in content)
        '''计算方式: 代码行是否包含操作符, 意义: 反映代码的复杂度和信息量, 特点: AI代码可能有更长的行，包含更多信息'''
        has_operator = any(op in content for op in ['=', '+', '-', '*', '/', '<', '>', '!', '&', '|'])
        
        # 复杂度特征 (3维)
        '''计算方式: 代码行中括号、方括号、花括号数量之和, 意义: 反映代码的结构复杂度, 特点: AI代码可能使用更多的嵌套结构'''
        bracket_complexity = content.count('(') + content.count('[') + content.count('{')
        '''计算方式: 代码行字符串分割后的单词数量 ÷ 代码行字符长度, 意义: 反映代码的词汇密度, 特点: AI代码可能有更密集的词汇'''
        word_density = word_count / max(len(content), 1)
        '''计算方式: 代码行字符串分割后的单词数量 ÷ 代码行字符长度, 意义: 反映代码的词汇多样性, 特点: AI代码可能有更丰富的词汇'''
        char_diversity = len(set(content)) / max(len(content), 1)

        # AI模式特征 (5维)
        '''计算方式: 代码行是否包含AI生成代码的模式, 意义: 反映代码的AI生成程度, 特点: AI代码可能包含更多的AI生成模式'''
        has_ai_pattern = any(pattern in content.lower() for pattern in self.ai_patterns)
        '''计算方式: 代码行是否包含文档字符串, 意义: 反映代码的文档完整性, 特点: AI代码通常有更完整的文档'''
        has_docstring = any(pattern in content for pattern in self.style_indicators['docstring_start'])
        '''计算方式: 代码行是否包含类型注解, 意义: 反映代码的类型系统使用程度, 特点: AI代码更倾向于使用类型提示'''
        has_type_annotation = any(pattern in content for pattern in self.style_indicators['type_annotations'])
        '''计算方式: 代码行是否包含日志相关语句, 意义: 反映代码的日志系统完整性, 特点: AI代码通常有更完善的日志'''
        has_logging = any(pattern in content for pattern in self.style_indicators['logging_patterns'])
        '''计算方式: 代码行是否包含高级导入语句, 意义: 反映代码的依赖复杂度, 特点: AI代码可能导入更多模块'''
        has_advanced_import = any(pattern in content for pattern in self.style_indicators['import_patterns'])
        
        # 上下文特征 (3维)
        '''计算方式: 代码行是否为首行, 意义: 反映代码的结构完整性, 特点: AI代码可能更集中于某些区域'''
        is_first_line = line_number == 1
        '''计算方式: 代码行是否为末行, 意义: 反映代码的结构完整性, 特点: AI代码可能更集中于某些区域'''
        is_last_line = line_number == total_lines
        '''计算方式: 代码行是否为中间段, 意义: 反映代码的结构完整性, 特点: AI代码可能更集中于某些区域'''
        is_middle_section = 0.3 < relative_position < 0.7
        
        # 归一化特征
        features = [
            # 基础特征 (4维)
            min(length / 100.0, 1.0),           # 0. 归一化长度
            min(indent_level / 20.0, 1.0),      # 1. 归一化缩进
            relative_position,                   # 2. 相对位置
            min(word_count / 20.0, 1.0),        # 3. 词汇数量
            
            # 内容特征 (4维)
            float(has_comment),                  # 4. 注释
            float(has_string),                   # 5. 字符串
            float(has_number),                   # 6. 数字
            float(has_operator),                 # 7. 操作符
            
            # 复杂度特征 (3维)
            min(bracket_complexity / 10.0, 1.0), # 8. 括号复杂度
            word_density,                        # 9. 词汇密度
            char_diversity,                      # 10. 字符多样性
            
            # AI模式特征 (5维)
            float(has_ai_pattern),               # 11. AI模式
            float(has_docstring),                # 12. 文档字符串
            float(has_type_annotation),          # 13. 类型注解
            float(has_logging),                  # 14. 日志记录
            float(has_advanced_import),          # 15. 高级导入
            
            # 上下文特征 (3维)
            float(is_first_line),                # 16. 首行
            float(is_last_line),                 # 17. 末行
            float(is_middle_section),            # 18. 中间段
        ]
        
        return features


class FileFeatureExtractor:
    """文件级特征提取器 - 提取整个文件的统计特征(14维)"""
    
    def extract_features(self, file_info: Dict[str, Any]) -> List[float]:
        """
        提取文件级特征
        
        Args:
            file_info: 文件信息字典
            
        Returns:
            List[float]: 14维特征向量
        """
        lines = file_info['lines']
        total_lines = len(lines)
        
        if total_lines == 0:
            return [0.0] * 14  # 返回14维零向量
        
        # 统计特征
        non_empty_lines = [line for line in lines if not line['is_empty']]
        comment_lines = [line for line in non_empty_lines if '#' in line['content'] or '//' in line['content']]
        
        # 基础统计 (5维)
        '''计算方式: 文件总行数，归一化到1000行, 意义: 反映代码文件的规模大小, 特点: AI生成代码通常更长，功能更完整'''
        file_size = total_lines
        '''计算方式: 非空行数 ÷ 总行数, 意义: 反映代码的紧凑程度, 特点: AI代码可能有更规律的空行分布'''
        code_density = len(non_empty_lines) / total_lines
        '''计算方式: 注释行数 ÷ 非空行数, 意义: 反映代码的文档完整性, 特点: AI代码通常注释更充分'''
        comment_ratio = len(comment_lines) / max(len(non_empty_lines), 1)
        '''计算方式: 所有非空行的平均字符长度, 意义: 反映代码的复杂度和可读性, 特点: AI代码可能有更长的行长度'''
        avg_line_length = np.mean([len(line['content']) for line in non_empty_lines]) if non_empty_lines else 0
        '''计算方式: 所有非空行的平均缩进级别, 意义: 反映代码的嵌套深度, 特点: AI代码可能有更深的嵌套结构'''
        avg_indent = np.mean([line['indent_level'] for line in non_empty_lines]) if non_empty_lines else 0
        
        # 复杂度特征 (4维)
        '''计算方式: 所有括号数量的平均值, 意义: 反映代码的结构复杂度, 特点: AI代码可能使用更多的嵌套结构'''
        total_brackets = sum(line['content'].count('(') + line['content'].count('[') + line['content'].count('{') 
                           for line in non_empty_lines)
        avg_complexity = total_brackets / max(len(non_empty_lines), 1)
        
        '''计算方式: 统计函数定义的数量, 意义: 反映代码的模块化程度, 特点: AI代码可能更倾向于函数拆分'''
        function_count = sum(1 for line in non_empty_lines if 'def ' in line['content'] or 'function ' in line['content'])
        '''计算方式: 统计类定义的数量, 意义: 反映面向对象的程度, 特点: AI代码可能更多使用类封装'''
        class_count = sum(1 for line in non_empty_lines if 'class ' in line['content'])
        '''计算方式: 统计导入语句的数量, 意义: 反映依赖复杂度, 特点: AI代码可能导入更多模块'''
        import_count = sum(1 for line in non_empty_lines if line['content'].strip().startswith(('import ', 'from ')))
        
        # AI风格特征 (5维)
        '''计算方式: 统计文档字符串的数量, 意义: 反映文档完整性, 特点: AI代码通常有更完整的文档'''
        docstring_count = sum(1 for line in non_empty_lines if '"""' in line['content'] or "'''" in line['content'])
        '''计算方式: 统计类型注解的数量, 意义: 反映类型系统使用程度, 特点: AI代码更倾向于使用类型提示'''
        type_annotation_count = sum(1 for line in non_empty_lines if '->' in line['content'] or ': ' in line['content'])
        '''计算方式: 统计日志相关语句数量, 意义: 反映日志系统完整性, 特点: AI代码通常有更完善的日志'''
        logging_count = sum(1 for line in non_empty_lines if 'logger' in line['content'] or 'logging' in line['content'])
        '''计算方式: 统计异常处理相关语句, 意义: 反映错误处理完整性, 特点: AI代码可能有更多的异常处理'''
        exception_count = sum(1 for line in non_empty_lines if any(keyword in line['content'] for keyword in ['try:', 'except', 'raise', 'finally:']))
        '''计算方式: 统计高级词汇的使用频率, 意义: 反映代码风格的专业程度, 特点: AI代码倾向于使用更专业的词汇'''
        advanced_pattern_count = sum(1 for line in non_empty_lines if any(pattern in line['content'].lower() 
                                   for pattern in ['comprehensive', 'sophisticated', 'implementation', 'configuration']))
        
        # 归一化特征
        features = [
            # 基础统计 (5维)
            min(file_size / 1000.0, 1.0),           # 0. 文件大小
            code_density,                            # 1. 代码密度
            comment_ratio,                           # 2. 注释比例
            min(avg_line_length / 80.0, 1.0),       # 3. 平均行长度
            min(avg_indent / 10.0, 1.0),            # 4. 平均缩进
            
            # 复杂度特征 (4维)
            min(avg_complexity / 5.0, 1.0),         # 5. 平均复杂度
            min(function_count / 20.0, 1.0),        # 6. 函数数量
            min(class_count / 10.0, 1.0),           # 7. 类数量
            min(import_count / 20.0, 1.0),          # 8. 导入数量
            
            # AI风格特征 (5维)
            min(docstring_count / 10.0, 1.0),       # 9. 文档字符串
            min(type_annotation_count / 20.0, 1.0), # 10. 类型注解
            min(logging_count / 10.0, 1.0),         # 11. 日志记录
            min(exception_count / 10.0, 1.0),       # 12. 异常处理
            min(advanced_pattern_count / 10.0, 1.0) # 13. 高级模式
        ]
        
        return features 