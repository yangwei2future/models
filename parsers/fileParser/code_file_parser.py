import ast
import re
import os
import json
import logging
from typing import Dict, List, Any, Optional

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeFileParser:
    """
    代码文件解析器，用于将源代码文件转换为结构化数据
    支持 Python、Java、JavaScript 三种编程语言
    """
    
    def __init__(self, context_window_size: int = 2):
        """
        初始化解析器
        :param context_window_size: 上下文窗口大小（当前行前后行数）
        """
        self.context_window_size = context_window_size
        
        # 语言检测的文件扩展名映射
        self.extension_map = {
            '.py': 'Python',
            '.java': 'Java', 
            '.js': 'JavaScript',
            '.jsx': 'JavaScript',
            '.ts': 'JavaScript',
            '.tsx': 'JavaScript'
        }
        
        # 语言检测的启发式规则
        self.heuristic_patterns = {
            'Python': [r'def\s+\w+', r'import\s+\w+', r'from\s+\w+\s+import', r'if\s+__name__\s*==\s*["\']__main__["\']'],
            'Java': [r'public\s+class\s+\w+', r'import\s+[\w.]+;', r'public\s+static\s+void\s+main'],
            'JavaScript': [r'function\s+\w+', r'=>', r'var\s+\w+', r'let\s+\w+', r'const\s+\w+', r'require\s*\(']
        }
        
        # 导入语句的正则表达式
        self.import_patterns = {
            'Python': [
                r'import\s+([\w.]+)',
                r'from\s+([\w.]+)\s+import'
            ],
            'Java': [
                r'import\s+([\w.]+);'
            ],
            'JavaScript': [
                r'import.*from\s+["\']([^"\']+)["\']',
                r'import\s+["\']([^"\']+)["\']',
                r'require\s*\(\s*["\']([^"\']+)["\']\s*\)'
            ]
        }

    def parse(self, input_data: Dict[str, str]) -> Dict[str, Any]:
        """
        主解析方法
        :param input_data: 输入字典 {"file_path": str, "content": str}
        :return: 结构化解析结果字典
        """
        try:
            file_path = input_data.get("file_path", "")
            content = input_data.get("content", "")
            
            if not content:
                raise ValueError("Content cannot be empty")
            
            # 1. 识别编程语言
            language = self._detect_language(content, file_path)
            
            # 2. 解析文件级上下文
            file_context = self._parse_file_context(content, language, file_path)
            
            # 3. 解析行级信息
            lines_data = self._parse_lines(content, language)
            
            # 4. 组装结果
            result = {
                "language": language,
                "file_context": file_context,
                "lines": lines_data
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing file: {str(e)}")
            # 返回最基本的解析结果
            return self._create_fallback_result(input_data)

    def _detect_language(self, content: str, file_path: Optional[str] = None) -> str:
        """
        检测编程语言
        :param content: 文件内容
        :param file_path: 文件路径
        :return: 检测到的语言名称
        """
        # 优先根据文件扩展名判断
        if file_path:
            _, ext = os.path.splitext(file_path)
            if ext.lower() in self.extension_map:
                return self.extension_map[ext.lower()]
        
        # 使用启发式规则
        language_scores = {}
        for lang, patterns in self.heuristic_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content, re.IGNORECASE))
                score += matches
            language_scores[lang] = score
        
        # 返回得分最高的语言
        if language_scores and max(language_scores.values()) > 0:
            return max(language_scores.keys(), key=lambda x: language_scores[x])
        
        return "unknown"

    def _parse_file_context(self, content: str, language: str, file_path: str = "") -> Dict[str, Any]:
        """
        解析文件级上下文
        :param content: 文件内容
        :param language: 编程语言
        :param file_path: 文件路径
        :return: 文件上下文字典
        """
        # 提取文件名
        file_name = os.path.basename(file_path) if file_path else "unknown"
        
        # 提取导入列表
        imports = self._extract_imports(content, language)
        
        # 提取类名和函数名（简化版本，取第一个找到的）
        class_name = self._extract_class_name(content, language)
        function_name = self._extract_function_name(content, language)
        
        return {
            "file_name": file_name,
            "imports": imports,
            "class_name": class_name,
            "function_name": function_name
        }

    def _extract_imports(self, content: str, language: str) -> List[str]:
        """
        提取导入列表
        """
        imports = []
        if language in self.import_patterns:
            for pattern in self.import_patterns[language]:
                matches = re.findall(pattern, content)
                imports.extend(matches)
        return list(set(imports))  # 去重

    def _extract_class_name(self, content: str, language: str) -> Optional[str]:
        """
        提取类名
        """
        patterns = {
            'Python': r'class\s+(\w+)',
            'Java': r'(?:public\s+)?class\s+(\w+)',
            'JavaScript': r'class\s+(\w+)'
        }
        
        if language in patterns:
            match = re.search(patterns[language], content)
            if match:
                return match.group(1)
        return None

    def _extract_function_name(self, content: str, language: str) -> Optional[str]:
        """
        提取函数名
        """
        patterns = {
            'Python': r'def\s+(\w+)',
            'Java': r'(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)+(\w+)\s*\(',
            'JavaScript': r'function\s+(\w+)|(\w+)\s*=\s*(?:function|\([^)]*\)\s*=>)'
        }
        
        if language in patterns:
            match = re.search(patterns[language], content)
            if match:
                return match.group(1) or match.group(2)
        return None

    def _parse_lines(self, content: str, language: str) -> List[Dict[str, Any]]:
        """
        解析行级信息
        :param content: 文件内容
        :param language: 编程语言
        :return: 行级数据列表
        """
        lines = content.split('\n')
        lines_data = []
        
        for i, line in enumerate(lines):
            line_number = i + 1
            
            # 提取AST特征
            ast_features = self._extract_ast_features(line, line_number, content, language)
            
            # 构建上下文窗口
            context_window = self._build_context_window(lines, i)
            
            line_data = {
                "line_number": line_number,
                "content": line,
                "ast_features": ast_features,
                "context_window": context_window
            }
            
            lines_data.append(line_data)
        
        return lines_data

    def _extract_ast_features(self, line_content: str, line_num: int, 
                             full_content: str, language: str) -> Dict[str, Any]:
        """
        提取行的AST特征
        :param line_content: 当前行内容
        :param line_num: 行号
        :param full_content: 完整文件内容
        :param language: 编程语言
        :return: AST特征字典
        """
        if language == 'Python':
            return self._extract_python_ast_features(line_content, full_content)
        elif language == 'Java':
            return self._extract_java_ast_features(line_content)
        elif language == 'JavaScript':
            return self._extract_javascript_ast_features(line_content)
        else:
            return {
                "node_type": "unknown",
                "depth": 0,
                "children_types": []
            }

    def _extract_python_ast_features(self, line_content: str, full_content: str) -> Dict[str, Any]:
        """
        提取Python AST特征
        """
        try:
            # 尝试解析单行
            if line_content.strip():
                # 对于单行，我们尝试解析为表达式或语句
                try:
                    # 先尝试作为表达式解析
                    tree = ast.parse(line_content.strip(), mode='eval')
                    node = tree.body
                except SyntaxError:
                    try:
                        # 再尝试作为语句解析
                        tree = ast.parse(line_content.strip(), mode='exec')
                        if tree.body:
                            node = tree.body[0]
                        else:
                            raise SyntaxError("Empty statement")
                    except SyntaxError:
                        # 如果单行解析失败，尝试在完整文件上下文中找到对应节点
                        return self._fallback_ast_analysis(line_content)
                
                return {
                    "node_type": type(node).__name__,
                    "depth": self._calculate_ast_depth(node),
                    "children_types": [type(child).__name__ for child in ast.iter_child_nodes(node)]
                }
            else:
                return {
                    "node_type": "Empty",
                    "depth": 0,
                    "children_types": []
                }
        except Exception as e:
            logger.debug(f"AST parsing error for line '{line_content}': {str(e)}")
            return self._fallback_ast_analysis(line_content)

    def _extract_java_ast_features(self, line_content: str) -> Dict[str, Any]:
        """
        提取Java AST特征（简化版本）
        """
        # 由于Java AST解析较复杂，这里实现基础的模式匹配
        line = line_content.strip()
        
        if not line or line.startswith('//') or line.startswith('/*'):
            return {"node_type": "Comment", "depth": 0, "children_types": []}
        elif line.startswith('import '):
            return {"node_type": "ImportDeclaration", "depth": 0, "children_types": []}
        elif 'class ' in line:
            return {"node_type": "ClassDeclaration", "depth": 1, "children_types": ["Identifier"]}
        elif re.search(r'\b(public|private|protected)\s+.*\(.*\)', line):
            return {"node_type": "MethodDeclaration", "depth": 2, "children_types": ["Identifier", "Parameters"]}
        elif '=' in line and ';' in line:
            return {"node_type": "VariableDeclaration", "depth": 1, "children_types": ["Identifier", "Expression"]}
        elif line.endswith(';'):
            return {"node_type": "ExpressionStatement", "depth": 1, "children_types": ["Expression"]}
        else:
            return {"node_type": "unknown", "depth": 0, "children_types": []}

    def _extract_javascript_ast_features(self, line_content: str) -> Dict[str, Any]:
        """
        提取JavaScript AST特征（简化版本）
        """
        line = line_content.strip()
        
        if not line or line.startswith('//') or line.startswith('/*'):
            return {"node_type": "Comment", "depth": 0, "children_types": []}
        elif line.startswith('import ') or line.startswith('const ') and 'require(' in line:
            return {"node_type": "ImportDeclaration", "depth": 0, "children_types": []}
        elif line.startswith('function '):
            return {"node_type": "FunctionDeclaration", "depth": 1, "children_types": ["Identifier", "Parameters"]}
        elif '=>' in line:
            return {"node_type": "ArrowFunctionExpression", "depth": 1, "children_types": ["Parameters", "BlockStatement"]}
        elif 'class ' in line:
            return {"node_type": "ClassDeclaration", "depth": 1, "children_types": ["Identifier"]}
        elif re.search(r'\b(var|let|const)\s+\w+', line):
            return {"node_type": "VariableDeclaration", "depth": 1, "children_types": ["Identifier"]}
        else:
            return {"node_type": "unknown", "depth": 0, "children_types": []}

    def _fallback_ast_analysis(self, line_content: str) -> Dict[str, Any]:
        """
        AST解析失败时的降级分析
        """
        line = line_content.strip()
        
        # 基础的语法模式匹配
        if not line:
            return {"node_type": "Empty", "depth": 0, "children_types": []}
        elif line.startswith('#'):
            return {"node_type": "Comment", "depth": 0, "children_types": []}
        elif line.startswith('def '):
            return {"node_type": "FunctionDef", "depth": 1, "children_types": ["Name", "arguments"]}
        elif line.startswith('class '):
            return {"node_type": "ClassDef", "depth": 1, "children_types": ["Name"]}
        elif line.startswith('import ') or line.startswith('from '):
            return {"node_type": "Import", "depth": 0, "children_types": ["alias"]}
        elif '=' in line:
            return {"node_type": "Assign", "depth": 1, "children_types": ["Name", "Constant"]}
        else:
            return {"node_type": "unknown", "depth": 0, "children_types": []}

    def _calculate_ast_depth(self, node) -> int:
        """
        计算AST节点深度
        """
        if not hasattr(node, '_fields') or not node._fields:
            return 0
        
        max_depth = 0
        for child in ast.iter_child_nodes(node):
            child_depth = self._calculate_ast_depth(child)
            max_depth = max(max_depth, child_depth)
        
        return max_depth + 1

    def _build_context_window(self, lines: List[str], current_index: int) -> List[Dict[str, Any]]:
        """
        构建上下文窗口
        :param lines: 所有行的列表
        :param current_index: 当前行索引
        :return: 上下文窗口列表
        """
        context_window = []
        
        start_index = max(0, current_index - self.context_window_size)
        end_index = min(len(lines), current_index + self.context_window_size + 1)
        
        for i in range(start_index, end_index):
            context_window.append({
                "line": i + 1,
                "content": lines[i]
            })
        
        return context_window

    def _create_fallback_result(self, input_data: Dict[str, str]) -> Dict[str, Any]:
        """
        创建降级解析结果
        """
        content = input_data.get("content", "")
        file_path = input_data.get("file_path", "")
        lines = content.split('\n')
        
        lines_data = []
        for i, line in enumerate(lines):
            lines_data.append({
                "line_number": i + 1,
                "content": line,
                "ast_features": {
                    "node_type": "unknown",
                    "depth": 0,
                    "children_types": []
                },
                "context_window": [{"line": i + 1, "content": line}]
            })
        
        return {
            "language": "unknown",
            "file_context": {
                "file_name": os.path.basename(file_path) if file_path else "unknown",
                "imports": [],
                "class_name": None,
                "function_name": None
            },
            "lines": lines_data
        }


# 使用示例
if __name__ == "__main__":
    parser = CodeFileParser()
    
    # 测试Python代码
    test_input = {
        "file_path": "/projects/math_utils.py",
        "content": """import math

def factorial(n):
    '''计算阶乘'''
    return 1 if n <= 1 else n * factorial(n-1)

class MathUtils:
    def __init__(self):
        self.pi = math.pi
    
    def circle_area(self, radius):
        return self.pi * radius ** 2
"""
    }
    
    result = parser.parse(test_input)
    print(json.dumps(result, indent=2, ensure_ascii=False)) 