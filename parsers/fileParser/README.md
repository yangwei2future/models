# CodeFileParser - 代码文件解析器

一个用于代码来源检测系统的Python文件解析器，能够将源代码文件转换为结构化数据，支持Python、Java、JavaScript三种编程语言。

## 功能特性

### 🔍 语言检测
- **智能识别**: 优先根据文件扩展名，无扩展名时使用启发式规则
- **支持语言**: Python (.py)、Java (.java)、JavaScript (.js/.jsx/.ts/.tsx)
- **降级处理**: 无法识别的语言标记为"unknown"

### 📁 文件级上下文提取
- **文件信息**: 自动提取文件名
- **导入分析**: 识别各语言的import/require语句
- **结构检测**: 提取第一个类名和函数名

### 📝 行级解析
- **AST特征**: 提取每行的抽象语法树节点类型、深度和子节点信息
- **上下文窗口**: 为每行构建可配置大小的上下文环境
- **逐行处理**: 详细的行号和内容映射

### 🛡️ 错误处理
- **鲁棒性**: 单行解析错误不影响整个文件处理
- **降级策略**: AST解析失败时使用模式匹配
- **日志记录**: 详细的错误日志和调试信息

## 安装和使用

### 基础使用

```python
from code_file_parser import CodeFileParser

# 创建解析器实例
parser = CodeFileParser()

# 准备输入数据
input_data = {
    "file_path": "example.py",  # 可选，用于语言检测和文件名提取
    "content": """
import math

def factorial(n):
    return 1 if n <= 1 else n * factorial(n-1)

class MathUtils:
    def circle_area(self, radius):
        return math.pi * radius ** 2
"""
}

# 解析代码
result = parser.parse(input_data)

# 查看结果
print(f"检测语言: {result['language']}")
print(f"导入包: {result['file_context']['imports']}")
print(f"代码行数: {len(result['lines'])}")
```

### 自定义配置

```python
# 自定义上下文窗口大小
parser = CodeFileParser(context_window_size=3)  # 默认为2
```

## 输出格式

```json
{
  "language": "Python",
  "file_context": {
    "file_name": "example.py",
    "imports": ["math"],
    "class_name": "MathUtils",
    "function_name": "factorial"
  },
  "lines": [
    {
      "line_number": 1,
      "content": "import math",
      "ast_features": {
        "node_type": "Import",
        "depth": 2,
        "children_types": ["alias"]
      },
      "context_window": [
        {"line": 1, "content": "import math"},
        {"line": 2, "content": ""},
        {"line": 3, "content": "def factorial(n):"}
      ]
    }
  ]
}
```

## 核心类和方法

### CodeFileParser

主解析器类，提供以下核心方法：

- `__init__(context_window_size=2)`: 初始化解析器
- `parse(input_data)`: 主解析方法，返回结构化结果

### 内部方法

- `_detect_language()`: 语言检测
- `_parse_file_context()`: 文件上下文解析  
- `_parse_lines()`: 行级信息解析
- `_extract_ast_features()`: AST特征提取
- `_build_context_window()`: 上下文窗口构建

## 语言支持详情

### Python
- **AST解析**: 使用标准库`ast`模块
- **特征识别**: def、class、import语句
- **导入提取**: import和from...import语句

### Java  
- **模式匹配**: 基于正则表达式的简化AST
- **特征识别**: class、method、import声明
- **导入提取**: import语句解析

### JavaScript
- **模式匹配**: 识别function、class、箭头函数
- **特征识别**: ES6+语法支持
- **导入提取**: import和require语句

## 示例和测试

运行测试脚本查看各种使用场景：

```bash
python test_parser.py        # 基础功能测试
python example_usage.py      # 详细使用示例
```

### 测试覆盖

- ✅ 多语言代码解析
- ✅ AST特征提取
- ✅ 上下文窗口构建
- ✅ 错误处理和降级
- ✅ 边界情况处理

## 设计特点

1. **模块化设计**: 清晰的方法分离，易于扩展和维护
2. **性能优化**: 逐行处理，避免大文件内存问题
3. **容错机制**: 优雅处理语法错误和解析失败
4. **标准化输出**: 统一的JSON格式，便于后续处理
5. **日志支持**: 可配置的日志级别和调试信息

## 应用场景

- **代码来源检测**: 为AI生成代码检测提供结构化特征
- **代码分析工具**: 静态分析和代码质量评估
- **教育工具**: 代码结构可视化和学习辅助
- **开发工具**: IDE插件和代码处理管道

## 注意事项

1. **依赖要求**: 仅使用Python标准库，无外部依赖
2. **性能考虑**: 适合中小型文件，大文件建议分块处理
3. **精度权衡**: Java和JavaScript使用简化AST，精度相对较低
4. **编码支持**: 假设输入为UTF-8编码的文本

## 扩展建议

如需支持更多语言或提高解析精度，可以考虑：

- 集成专业的AST解析库（如javalang、esprima）
- 添加更多启发式规则
- 支持更多文件格式和编程语言
- 优化大文件处理性能 