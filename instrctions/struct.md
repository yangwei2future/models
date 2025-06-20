{
    "language": "python",
    "file_context": {
      "file_name": "math_utils.py",
      "imports": ["math"],
      "class_name": null,
      "function_name": "calculate_factorial"
    },
    "lines": [
      {
        "line_number": 1,
        "content": "import math",
        "ast_features": {
          "node_type": "import_statement",
          "depth": 1,
          "children_types": ["module_name"]
        },
        "context_window": [
          {"line": 1, "content": "import math"},
          {"line": 2, "content": ""},
          {"line": 3, "content": "def calculate_factorial(n):"}
        ]
      },
      {
        "line_number": 3,
        "content": "def calculate_factorial(n):",
        "ast_features": {
          "node_type": "function_definition",
          "depth": 1,
          "children_types": ["function_name", "parameters"]
        },
        "context_window": [
          {"line": 2, "content": ""},
          {"line": 3, "content": "def calculate_factorial(n):"},
          {"line": 4, "content": "    if n < 0:"}
        ]
      },
      {
        "line_number": 4,
        "content": "    if n < 0:",
        "ast_features": {
          "node_type": "if_statement",
          "depth": 2,
          "children_types": ["comparison", "body"]
        },
        "context_window": [
          {"line": 3, "content": "def calculate_factorial(n):"},
          {"line": 4, "content": "    if n < 0:"},
          {"line": 5, "content": "        raise ValueError(\"Factorial not defined for negative numbers\")"}
        ]
      },
      // ... 其他行
    ]
  }。这个现在是我现在训练时的输入，包含lable一系列的输入。{
  "type": "single_file",
  "result": {
    "language": "Python",
    "file_context": {
      "file_name": "code_file_parser.py",
      "imports": [
        "logging",
        "os",
        "typing",
        "json",
        "math",
        "Dict",
        "re",
        "ast"
      ],
      "class_name": "CodeFileParser",
      "function_name": "__init__"
    },
    "lines": [
      {
        "line_number": 1,
        "content": "import ast",
        "ast_features": {
          "node_type": "Import",
          "depth": 2,
          "children_types": [
            "alias"
          ]
        },
        "context_window": [
          {
            "line": 1,
            "content": "import ast"
          },
          {
            "line": 2,
            "content": "import re"
          },
          {
            "line": 3,
            "content": "import os"
          }
        ]
      },
      {
        "line_number": 2,
        "content": "import re",
        "ast_features": {
          "node_type": "Import",
          "depth": 2,
          "children_types": [
            "alias"
          ]
        },
        "context_window": [
          {
            "line": 1,
            "content": "import ast"
          },
          {
            "line": 2,
            "content": "import re"
          },
          {
            "line": 3,
            "content": "import os"
          },
          {
            "line": 4,
            "content": "import json"
          }
        ]
      },
      {
        "line_number": 3,
        "content": "import os",
        "ast_features": {
          "node_type": "Import",
          "depth": 2,
          "children_types": [
            "alias"
          ]
        },
        "context_window": [
          {
            "line": 1,
            "content": "import ast"
          },
          {
            "line": 2,
            "content": "import re"
          },
          {
            "line": 3,
            "content": "import os"
          },
          {
            "line": 4,
            "content": "import json"
          },
          {
            "line": 5,
            "content": "import logging"
          }
        ]
      },
      {
        "line_number": 4,
        "content": "import json",
        "ast_features": {
          "node_type": "Import",
          "depth": 2,
          "children_types": [
            "alias"
          ]
        },
        "context_window": [
          {
            "line": 2,
            "content": "import re"
          },
          {
            "line": 3,
            "content": "import os"
          },
          {
            "line": 4,
            "content": "import json"
          },
          {
            "line": 5,
            "content": "import logging"
          },
          {
            "line": 6,
            "content": "from typing import Dict, List, Any, Optional"
          }
        ]
      },
      {
        "line_number": 5,
        "content": "import logging",
        "ast_features": {
          "node_type": "Import",
          "depth": 2,
          "children_types": [
            "alias"
          ]
        },
        "context_window": [
          {
            "line": 3,
            "content": "import os"
          },
          {
            "line": 4,
            "content": "import json"
          },
          {
            "line": 5,
            "content": "import logging"
          },
          {
            "line": 6,
            "content": "from typing import Dict, List, Any, Optional"
          },
          {
            "line": 7,
            "content": ""
          }
        ]
      },
      {
        "line_number": 6,
        "content": "from typing import Dict, List, Any, Optional",
        "ast_features": {
          "node_type": "ImportFrom",
          "depth": 2,
          "children_types": [
            "alias",
            "alias",
            "alias",
            "alias"
          ]
        },
        "context_window": [
          {
            "line": 4,
            "content": "import json"
          },
          {
            "line": 5,
            "content": "import logging"
          },
          {
            "line": 6,
            "content": "from typing import Dict, List, Any, Optional"
          },
          {
            "line": 7,
            "content": ""
          },
          {
            "line": 8,
            "content": "# 配置日志"
          }
        ]
      },
      {
        "line_number": 7,
        "content": "",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 5,
            "content": "import logging"
          },
          {
            "line": 6,
            "content": "from typing import Dict, List, Any, Optional"
          },
          {
            "line": 7,
            "content": ""
          },
          {
            "line": 8,
            "content": "# 配置日志"
          },
          {
            "line": 9,
            "content": "logging.basicConfig(level=logging.INFO)"
          }
        ]
      },
      {
        "line_number": 8,
        "content": "# 配置日志",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 6,
            "content": "from typing import Dict, List, Any, Optional"
          },
          {
            "line": 7,
            "content": ""
          },
          {
            "line": 8,
            "content": "# 配置日志"
          },
          {
            "line": 9,
            "content": "logging.basicConfig(level=logging.INFO)"
          },
          {
            "line": 10,
            "content": "logger = logging.getLogger(__name__)"
          }
        ]
      },
      {
        "line_number": 9,
        "content": "logging.basicConfig(level=logging.INFO)",
        "ast_features": {
          "node_type": "Call",
          "depth": 4,
          "children_types": [
            "Attribute",
            "keyword"
          ]
        },
        "context_window": [
          {
            "line": 7,
            "content": ""
          },
          {
            "line": 8,
            "content": "# 配置日志"
          },
          {
            "line": 9,
            "content": "logging.basicConfig(level=logging.INFO)"
          },
          {
            "line": 10,
            "content": "logger = logging.getLogger(__name__)"
          },
          {
            "line": 11,
            "content": ""
          }
        ]
      },
      {
        "line_number": 10,
        "content": "logger = logging.getLogger(__name__)",
        "ast_features": {
          "node_type": "Assign",
          "depth": 4,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 8,
            "content": "# 配置日志"
          },
          {
            "line": 9,
            "content": "logging.basicConfig(level=logging.INFO)"
          },
          {
            "line": 10,
            "content": "logger = logging.getLogger(__name__)"
          },
          {
            "line": 11,
            "content": ""
          },
          {
            "line": 12,
            "content": "class CodeFileParser:"
          }
        ]
      },
      {
        "line_number": 11,
        "content": "",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 9,
            "content": "logging.basicConfig(level=logging.INFO)"
          },
          {
            "line": 10,
            "content": "logger = logging.getLogger(__name__)"
          },
          {
            "line": 11,
            "content": ""
          },
          {
            "line": 12,
            "content": "class CodeFileParser:"
          },
          {
            "line": 13,
            "content": "    \"\"\""
          }
        ]
      },
      {
        "line_number": 12,
        "content": "class CodeFileParser:",
        "ast_features": {
          "node_type": "ClassDef",
          "depth": 1,
          "children_types": [
            "Name"
          ]
        },
        "context_window": [
          {
            "line": 10,
            "content": "logger = logging.getLogger(__name__)"
          },
          {
            "line": 11,
            "content": ""
          },
          {
            "line": 12,
            "content": "class CodeFileParser:"
          },
          {
            "line": 13,
            "content": "    \"\"\""
          },
          {
            "line": 14,
            "content": "    代码文件解析器，用于将源代码文件转换为结构化数据"
          }
        ]
      },
      {
        "line_number": 13,
        "content": "    \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 11,
            "content": ""
          },
          {
            "line": 12,
            "content": "class CodeFileParser:"
          },
          {
            "line": 13,
            "content": "    \"\"\""
          },
          {
            "line": 14,
            "content": "    代码文件解析器，用于将源代码文件转换为结构化数据"
          },
          {
            "line": 15,
            "content": "    支持 Python、Java、JavaScript 三种编程语言"
          }
        ]
      },
      {
        "line_number": 14,
        "content": "    代码文件解析器，用于将源代码文件转换为结构化数据",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 12,
            "content": "class CodeFileParser:"
          },
          {
            "line": 13,
            "content": "    \"\"\""
          },
          {
            "line": 14,
            "content": "    代码文件解析器，用于将源代码文件转换为结构化数据"
          },
          {
            "line": 15,
            "content": "    支持 Python、Java、JavaScript 三种编程语言"
          },
          {
            "line": 16,
            "content": "    \"\"\""
          }
        ]
      },
      {
        "line_number": 15,
        "content": "    支持 Python、Java、JavaScript 三种编程语言",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 13,
            "content": "    \"\"\""
          },
          {
            "line": 14,
            "content": "    代码文件解析器，用于将源代码文件转换为结构化数据"
          },
          {
            "line": 15,
            "content": "    支持 Python、Java、JavaScript 三种编程语言"
          },
          {
            "line": 16,
            "content": "    \"\"\""
          },
          {
            "line": 17,
            "content": "    "
          }
        ]
      },
      {
        "line_number": 16,
        "content": "    \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 14,
            "content": "    代码文件解析器，用于将源代码文件转换为结构化数据"
          },
          {
            "line": 15,
            "content": "    支持 Python、Java、JavaScript 三种编程语言"
          },
          {
            "line": 16,
            "content": "    \"\"\""
          },
          {
            "line": 17,
            "content": "    "
          },
          {
            "line": 18,
            "content": "    def __init__(self, context_window_size: int = 2):"
          }
        ]
      },
      {
        "line_number": 17,
        "content": "    ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 15,
            "content": "    支持 Python、Java、JavaScript 三种编程语言"
          },
          {
            "line": 16,
            "content": "    \"\"\""
          },
          {
            "line": 17,
            "content": "    "
          },
          {
            "line": 18,
            "content": "    def __init__(self, context_window_size: int = 2):"
          },
          {
            "line": 19,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 18,
        "content": "    def __init__(self, context_window_size: int = 2):",
        "ast_features": {
          "node_type": "FunctionDef",
          "depth": 1,
          "children_types": [
            "Name",
            "arguments"
          ]
        },
        "context_window": [
          {
            "line": 16,
            "content": "    \"\"\""
          },
          {
            "line": 17,
            "content": "    "
          },
          {
            "line": 18,
            "content": "    def __init__(self, context_window_size: int = 2):"
          },
          {
            "line": 19,
            "content": "        \"\"\""
          },
          {
            "line": 20,
            "content": "        初始化解析器"
          }
        ]
      },
      {
        "line_number": 19,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 17,
            "content": "    "
          },
          {
            "line": 18,
            "content": "    def __init__(self, context_window_size: int = 2):"
          },
          {
            "line": 19,
            "content": "        \"\"\""
          },
          {
            "line": 20,
            "content": "        初始化解析器"
          },
          {
            "line": 21,
            "content": "        :param context_window_size: 上下文窗口大小（当前行前后行数）"
          }
        ]
      },
      {
        "line_number": 20,
        "content": "        初始化解析器",
        "ast_features": {
          "node_type": "Name",
          "depth": 1,
          "children_types": [
            "Load"
          ]
        },
        "context_window": [
          {
            "line": 18,
            "content": "    def __init__(self, context_window_size: int = 2):"
          },
          {
            "line": 19,
            "content": "        \"\"\""
          },
          {
            "line": 20,
            "content": "        初始化解析器"
          },
          {
            "line": 21,
            "content": "        :param context_window_size: 上下文窗口大小（当前行前后行数）"
          },
          {
            "line": 22,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 21,
        "content": "        :param context_window_size: 上下文窗口大小（当前行前后行数）",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 19,
            "content": "        \"\"\""
          },
          {
            "line": 20,
            "content": "        初始化解析器"
          },
          {
            "line": 21,
            "content": "        :param context_window_size: 上下文窗口大小（当前行前后行数）"
          },
          {
            "line": 22,
            "content": "        \"\"\""
          },
          {
            "line": 23,
            "content": "        self.context_window_size = context_window_size"
          }
        ]
      },
      {
        "line_number": 22,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 20,
            "content": "        初始化解析器"
          },
          {
            "line": 21,
            "content": "        :param context_window_size: 上下文窗口大小（当前行前后行数）"
          },
          {
            "line": 22,
            "content": "        \"\"\""
          },
          {
            "line": 23,
            "content": "        self.context_window_size = context_window_size"
          },
          {
            "line": 24,
            "content": "        "
          }
        ]
      },
      {
        "line_number": 23,
        "content": "        self.context_window_size = context_window_size",
        "ast_features": {
          "node_type": "Assign",
          "depth": 3,
          "children_types": [
            "Attribute",
            "Name"
          ]
        },
        "context_window": [
          {
            "line": 21,
            "content": "        :param context_window_size: 上下文窗口大小（当前行前后行数）"
          },
          {
            "line": 22,
            "content": "        \"\"\""
          },
          {
            "line": 23,
            "content": "        self.context_window_size = context_window_size"
          },
          {
            "line": 24,
            "content": "        "
          },
          {
            "line": 25,
            "content": "        # 语言检测的文件扩展名映射"
          }
        ]
      },
      {
        "line_number": 24,
        "content": "        ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 22,
            "content": "        \"\"\""
          },
          {
            "line": 23,
            "content": "        self.context_window_size = context_window_size"
          },
          {
            "line": 24,
            "content": "        "
          },
          {
            "line": 25,
            "content": "        # 语言检测的文件扩展名映射"
          },
          {
            "line": 26,
            "content": "        self.extension_map = {"
          }
        ]
      },
      {
        "line_number": 25,
        "content": "        # 语言检测的文件扩展名映射",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 23,
            "content": "        self.context_window_size = context_window_size"
          },
          {
            "line": 24,
            "content": "        "
          },
          {
            "line": 25,
            "content": "        # 语言检测的文件扩展名映射"
          },
          {
            "line": 26,
            "content": "        self.extension_map = {"
          },
          {
            "line": 27,
            "content": "            '.py': 'Python',"
          }
        ]
      },
      {
        "line_number": 26,
        "content": "        self.extension_map = {",
        "ast_features": {
          "node_type": "Assign",
          "depth": 1,
          "children_types": [
            "Name",
            "Constant"
          ]
        },
        "context_window": [
          {
            "line": 24,
            "content": "        "
          },
          {
            "line": 25,
            "content": "        # 语言检测的文件扩展名映射"
          },
          {
            "line": 26,
            "content": "        self.extension_map = {"
          },
          {
            "line": 27,
            "content": "            '.py': 'Python',"
          },
          {
            "line": 28,
            "content": "            '.java': 'Java', "
          }
        ]
      },
      {
        "line_number": 27,
        "content": "            '.py': 'Python',",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 25,
            "content": "        # 语言检测的文件扩展名映射"
          },
          {
            "line": 26,
            "content": "        self.extension_map = {"
          },
          {
            "line": 27,
            "content": "            '.py': 'Python',"
          },
          {
            "line": 28,
            "content": "            '.java': 'Java', "
          },
          {
            "line": 29,
            "content": "            '.js': 'JavaScript',"
          }
        ]
      },
      {
        "line_number": 28,
        "content": "            '.java': 'Java', ",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 26,
            "content": "        self.extension_map = {"
          },
          {
            "line": 27,
            "content": "            '.py': 'Python',"
          },
          {
            "line": 28,
            "content": "            '.java': 'Java', "
          },
          {
            "line": 29,
            "content": "            '.js': 'JavaScript',"
          },
          {
            "line": 30,
            "content": "            '.jsx': 'JavaScript',"
          }
        ]
      },
      {
        "line_number": 29,
        "content": "            '.js': 'JavaScript',",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 27,
            "content": "            '.py': 'Python',"
          },
          {
            "line": 28,
            "content": "            '.java': 'Java', "
          },
          {
            "line": 29,
            "content": "            '.js': 'JavaScript',"
          },
          {
            "line": 30,
            "content": "            '.jsx': 'JavaScript',"
          },
          {
            "line": 31,
            "content": "            '.ts': 'JavaScript',"
          }
        ]
      },
      {
        "line_number": 30,
        "content": "            '.jsx': 'JavaScript',",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 28,
            "content": "            '.java': 'Java', "
          },
          {
            "line": 29,
            "content": "            '.js': 'JavaScript',"
          },
          {
            "line": 30,
            "content": "            '.jsx': 'JavaScript',"
          },
          {
            "line": 31,
            "content": "            '.ts': 'JavaScript',"
          },
          {
            "line": 32,
            "content": "            '.tsx': 'JavaScript'"
          }
        ]
      },
      {
        "line_number": 31,
        "content": "            '.ts': 'JavaScript',",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 29,
            "content": "            '.js': 'JavaScript',"
          },
          {
            "line": 30,
            "content": "            '.jsx': 'JavaScript',"
          },
          {
            "line": 31,
            "content": "            '.ts': 'JavaScript',"
          },
          {
            "line": 32,
            "content": "            '.tsx': 'JavaScript'"
          },
          {
            "line": 33,
            "content": "        }"
          }
        ]
      },
      {
        "line_number": 32,
        "content": "            '.tsx': 'JavaScript'",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 30,
            "content": "            '.jsx': 'JavaScript',"
          },
          {
            "line": 31,
            "content": "            '.ts': 'JavaScript',"
          },
          {
            "line": 32,
            "content": "            '.tsx': 'JavaScript'"
          },
          {
            "line": 33,
            "content": "        }"
          },
          {
            "line": 34,
            "content": "        "
          }
        ]
      },
      {
        "line_number": 33,
        "content": "        }",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 31,
            "content": "            '.ts': 'JavaScript',"
          },
          {
            "line": 32,
            "content": "            '.tsx': 'JavaScript'"
          },
          {
            "line": 33,
            "content": "        }"
          },
          {
            "line": 34,
            "content": "        "
          },
          {
            "line": 35,
            "content": "        # 语言检测的启发式规则"
          }
        ]
      },
      {
        "line_number": 34,
        "content": "        ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 32,
            "content": "            '.tsx': 'JavaScript'"
          },
          {
            "line": 33,
            "content": "        }"
          },
          {
            "line": 34,
            "content": "        "
          },
          {
            "line": 35,
            "content": "        # 语言检测的启发式规则"
          },
          {
            "line": 36,
            "content": "        self.heuristic_patterns = {"
          }
        ]
      },
      {
        "line_number": 35,
        "content": "        # 语言检测的启发式规则",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 33,
            "content": "        }"
          },
          {
            "line": 34,
            "content": "        "
          },
          {
            "line": 35,
            "content": "        # 语言检测的启发式规则"
          },
          {
            "line": 36,
            "content": "        self.heuristic_patterns = {"
          },
          {
            "line": 37,
            "content": "            'Python': [r'def\\s+\\w+', r'import\\s+\\w+', r'from\\s+\\w+\\s+import', r'if\\s+__name__\\s*==\\s*[\"\\']__main__[\"\\']'],"
          }
        ]
      },
      {
        "line_number": 36,
        "content": "        self.heuristic_patterns = {",
        "ast_features": {
          "node_type": "Assign",
          "depth": 1,
          "children_types": [
            "Name",
            "Constant"
          ]
        },
        "context_window": [
          {
            "line": 34,
            "content": "        "
          },
          {
            "line": 35,
            "content": "        # 语言检测的启发式规则"
          },
          {
            "line": 36,
            "content": "        self.heuristic_patterns = {"
          },
          {
            "line": 37,
            "content": "            'Python': [r'def\\s+\\w+', r'import\\s+\\w+', r'from\\s+\\w+\\s+import', r'if\\s+__name__\\s*==\\s*[\"\\']__main__[\"\\']'],"
          },
          {
            "line": 38,
            "content": "            'Java': [r'public\\s+class\\s+\\w+', r'import\\s+[\\w.]+;', r'public\\s+static\\s+void\\s+main'],"
          }
        ]
      },
      {
        "line_number": 37,
        "content": "            'Python': [r'def\\s+\\w+', r'import\\s+\\w+', r'from\\s+\\w+\\s+import', r'if\\s+__name__\\s*==\\s*[\"\\']__main__[\"\\']'],",
        "ast_features": {
          "node_type": "Assign",
          "depth": 1,
          "children_types": [
            "Name",
            "Constant"
          ]
        },
        "context_window": [
          {
            "line": 35,
            "content": "        # 语言检测的启发式规则"
          },
          {
            "line": 36,
            "content": "        self.heuristic_patterns = {"
          },
          {
            "line": 37,
            "content": "            'Python': [r'def\\s+\\w+', r'import\\s+\\w+', r'from\\s+\\w+\\s+import', r'if\\s+__name__\\s*==\\s*[\"\\']__main__[\"\\']'],"
          },
          {
            "line": 38,
            "content": "            'Java': [r'public\\s+class\\s+\\w+', r'import\\s+[\\w.]+;', r'public\\s+static\\s+void\\s+main'],"
          },
          {
            "line": 39,
            "content": "            'JavaScript': [r'function\\s+\\w+', r'=>', r'var\\s+\\w+', r'let\\s+\\w+', r'const\\s+\\w+', r'require\\s*\\(']"
          }
        ]
      },
      {
        "line_number": 38,
        "content": "            'Java': [r'public\\s+class\\s+\\w+', r'import\\s+[\\w.]+;', r'public\\s+static\\s+void\\s+main'],",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 36,
            "content": "        self.heuristic_patterns = {"
          },
          {
            "line": 37,
            "content": "            'Python': [r'def\\s+\\w+', r'import\\s+\\w+', r'from\\s+\\w+\\s+import', r'if\\s+__name__\\s*==\\s*[\"\\']__main__[\"\\']'],"
          },
          {
            "line": 38,
            "content": "            'Java': [r'public\\s+class\\s+\\w+', r'import\\s+[\\w.]+;', r'public\\s+static\\s+void\\s+main'],"
          },
          {
            "line": 39,
            "content": "            'JavaScript': [r'function\\s+\\w+', r'=>', r'var\\s+\\w+', r'let\\s+\\w+', r'const\\s+\\w+', r'require\\s*\\(']"
          },
          {
            "line": 40,
            "content": "        }"
          }
        ]
      },
      {
        "line_number": 39,
        "content": "            'JavaScript': [r'function\\s+\\w+', r'=>', r'var\\s+\\w+', r'let\\s+\\w+', r'const\\s+\\w+', r'require\\s*\\(']",
        "ast_features": {
          "node_type": "Assign",
          "depth": 1,
          "children_types": [
            "Name",
            "Constant"
          ]
        },
        "context_window": [
          {
            "line": 37,
            "content": "            'Python': [r'def\\s+\\w+', r'import\\s+\\w+', r'from\\s+\\w+\\s+import', r'if\\s+__name__\\s*==\\s*[\"\\']__main__[\"\\']'],"
          },
          {
            "line": 38,
            "content": "            'Java': [r'public\\s+class\\s+\\w+', r'import\\s+[\\w.]+;', r'public\\s+static\\s+void\\s+main'],"
          },
          {
            "line": 39,
            "content": "            'JavaScript': [r'function\\s+\\w+', r'=>', r'var\\s+\\w+', r'let\\s+\\w+', r'const\\s+\\w+', r'require\\s*\\(']"
          },
          {
            "line": 40,
            "content": "        }"
          },
          {
            "line": 41,
            "content": "        "
          }
        ]
      },
      {
        "line_number": 40,
        "content": "        }",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 38,
            "content": "            'Java': [r'public\\s+class\\s+\\w+', r'import\\s+[\\w.]+;', r'public\\s+static\\s+void\\s+main'],"
          },
          {
            "line": 39,
            "content": "            'JavaScript': [r'function\\s+\\w+', r'=>', r'var\\s+\\w+', r'let\\s+\\w+', r'const\\s+\\w+', r'require\\s*\\(']"
          },
          {
            "line": 40,
            "content": "        }"
          },
          {
            "line": 41,
            "content": "        "
          },
          {
            "line": 42,
            "content": "        # 导入语句的正则表达式"
          }
        ]
      },
      {
        "line_number": 41,
        "content": "        ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 39,
            "content": "            'JavaScript': [r'function\\s+\\w+', r'=>', r'var\\s+\\w+', r'let\\s+\\w+', r'const\\s+\\w+', r'require\\s*\\(']"
          },
          {
            "line": 40,
            "content": "        }"
          },
          {
            "line": 41,
            "content": "        "
          },
          {
            "line": 42,
            "content": "        # 导入语句的正则表达式"
          },
          {
            "line": 43,
            "content": "        self.import_patterns = {"
          }
        ]
      },
      {
        "line_number": 42,
        "content": "        # 导入语句的正则表达式",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 40,
            "content": "        }"
          },
          {
            "line": 41,
            "content": "        "
          },
          {
            "line": 42,
            "content": "        # 导入语句的正则表达式"
          },
          {
            "line": 43,
            "content": "        self.import_patterns = {"
          },
          {
            "line": 44,
            "content": "            'Python': ["
          }
        ]
      },
      {
        "line_number": 43,
        "content": "        self.import_patterns = {",
        "ast_features": {
          "node_type": "Assign",
          "depth": 1,
          "children_types": [
            "Name",
            "Constant"
          ]
        },
        "context_window": [
          {
            "line": 41,
            "content": "        "
          },
          {
            "line": 42,
            "content": "        # 导入语句的正则表达式"
          },
          {
            "line": 43,
            "content": "        self.import_patterns = {"
          },
          {
            "line": 44,
            "content": "            'Python': ["
          },
          {
            "line": 45,
            "content": "                r'import\\s+([\\w.]+)',"
          }
        ]
      },
      {
        "line_number": 44,
        "content": "            'Python': [",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 42,
            "content": "        # 导入语句的正则表达式"
          },
          {
            "line": 43,
            "content": "        self.import_patterns = {"
          },
          {
            "line": 44,
            "content": "            'Python': ["
          },
          {
            "line": 45,
            "content": "                r'import\\s+([\\w.]+)',"
          },
          {
            "line": 46,
            "content": "                r'from\\s+([\\w.]+)\\s+import'"
          }
        ]
      },
      {
        "line_number": 45,
        "content": "                r'import\\s+([\\w.]+)',",
        "ast_features": {
          "node_type": "Tuple",
          "depth": 2,
          "children_types": [
            "Constant",
            "Load"
          ]
        },
        "context_window": [
          {
            "line": 43,
            "content": "        self.import_patterns = {"
          },
          {
            "line": 44,
            "content": "            'Python': ["
          },
          {
            "line": 45,
            "content": "                r'import\\s+([\\w.]+)',"
          },
          {
            "line": 46,
            "content": "                r'from\\s+([\\w.]+)\\s+import'"
          },
          {
            "line": 47,
            "content": "            ],"
          }
        ]
      },
      {
        "line_number": 46,
        "content": "                r'from\\s+([\\w.]+)\\s+import'",
        "ast_features": {
          "node_type": "Constant",
          "depth": 1,
          "children_types": []
        },
        "context_window": [
          {
            "line": 44,
            "content": "            'Python': ["
          },
          {
            "line": 45,
            "content": "                r'import\\s+([\\w.]+)',"
          },
          {
            "line": 46,
            "content": "                r'from\\s+([\\w.]+)\\s+import'"
          },
          {
            "line": 47,
            "content": "            ],"
          },
          {
            "line": 48,
            "content": "            'Java': ["
          }
        ]
      },
      {
        "line_number": 47,
        "content": "            ],",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 45,
            "content": "                r'import\\s+([\\w.]+)',"
          },
          {
            "line": 46,
            "content": "                r'from\\s+([\\w.]+)\\s+import'"
          },
          {
            "line": 47,
            "content": "            ],"
          },
          {
            "line": 48,
            "content": "            'Java': ["
          },
          {
            "line": 49,
            "content": "                r'import\\s+([\\w.]+);'"
          }
        ]
      },
      {
        "line_number": 48,
        "content": "            'Java': [",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 46,
            "content": "                r'from\\s+([\\w.]+)\\s+import'"
          },
          {
            "line": 47,
            "content": "            ],"
          },
          {
            "line": 48,
            "content": "            'Java': ["
          },
          {
            "line": 49,
            "content": "                r'import\\s+([\\w.]+);'"
          },
          {
            "line": 50,
            "content": "            ],"
          }
        ]
      },
      {
        "line_number": 49,
        "content": "                r'import\\s+([\\w.]+);'",
        "ast_features": {
          "node_type": "Constant",
          "depth": 1,
          "children_types": []
        },
        "context_window": [
          {
            "line": 47,
            "content": "            ],"
          },
          {
            "line": 48,
            "content": "            'Java': ["
          },
          {
            "line": 49,
            "content": "                r'import\\s+([\\w.]+);'"
          },
          {
            "line": 50,
            "content": "            ],"
          },
          {
            "line": 51,
            "content": "            'JavaScript': ["
          }
        ]
      },
      {
        "line_number": 50,
        "content": "            ],",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 48,
            "content": "            'Java': ["
          },
          {
            "line": 49,
            "content": "                r'import\\s+([\\w.]+);'"
          },
          {
            "line": 50,
            "content": "            ],"
          },
          {
            "line": 51,
            "content": "            'JavaScript': ["
          },
          {
            "line": 52,
            "content": "                r'import.*from\\s+[\"\\']([^\"\\']+)[\"\\']',"
          }
        ]
      },
      {
        "line_number": 51,
        "content": "            'JavaScript': [",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 49,
            "content": "                r'import\\s+([\\w.]+);'"
          },
          {
            "line": 50,
            "content": "            ],"
          },
          {
            "line": 51,
            "content": "            'JavaScript': ["
          },
          {
            "line": 52,
            "content": "                r'import.*from\\s+[\"\\']([^\"\\']+)[\"\\']',"
          },
          {
            "line": 53,
            "content": "                r'import\\s+[\"\\']([^\"\\']+)[\"\\']',"
          }
        ]
      },
      {
        "line_number": 52,
        "content": "                r'import.*from\\s+[\"\\']([^\"\\']+)[\"\\']',",
        "ast_features": {
          "node_type": "Tuple",
          "depth": 2,
          "children_types": [
            "Constant",
            "Load"
          ]
        },
        "context_window": [
          {
            "line": 50,
            "content": "            ],"
          },
          {
            "line": 51,
            "content": "            'JavaScript': ["
          },
          {
            "line": 52,
            "content": "                r'import.*from\\s+[\"\\']([^\"\\']+)[\"\\']',"
          },
          {
            "line": 53,
            "content": "                r'import\\s+[\"\\']([^\"\\']+)[\"\\']',"
          },
          {
            "line": 54,
            "content": "                r'require\\s*\\(\\s*[\"\\']([^\"\\']+)[\"\\']\\s*\\)'"
          }
        ]
      },
      {
        "line_number": 53,
        "content": "                r'import\\s+[\"\\']([^\"\\']+)[\"\\']',",
        "ast_features": {
          "node_type": "Tuple",
          "depth": 2,
          "children_types": [
            "Constant",
            "Load"
          ]
        },
        "context_window": [
          {
            "line": 51,
            "content": "            'JavaScript': ["
          },
          {
            "line": 52,
            "content": "                r'import.*from\\s+[\"\\']([^\"\\']+)[\"\\']',"
          },
          {
            "line": 53,
            "content": "                r'import\\s+[\"\\']([^\"\\']+)[\"\\']',"
          },
          {
            "line": 54,
            "content": "                r'require\\s*\\(\\s*[\"\\']([^\"\\']+)[\"\\']\\s*\\)'"
          },
          {
            "line": 55,
            "content": "            ]"
          }
        ]
      },
      {
        "line_number": 54,
        "content": "                r'require\\s*\\(\\s*[\"\\']([^\"\\']+)[\"\\']\\s*\\)'",
        "ast_features": {
          "node_type": "Constant",
          "depth": 1,
          "children_types": []
        },
        "context_window": [
          {
            "line": 52,
            "content": "                r'import.*from\\s+[\"\\']([^\"\\']+)[\"\\']',"
          },
          {
            "line": 53,
            "content": "                r'import\\s+[\"\\']([^\"\\']+)[\"\\']',"
          },
          {
            "line": 54,
            "content": "                r'require\\s*\\(\\s*[\"\\']([^\"\\']+)[\"\\']\\s*\\)'"
          },
          {
            "line": 55,
            "content": "            ]"
          },
          {
            "line": 56,
            "content": "        }"
          }
        ]
      },
      {
        "line_number": 55,
        "content": "            ]",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 53,
            "content": "                r'import\\s+[\"\\']([^\"\\']+)[\"\\']',"
          },
          {
            "line": 54,
            "content": "                r'require\\s*\\(\\s*[\"\\']([^\"\\']+)[\"\\']\\s*\\)'"
          },
          {
            "line": 55,
            "content": "            ]"
          },
          {
            "line": 56,
            "content": "        }"
          },
          {
            "line": 57,
            "content": ""
          }
        ]
      },
      {
        "line_number": 56,
        "content": "        }",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 54,
            "content": "                r'require\\s*\\(\\s*[\"\\']([^\"\\']+)[\"\\']\\s*\\)'"
          },
          {
            "line": 55,
            "content": "            ]"
          },
          {
            "line": 56,
            "content": "        }"
          },
          {
            "line": 57,
            "content": ""
          },
          {
            "line": 58,
            "content": "    def parse(self, input_data: Dict[str, str]) -> Dict[str, Any]:"
          }
        ]
      },
      {
        "line_number": 57,
        "content": "",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 55,
            "content": "            ]"
          },
          {
            "line": 56,
            "content": "        }"
          },
          {
            "line": 57,
            "content": ""
          },
          {
            "line": 58,
            "content": "    def parse(self, input_data: Dict[str, str]) -> Dict[str, Any]:"
          },
          {
            "line": 59,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 58,
        "content": "    def parse(self, input_data: Dict[str, str]) -> Dict[str, Any]:",
        "ast_features": {
          "node_type": "FunctionDef",
          "depth": 1,
          "children_types": [
            "Name",
            "arguments"
          ]
        },
        "context_window": [
          {
            "line": 56,
            "content": "        }"
          },
          {
            "line": 57,
            "content": ""
          },
          {
            "line": 58,
            "content": "    def parse(self, input_data: Dict[str, str]) -> Dict[str, Any]:"
          },
          {
            "line": 59,
            "content": "        \"\"\""
          },
          {
            "line": 60,
            "content": "        主解析方法"
          }
        ]
      },
      {
        "line_number": 59,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 57,
            "content": ""
          },
          {
            "line": 58,
            "content": "    def parse(self, input_data: Dict[str, str]) -> Dict[str, Any]:"
          },
          {
            "line": 59,
            "content": "        \"\"\""
          },
          {
            "line": 60,
            "content": "        主解析方法"
          },
          {
            "line": 61,
            "content": "        :param input_data: 输入字典 {\"file_path\": str, \"content\": str}"
          }
        ]
      },
      {
        "line_number": 60,
        "content": "        主解析方法",
        "ast_features": {
          "node_type": "Name",
          "depth": 1,
          "children_types": [
            "Load"
          ]
        },
        "context_window": [
          {
            "line": 58,
            "content": "    def parse(self, input_data: Dict[str, str]) -> Dict[str, Any]:"
          },
          {
            "line": 59,
            "content": "        \"\"\""
          },
          {
            "line": 60,
            "content": "        主解析方法"
          },
          {
            "line": 61,
            "content": "        :param input_data: 输入字典 {\"file_path\": str, \"content\": str}"
          },
          {
            "line": 62,
            "content": "        :return: 结构化解析结果字典"
          }
        ]
      },
      {
        "line_number": 61,
        "content": "        :param input_data: 输入字典 {\"file_path\": str, \"content\": str}",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 59,
            "content": "        \"\"\""
          },
          {
            "line": 60,
            "content": "        主解析方法"
          },
          {
            "line": 61,
            "content": "        :param input_data: 输入字典 {\"file_path\": str, \"content\": str}"
          },
          {
            "line": 62,
            "content": "        :return: 结构化解析结果字典"
          },
          {
            "line": 63,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 62,
        "content": "        :return: 结构化解析结果字典",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 60,
            "content": "        主解析方法"
          },
          {
            "line": 61,
            "content": "        :param input_data: 输入字典 {\"file_path\": str, \"content\": str}"
          },
          {
            "line": 62,
            "content": "        :return: 结构化解析结果字典"
          },
          {
            "line": 63,
            "content": "        \"\"\""
          },
          {
            "line": 64,
            "content": "        try:"
          }
        ]
      },
      {
        "line_number": 63,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 61,
            "content": "        :param input_data: 输入字典 {\"file_path\": str, \"content\": str}"
          },
          {
            "line": 62,
            "content": "        :return: 结构化解析结果字典"
          },
          {
            "line": 63,
            "content": "        \"\"\""
          },
          {
            "line": 64,
            "content": "        try:"
          },
          {
            "line": 65,
            "content": "            file_path = input_data.get(\"file_path\", \"\")"
          }
        ]
      },
      {
        "line_number": 64,
        "content": "        try:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 62,
            "content": "        :return: 结构化解析结果字典"
          },
          {
            "line": 63,
            "content": "        \"\"\""
          },
          {
            "line": 64,
            "content": "        try:"
          },
          {
            "line": 65,
            "content": "            file_path = input_data.get(\"file_path\", \"\")"
          },
          {
            "line": 66,
            "content": "            content = input_data.get(\"content\", \"\")"
          }
        ]
      },
      {
        "line_number": 65,
        "content": "            file_path = input_data.get(\"file_path\", \"\")",
        "ast_features": {
          "node_type": "Assign",
          "depth": 4,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 63,
            "content": "        \"\"\""
          },
          {
            "line": 64,
            "content": "        try:"
          },
          {
            "line": 65,
            "content": "            file_path = input_data.get(\"file_path\", \"\")"
          },
          {
            "line": 66,
            "content": "            content = input_data.get(\"content\", \"\")"
          },
          {
            "line": 67,
            "content": "            "
          }
        ]
      },
      {
        "line_number": 66,
        "content": "            content = input_data.get(\"content\", \"\")",
        "ast_features": {
          "node_type": "Assign",
          "depth": 4,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 64,
            "content": "        try:"
          },
          {
            "line": 65,
            "content": "            file_path = input_data.get(\"file_path\", \"\")"
          },
          {
            "line": 66,
            "content": "            content = input_data.get(\"content\", \"\")"
          },
          {
            "line": 67,
            "content": "            "
          },
          {
            "line": 68,
            "content": "            if not content:"
          }
        ]
      },
      {
        "line_number": 67,
        "content": "            ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 65,
            "content": "            file_path = input_data.get(\"file_path\", \"\")"
          },
          {
            "line": 66,
            "content": "            content = input_data.get(\"content\", \"\")"
          },
          {
            "line": 67,
            "content": "            "
          },
          {
            "line": 68,
            "content": "            if not content:"
          },
          {
            "line": 69,
            "content": "                raise ValueError(\"Content cannot be empty\")"
          }
        ]
      },
      {
        "line_number": 68,
        "content": "            if not content:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 66,
            "content": "            content = input_data.get(\"content\", \"\")"
          },
          {
            "line": 67,
            "content": "            "
          },
          {
            "line": 68,
            "content": "            if not content:"
          },
          {
            "line": 69,
            "content": "                raise ValueError(\"Content cannot be empty\")"
          },
          {
            "line": 70,
            "content": "            "
          }
        ]
      },
      {
        "line_number": 69,
        "content": "                raise ValueError(\"Content cannot be empty\")",
        "ast_features": {
          "node_type": "Raise",
          "depth": 3,
          "children_types": [
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 67,
            "content": "            "
          },
          {
            "line": 68,
            "content": "            if not content:"
          },
          {
            "line": 69,
            "content": "                raise ValueError(\"Content cannot be empty\")"
          },
          {
            "line": 70,
            "content": "            "
          },
          {
            "line": 71,
            "content": "            # 1. 识别编程语言"
          }
        ]
      },
      {
        "line_number": 70,
        "content": "            ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 68,
            "content": "            if not content:"
          },
          {
            "line": 69,
            "content": "                raise ValueError(\"Content cannot be empty\")"
          },
          {
            "line": 70,
            "content": "            "
          },
          {
            "line": 71,
            "content": "            # 1. 识别编程语言"
          },
          {
            "line": 72,
            "content": "            language = self._detect_language(content, file_path)"
          }
        ]
      },
      {
        "line_number": 71,
        "content": "            # 1. 识别编程语言",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 69,
            "content": "                raise ValueError(\"Content cannot be empty\")"
          },
          {
            "line": 70,
            "content": "            "
          },
          {
            "line": 71,
            "content": "            # 1. 识别编程语言"
          },
          {
            "line": 72,
            "content": "            language = self._detect_language(content, file_path)"
          },
          {
            "line": 73,
            "content": "            "
          }
        ]
      },
      {
        "line_number": 72,
        "content": "            language = self._detect_language(content, file_path)",
        "ast_features": {
          "node_type": "Assign",
          "depth": 4,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 70,
            "content": "            "
          },
          {
            "line": 71,
            "content": "            # 1. 识别编程语言"
          },
          {
            "line": 72,
            "content": "            language = self._detect_language(content, file_path)"
          },
          {
            "line": 73,
            "content": "            "
          },
          {
            "line": 74,
            "content": "            # 2. 解析文件级上下文"
          }
        ]
      },
      {
        "line_number": 73,
        "content": "            ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 71,
            "content": "            # 1. 识别编程语言"
          },
          {
            "line": 72,
            "content": "            language = self._detect_language(content, file_path)"
          },
          {
            "line": 73,
            "content": "            "
          },
          {
            "line": 74,
            "content": "            # 2. 解析文件级上下文"
          },
          {
            "line": 75,
            "content": "            file_context = self._parse_file_context(content, language, file_path)"
          }
        ]
      },
      {
        "line_number": 74,
        "content": "            # 2. 解析文件级上下文",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 72,
            "content": "            language = self._detect_language(content, file_path)"
          },
          {
            "line": 73,
            "content": "            "
          },
          {
            "line": 74,
            "content": "            # 2. 解析文件级上下文"
          },
          {
            "line": 75,
            "content": "            file_context = self._parse_file_context(content, language, file_path)"
          },
          {
            "line": 76,
            "content": "            "
          }
        ]
      },
      {
        "line_number": 75,
        "content": "            file_context = self._parse_file_context(content, language, file_path)",
        "ast_features": {
          "node_type": "Assign",
          "depth": 4,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 73,
            "content": "            "
          },
          {
            "line": 74,
            "content": "            # 2. 解析文件级上下文"
          },
          {
            "line": 75,
            "content": "            file_context = self._parse_file_context(content, language, file_path)"
          },
          {
            "line": 76,
            "content": "            "
          },
          {
            "line": 77,
            "content": "            # 3. 解析行级信息"
          }
        ]
      },
      {
        "line_number": 76,
        "content": "            ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 74,
            "content": "            # 2. 解析文件级上下文"
          },
          {
            "line": 75,
            "content": "            file_context = self._parse_file_context(content, language, file_path)"
          },
          {
            "line": 76,
            "content": "            "
          },
          {
            "line": 77,
            "content": "            # 3. 解析行级信息"
          },
          {
            "line": 78,
            "content": "            lines_data = self._parse_lines(content, language)"
          }
        ]
      },
      {
        "line_number": 77,
        "content": "            # 3. 解析行级信息",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 75,
            "content": "            file_context = self._parse_file_context(content, language, file_path)"
          },
          {
            "line": 76,
            "content": "            "
          },
          {
            "line": 77,
            "content": "            # 3. 解析行级信息"
          },
          {
            "line": 78,
            "content": "            lines_data = self._parse_lines(content, language)"
          },
          {
            "line": 79,
            "content": "            "
          }
        ]
      },
      {
        "line_number": 78,
        "content": "            lines_data = self._parse_lines(content, language)",
        "ast_features": {
          "node_type": "Assign",
          "depth": 4,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 76,
            "content": "            "
          },
          {
            "line": 77,
            "content": "            # 3. 解析行级信息"
          },
          {
            "line": 78,
            "content": "            lines_data = self._parse_lines(content, language)"
          },
          {
            "line": 79,
            "content": "            "
          },
          {
            "line": 80,
            "content": "            # 4. 组装结果"
          }
        ]
      },
      {
        "line_number": 79,
        "content": "            ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 77,
            "content": "            # 3. 解析行级信息"
          },
          {
            "line": 78,
            "content": "            lines_data = self._parse_lines(content, language)"
          },
          {
            "line": 79,
            "content": "            "
          },
          {
            "line": 80,
            "content": "            # 4. 组装结果"
          },
          {
            "line": 81,
            "content": "            result = {"
          }
        ]
      },
      {
        "line_number": 80,
        "content": "            # 4. 组装结果",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 78,
            "content": "            lines_data = self._parse_lines(content, language)"
          },
          {
            "line": 79,
            "content": "            "
          },
          {
            "line": 80,
            "content": "            # 4. 组装结果"
          },
          {
            "line": 81,
            "content": "            result = {"
          },
          {
            "line": 82,
            "content": "                \"language\": language,"
          }
        ]
      },
      {
        "line_number": 81,
        "content": "            result = {",
        "ast_features": {
          "node_type": "Assign",
          "depth": 1,
          "children_types": [
            "Name",
            "Constant"
          ]
        },
        "context_window": [
          {
            "line": 79,
            "content": "            "
          },
          {
            "line": 80,
            "content": "            # 4. 组装结果"
          },
          {
            "line": 81,
            "content": "            result = {"
          },
          {
            "line": 82,
            "content": "                \"language\": language,"
          },
          {
            "line": 83,
            "content": "                \"file_context\": file_context,"
          }
        ]
      },
      {
        "line_number": 82,
        "content": "                \"language\": language,",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 80,
            "content": "            # 4. 组装结果"
          },
          {
            "line": 81,
            "content": "            result = {"
          },
          {
            "line": 82,
            "content": "                \"language\": language,"
          },
          {
            "line": 83,
            "content": "                \"file_context\": file_context,"
          },
          {
            "line": 84,
            "content": "                \"lines\": lines_data"
          }
        ]
      },
      {
        "line_number": 83,
        "content": "                \"file_context\": file_context,",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 81,
            "content": "            result = {"
          },
          {
            "line": 82,
            "content": "                \"language\": language,"
          },
          {
            "line": 83,
            "content": "                \"file_context\": file_context,"
          },
          {
            "line": 84,
            "content": "                \"lines\": lines_data"
          },
          {
            "line": 85,
            "content": "            }"
          }
        ]
      },
      {
        "line_number": 84,
        "content": "                \"lines\": lines_data",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 82,
            "content": "                \"language\": language,"
          },
          {
            "line": 83,
            "content": "                \"file_context\": file_context,"
          },
          {
            "line": 84,
            "content": "                \"lines\": lines_data"
          },
          {
            "line": 85,
            "content": "            }"
          },
          {
            "line": 86,
            "content": "            "
          }
        ]
      },
      {
        "line_number": 85,
        "content": "            }",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 83,
            "content": "                \"file_context\": file_context,"
          },
          {
            "line": 84,
            "content": "                \"lines\": lines_data"
          },
          {
            "line": 85,
            "content": "            }"
          },
          {
            "line": 86,
            "content": "            "
          },
          {
            "line": 87,
            "content": "            return result"
          }
        ]
      },
      {
        "line_number": 86,
        "content": "            ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 84,
            "content": "                \"lines\": lines_data"
          },
          {
            "line": 85,
            "content": "            }"
          },
          {
            "line": 86,
            "content": "            "
          },
          {
            "line": 87,
            "content": "            return result"
          },
          {
            "line": 88,
            "content": "            "
          }
        ]
      },
      {
        "line_number": 87,
        "content": "            return result",
        "ast_features": {
          "node_type": "Return",
          "depth": 2,
          "children_types": [
            "Name"
          ]
        },
        "context_window": [
          {
            "line": 85,
            "content": "            }"
          },
          {
            "line": 86,
            "content": "            "
          },
          {
            "line": 87,
            "content": "            return result"
          },
          {
            "line": 88,
            "content": "            "
          },
          {
            "line": 89,
            "content": "        except Exception as e:"
          }
        ]
      },
      {
        "line_number": 88,
        "content": "            ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 86,
            "content": "            "
          },
          {
            "line": 87,
            "content": "            return result"
          },
          {
            "line": 88,
            "content": "            "
          },
          {
            "line": 89,
            "content": "        except Exception as e:"
          },
          {
            "line": 90,
            "content": "            logger.error(f\"Error parsing file: {str(e)}\")"
          }
        ]
      },
      {
        "line_number": 89,
        "content": "        except Exception as e:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 87,
            "content": "            return result"
          },
          {
            "line": 88,
            "content": "            "
          },
          {
            "line": 89,
            "content": "        except Exception as e:"
          },
          {
            "line": 90,
            "content": "            logger.error(f\"Error parsing file: {str(e)}\")"
          },
          {
            "line": 91,
            "content": "            # 返回最基本的解析结果"
          }
        ]
      },
      {
        "line_number": 90,
        "content": "            logger.error(f\"Error parsing file: {str(e)}\")",
        "ast_features": {
          "node_type": "Call",
          "depth": 5,
          "children_types": [
            "Attribute",
            "JoinedStr"
          ]
        },
        "context_window": [
          {
            "line": 88,
            "content": "            "
          },
          {
            "line": 89,
            "content": "        except Exception as e:"
          },
          {
            "line": 90,
            "content": "            logger.error(f\"Error parsing file: {str(e)}\")"
          },
          {
            "line": 91,
            "content": "            # 返回最基本的解析结果"
          },
          {
            "line": 92,
            "content": "            return self._create_fallback_result(input_data)"
          }
        ]
      },
      {
        "line_number": 91,
        "content": "            # 返回最基本的解析结果",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 89,
            "content": "        except Exception as e:"
          },
          {
            "line": 90,
            "content": "            logger.error(f\"Error parsing file: {str(e)}\")"
          },
          {
            "line": 91,
            "content": "            # 返回最基本的解析结果"
          },
          {
            "line": 92,
            "content": "            return self._create_fallback_result(input_data)"
          },
          {
            "line": 93,
            "content": ""
          }
        ]
      },
      {
        "line_number": 92,
        "content": "            return self._create_fallback_result(input_data)",
        "ast_features": {
          "node_type": "Return",
          "depth": 4,
          "children_types": [
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 90,
            "content": "            logger.error(f\"Error parsing file: {str(e)}\")"
          },
          {
            "line": 91,
            "content": "            # 返回最基本的解析结果"
          },
          {
            "line": 92,
            "content": "            return self._create_fallback_result(input_data)"
          },
          {
            "line": 93,
            "content": ""
          },
          {
            "line": 94,
            "content": "    def _detect_language(self, content: str, file_path: Optional[str] = None) -> str:"
          }
        ]
      },
      {
        "line_number": 93,
        "content": "",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 91,
            "content": "            # 返回最基本的解析结果"
          },
          {
            "line": 92,
            "content": "            return self._create_fallback_result(input_data)"
          },
          {
            "line": 93,
            "content": ""
          },
          {
            "line": 94,
            "content": "    def _detect_language(self, content: str, file_path: Optional[str] = None) -> str:"
          },
          {
            "line": 95,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 94,
        "content": "    def _detect_language(self, content: str, file_path: Optional[str] = None) -> str:",
        "ast_features": {
          "node_type": "FunctionDef",
          "depth": 1,
          "children_types": [
            "Name",
            "arguments"
          ]
        },
        "context_window": [
          {
            "line": 92,
            "content": "            return self._create_fallback_result(input_data)"
          },
          {
            "line": 93,
            "content": ""
          },
          {
            "line": 94,
            "content": "    def _detect_language(self, content: str, file_path: Optional[str] = None) -> str:"
          },
          {
            "line": 95,
            "content": "        \"\"\""
          },
          {
            "line": 96,
            "content": "        检测编程语言"
          }
        ]
      },
      {
        "line_number": 95,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 93,
            "content": ""
          },
          {
            "line": 94,
            "content": "    def _detect_language(self, content: str, file_path: Optional[str] = None) -> str:"
          },
          {
            "line": 95,
            "content": "        \"\"\""
          },
          {
            "line": 96,
            "content": "        检测编程语言"
          },
          {
            "line": 97,
            "content": "        :param content: 文件内容"
          }
        ]
      },
      {
        "line_number": 96,
        "content": "        检测编程语言",
        "ast_features": {
          "node_type": "Name",
          "depth": 1,
          "children_types": [
            "Load"
          ]
        },
        "context_window": [
          {
            "line": 94,
            "content": "    def _detect_language(self, content: str, file_path: Optional[str] = None) -> str:"
          },
          {
            "line": 95,
            "content": "        \"\"\""
          },
          {
            "line": 96,
            "content": "        检测编程语言"
          },
          {
            "line": 97,
            "content": "        :param content: 文件内容"
          },
          {
            "line": 98,
            "content": "        :param file_path: 文件路径"
          }
        ]
      },
      {
        "line_number": 97,
        "content": "        :param content: 文件内容",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 95,
            "content": "        \"\"\""
          },
          {
            "line": 96,
            "content": "        检测编程语言"
          },
          {
            "line": 97,
            "content": "        :param content: 文件内容"
          },
          {
            "line": 98,
            "content": "        :param file_path: 文件路径"
          },
          {
            "line": 99,
            "content": "        :return: 检测到的语言名称"
          }
        ]
      },
      {
        "line_number": 98,
        "content": "        :param file_path: 文件路径",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 96,
            "content": "        检测编程语言"
          },
          {
            "line": 97,
            "content": "        :param content: 文件内容"
          },
          {
            "line": 98,
            "content": "        :param file_path: 文件路径"
          },
          {
            "line": 99,
            "content": "        :return: 检测到的语言名称"
          },
          {
            "line": 100,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 99,
        "content": "        :return: 检测到的语言名称",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 97,
            "content": "        :param content: 文件内容"
          },
          {
            "line": 98,
            "content": "        :param file_path: 文件路径"
          },
          {
            "line": 99,
            "content": "        :return: 检测到的语言名称"
          },
          {
            "line": 100,
            "content": "        \"\"\""
          },
          {
            "line": 101,
            "content": "        # 优先根据文件扩展名判断"
          }
        ]
      },
      {
        "line_number": 100,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 98,
            "content": "        :param file_path: 文件路径"
          },
          {
            "line": 99,
            "content": "        :return: 检测到的语言名称"
          },
          {
            "line": 100,
            "content": "        \"\"\""
          },
          {
            "line": 101,
            "content": "        # 优先根据文件扩展名判断"
          },
          {
            "line": 102,
            "content": "        if file_path:"
          }
        ]
      },
      {
        "line_number": 101,
        "content": "        # 优先根据文件扩展名判断",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 99,
            "content": "        :return: 检测到的语言名称"
          },
          {
            "line": 100,
            "content": "        \"\"\""
          },
          {
            "line": 101,
            "content": "        # 优先根据文件扩展名判断"
          },
          {
            "line": 102,
            "content": "        if file_path:"
          },
          {
            "line": 103,
            "content": "            _, ext = os.path.splitext(file_path)"
          }
        ]
      },
      {
        "line_number": 102,
        "content": "        if file_path:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 100,
            "content": "        \"\"\""
          },
          {
            "line": 101,
            "content": "        # 优先根据文件扩展名判断"
          },
          {
            "line": 102,
            "content": "        if file_path:"
          },
          {
            "line": 103,
            "content": "            _, ext = os.path.splitext(file_path)"
          },
          {
            "line": 104,
            "content": "            if ext.lower() in self.extension_map:"
          }
        ]
      },
      {
        "line_number": 103,
        "content": "            _, ext = os.path.splitext(file_path)",
        "ast_features": {
          "node_type": "Assign",
          "depth": 5,
          "children_types": [
            "Tuple",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 101,
            "content": "        # 优先根据文件扩展名判断"
          },
          {
            "line": 102,
            "content": "        if file_path:"
          },
          {
            "line": 103,
            "content": "            _, ext = os.path.splitext(file_path)"
          },
          {
            "line": 104,
            "content": "            if ext.lower() in self.extension_map:"
          },
          {
            "line": 105,
            "content": "                return self.extension_map[ext.lower()]"
          }
        ]
      },
      {
        "line_number": 104,
        "content": "            if ext.lower() in self.extension_map:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 102,
            "content": "        if file_path:"
          },
          {
            "line": 103,
            "content": "            _, ext = os.path.splitext(file_path)"
          },
          {
            "line": 104,
            "content": "            if ext.lower() in self.extension_map:"
          },
          {
            "line": 105,
            "content": "                return self.extension_map[ext.lower()]"
          },
          {
            "line": 106,
            "content": "        "
          }
        ]
      },
      {
        "line_number": 105,
        "content": "                return self.extension_map[ext.lower()]",
        "ast_features": {
          "node_type": "Return",
          "depth": 5,
          "children_types": [
            "Subscript"
          ]
        },
        "context_window": [
          {
            "line": 103,
            "content": "            _, ext = os.path.splitext(file_path)"
          },
          {
            "line": 104,
            "content": "            if ext.lower() in self.extension_map:"
          },
          {
            "line": 105,
            "content": "                return self.extension_map[ext.lower()]"
          },
          {
            "line": 106,
            "content": "        "
          },
          {
            "line": 107,
            "content": "        # 使用启发式规则"
          }
        ]
      },
      {
        "line_number": 106,
        "content": "        ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 104,
            "content": "            if ext.lower() in self.extension_map:"
          },
          {
            "line": 105,
            "content": "                return self.extension_map[ext.lower()]"
          },
          {
            "line": 106,
            "content": "        "
          },
          {
            "line": 107,
            "content": "        # 使用启发式规则"
          },
          {
            "line": 108,
            "content": "        language_scores = {}"
          }
        ]
      },
      {
        "line_number": 107,
        "content": "        # 使用启发式规则",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 105,
            "content": "                return self.extension_map[ext.lower()]"
          },
          {
            "line": 106,
            "content": "        "
          },
          {
            "line": 107,
            "content": "        # 使用启发式规则"
          },
          {
            "line": 108,
            "content": "        language_scores = {}"
          },
          {
            "line": 109,
            "content": "        for lang, patterns in self.heuristic_patterns.items():"
          }
        ]
      },
      {
        "line_number": 108,
        "content": "        language_scores = {}",
        "ast_features": {
          "node_type": "Assign",
          "depth": 2,
          "children_types": [
            "Name",
            "Dict"
          ]
        },
        "context_window": [
          {
            "line": 106,
            "content": "        "
          },
          {
            "line": 107,
            "content": "        # 使用启发式规则"
          },
          {
            "line": 108,
            "content": "        language_scores = {}"
          },
          {
            "line": 109,
            "content": "        for lang, patterns in self.heuristic_patterns.items():"
          },
          {
            "line": 110,
            "content": "            score = 0"
          }
        ]
      },
      {
        "line_number": 109,
        "content": "        for lang, patterns in self.heuristic_patterns.items():",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 107,
            "content": "        # 使用启发式规则"
          },
          {
            "line": 108,
            "content": "        language_scores = {}"
          },
          {
            "line": 109,
            "content": "        for lang, patterns in self.heuristic_patterns.items():"
          },
          {
            "line": 110,
            "content": "            score = 0"
          },
          {
            "line": 111,
            "content": "            for pattern in patterns:"
          }
        ]
      },
      {
        "line_number": 110,
        "content": "            score = 0",
        "ast_features": {
          "node_type": "Assign",
          "depth": 2,
          "children_types": [
            "Name",
            "Constant"
          ]
        },
        "context_window": [
          {
            "line": 108,
            "content": "        language_scores = {}"
          },
          {
            "line": 109,
            "content": "        for lang, patterns in self.heuristic_patterns.items():"
          },
          {
            "line": 110,
            "content": "            score = 0"
          },
          {
            "line": 111,
            "content": "            for pattern in patterns:"
          },
          {
            "line": 112,
            "content": "                matches = len(re.findall(pattern, content, re.IGNORECASE))"
          }
        ]
      },
      {
        "line_number": 111,
        "content": "            for pattern in patterns:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 109,
            "content": "        for lang, patterns in self.heuristic_patterns.items():"
          },
          {
            "line": 110,
            "content": "            score = 0"
          },
          {
            "line": 111,
            "content": "            for pattern in patterns:"
          },
          {
            "line": 112,
            "content": "                matches = len(re.findall(pattern, content, re.IGNORECASE))"
          },
          {
            "line": 113,
            "content": "                score += matches"
          }
        ]
      },
      {
        "line_number": 112,
        "content": "                matches = len(re.findall(pattern, content, re.IGNORECASE))",
        "ast_features": {
          "node_type": "Assign",
          "depth": 5,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 110,
            "content": "            score = 0"
          },
          {
            "line": 111,
            "content": "            for pattern in patterns:"
          },
          {
            "line": 112,
            "content": "                matches = len(re.findall(pattern, content, re.IGNORECASE))"
          },
          {
            "line": 113,
            "content": "                score += matches"
          },
          {
            "line": 114,
            "content": "            language_scores[lang] = score"
          }
        ]
      },
      {
        "line_number": 113,
        "content": "                score += matches",
        "ast_features": {
          "node_type": "AugAssign",
          "depth": 2,
          "children_types": [
            "Name",
            "Add",
            "Name"
          ]
        },
        "context_window": [
          {
            "line": 111,
            "content": "            for pattern in patterns:"
          },
          {
            "line": 112,
            "content": "                matches = len(re.findall(pattern, content, re.IGNORECASE))"
          },
          {
            "line": 113,
            "content": "                score += matches"
          },
          {
            "line": 114,
            "content": "            language_scores[lang] = score"
          },
          {
            "line": 115,
            "content": "        "
          }
        ]
      },
      {
        "line_number": 114,
        "content": "            language_scores[lang] = score",
        "ast_features": {
          "node_type": "Assign",
          "depth": 3,
          "children_types": [
            "Subscript",
            "Name"
          ]
        },
        "context_window": [
          {
            "line": 112,
            "content": "                matches = len(re.findall(pattern, content, re.IGNORECASE))"
          },
          {
            "line": 113,
            "content": "                score += matches"
          },
          {
            "line": 114,
            "content": "            language_scores[lang] = score"
          },
          {
            "line": 115,
            "content": "        "
          },
          {
            "line": 116,
            "content": "        # 返回得分最高的语言"
          }
        ]
      },
      {
        "line_number": 115,
        "content": "        ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 113,
            "content": "                score += matches"
          },
          {
            "line": 114,
            "content": "            language_scores[lang] = score"
          },
          {
            "line": 115,
            "content": "        "
          },
          {
            "line": 116,
            "content": "        # 返回得分最高的语言"
          },
          {
            "line": 117,
            "content": "        if language_scores and max(language_scores.values()) > 0:"
          }
        ]
      },
      {
        "line_number": 116,
        "content": "        # 返回得分最高的语言",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 114,
            "content": "            language_scores[lang] = score"
          },
          {
            "line": 115,
            "content": "        "
          },
          {
            "line": 116,
            "content": "        # 返回得分最高的语言"
          },
          {
            "line": 117,
            "content": "        if language_scores and max(language_scores.values()) > 0:"
          },
          {
            "line": 118,
            "content": "            return max(language_scores.keys(), key=lambda x: language_scores[x])"
          }
        ]
      },
      {
        "line_number": 117,
        "content": "        if language_scores and max(language_scores.values()) > 0:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 115,
            "content": "        "
          },
          {
            "line": 116,
            "content": "        # 返回得分最高的语言"
          },
          {
            "line": 117,
            "content": "        if language_scores and max(language_scores.values()) > 0:"
          },
          {
            "line": 118,
            "content": "            return max(language_scores.keys(), key=lambda x: language_scores[x])"
          },
          {
            "line": 119,
            "content": "        "
          }
        ]
      },
      {
        "line_number": 118,
        "content": "            return max(language_scores.keys(), key=lambda x: language_scores[x])",
        "ast_features": {
          "node_type": "Return",
          "depth": 6,
          "children_types": [
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 116,
            "content": "        # 返回得分最高的语言"
          },
          {
            "line": 117,
            "content": "        if language_scores and max(language_scores.values()) > 0:"
          },
          {
            "line": 118,
            "content": "            return max(language_scores.keys(), key=lambda x: language_scores[x])"
          },
          {
            "line": 119,
            "content": "        "
          },
          {
            "line": 120,
            "content": "        return \"unknown\""
          }
        ]
      },
      {
        "line_number": 119,
        "content": "        ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 117,
            "content": "        if language_scores and max(language_scores.values()) > 0:"
          },
          {
            "line": 118,
            "content": "            return max(language_scores.keys(), key=lambda x: language_scores[x])"
          },
          {
            "line": 119,
            "content": "        "
          },
          {
            "line": 120,
            "content": "        return \"unknown\""
          },
          {
            "line": 121,
            "content": ""
          }
        ]
      },
      {
        "line_number": 120,
        "content": "        return \"unknown\"",
        "ast_features": {
          "node_type": "Return",
          "depth": 2,
          "children_types": [
            "Constant"
          ]
        },
        "context_window": [
          {
            "line": 118,
            "content": "            return max(language_scores.keys(), key=lambda x: language_scores[x])"
          },
          {
            "line": 119,
            "content": "        "
          },
          {
            "line": 120,
            "content": "        return \"unknown\""
          },
          {
            "line": 121,
            "content": ""
          },
          {
            "line": 122,
            "content": "    def _parse_file_context(self, content: str, language: str, file_path: str = \"\") -> Dict[str, Any]:"
          }
        ]
      },
      {
        "line_number": 121,
        "content": "",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 119,
            "content": "        "
          },
          {
            "line": 120,
            "content": "        return \"unknown\""
          },
          {
            "line": 121,
            "content": ""
          },
          {
            "line": 122,
            "content": "    def _parse_file_context(self, content: str, language: str, file_path: str = \"\") -> Dict[str, Any]:"
          },
          {
            "line": 123,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 122,
        "content": "    def _parse_file_context(self, content: str, language: str, file_path: str = \"\") -> Dict[str, Any]:",
        "ast_features": {
          "node_type": "FunctionDef",
          "depth": 1,
          "children_types": [
            "Name",
            "arguments"
          ]
        },
        "context_window": [
          {
            "line": 120,
            "content": "        return \"unknown\""
          },
          {
            "line": 121,
            "content": ""
          },
          {
            "line": 122,
            "content": "    def _parse_file_context(self, content: str, language: str, file_path: str = \"\") -> Dict[str, Any]:"
          },
          {
            "line": 123,
            "content": "        \"\"\""
          },
          {
            "line": 124,
            "content": "        解析文件级上下文"
          }
        ]
      },
      {
        "line_number": 123,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 121,
            "content": ""
          },
          {
            "line": 122,
            "content": "    def _parse_file_context(self, content: str, language: str, file_path: str = \"\") -> Dict[str, Any]:"
          },
          {
            "line": 123,
            "content": "        \"\"\""
          },
          {
            "line": 124,
            "content": "        解析文件级上下文"
          },
          {
            "line": 125,
            "content": "        :param content: 文件内容"
          }
        ]
      },
      {
        "line_number": 124,
        "content": "        解析文件级上下文",
        "ast_features": {
          "node_type": "Name",
          "depth": 1,
          "children_types": [
            "Load"
          ]
        },
        "context_window": [
          {
            "line": 122,
            "content": "    def _parse_file_context(self, content: str, language: str, file_path: str = \"\") -> Dict[str, Any]:"
          },
          {
            "line": 123,
            "content": "        \"\"\""
          },
          {
            "line": 124,
            "content": "        解析文件级上下文"
          },
          {
            "line": 125,
            "content": "        :param content: 文件内容"
          },
          {
            "line": 126,
            "content": "        :param language: 编程语言"
          }
        ]
      },
      {
        "line_number": 125,
        "content": "        :param content: 文件内容",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 123,
            "content": "        \"\"\""
          },
          {
            "line": 124,
            "content": "        解析文件级上下文"
          },
          {
            "line": 125,
            "content": "        :param content: 文件内容"
          },
          {
            "line": 126,
            "content": "        :param language: 编程语言"
          },
          {
            "line": 127,
            "content": "        :param file_path: 文件路径"
          }
        ]
      },
      {
        "line_number": 126,
        "content": "        :param language: 编程语言",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 124,
            "content": "        解析文件级上下文"
          },
          {
            "line": 125,
            "content": "        :param content: 文件内容"
          },
          {
            "line": 126,
            "content": "        :param language: 编程语言"
          },
          {
            "line": 127,
            "content": "        :param file_path: 文件路径"
          },
          {
            "line": 128,
            "content": "        :return: 文件上下文字典"
          }
        ]
      },
      {
        "line_number": 127,
        "content": "        :param file_path: 文件路径",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 125,
            "content": "        :param content: 文件内容"
          },
          {
            "line": 126,
            "content": "        :param language: 编程语言"
          },
          {
            "line": 127,
            "content": "        :param file_path: 文件路径"
          },
          {
            "line": 128,
            "content": "        :return: 文件上下文字典"
          },
          {
            "line": 129,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 128,
        "content": "        :return: 文件上下文字典",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 126,
            "content": "        :param language: 编程语言"
          },
          {
            "line": 127,
            "content": "        :param file_path: 文件路径"
          },
          {
            "line": 128,
            "content": "        :return: 文件上下文字典"
          },
          {
            "line": 129,
            "content": "        \"\"\""
          },
          {
            "line": 130,
            "content": "        # 提取文件名"
          }
        ]
      },
      {
        "line_number": 129,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 127,
            "content": "        :param file_path: 文件路径"
          },
          {
            "line": 128,
            "content": "        :return: 文件上下文字典"
          },
          {
            "line": 129,
            "content": "        \"\"\""
          },
          {
            "line": 130,
            "content": "        # 提取文件名"
          },
          {
            "line": 131,
            "content": "        file_name = os.path.basename(file_path) if file_path else \"unknown\""
          }
        ]
      },
      {
        "line_number": 130,
        "content": "        # 提取文件名",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 128,
            "content": "        :return: 文件上下文字典"
          },
          {
            "line": 129,
            "content": "        \"\"\""
          },
          {
            "line": 130,
            "content": "        # 提取文件名"
          },
          {
            "line": 131,
            "content": "        file_name = os.path.basename(file_path) if file_path else \"unknown\""
          },
          {
            "line": 132,
            "content": "        "
          }
        ]
      },
      {
        "line_number": 131,
        "content": "        file_name = os.path.basename(file_path) if file_path else \"unknown\"",
        "ast_features": {
          "node_type": "Assign",
          "depth": 6,
          "children_types": [
            "Name",
            "IfExp"
          ]
        },
        "context_window": [
          {
            "line": 129,
            "content": "        \"\"\""
          },
          {
            "line": 130,
            "content": "        # 提取文件名"
          },
          {
            "line": 131,
            "content": "        file_name = os.path.basename(file_path) if file_path else \"unknown\""
          },
          {
            "line": 132,
            "content": "        "
          },
          {
            "line": 133,
            "content": "        # 提取导入列表"
          }
        ]
      },
      {
        "line_number": 132,
        "content": "        ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 130,
            "content": "        # 提取文件名"
          },
          {
            "line": 131,
            "content": "        file_name = os.path.basename(file_path) if file_path else \"unknown\""
          },
          {
            "line": 132,
            "content": "        "
          },
          {
            "line": 133,
            "content": "        # 提取导入列表"
          },
          {
            "line": 134,
            "content": "        imports = self._extract_imports(content, language)"
          }
        ]
      },
      {
        "line_number": 133,
        "content": "        # 提取导入列表",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 131,
            "content": "        file_name = os.path.basename(file_path) if file_path else \"unknown\""
          },
          {
            "line": 132,
            "content": "        "
          },
          {
            "line": 133,
            "content": "        # 提取导入列表"
          },
          {
            "line": 134,
            "content": "        imports = self._extract_imports(content, language)"
          },
          {
            "line": 135,
            "content": "        "
          }
        ]
      },
      {
        "line_number": 134,
        "content": "        imports = self._extract_imports(content, language)",
        "ast_features": {
          "node_type": "Assign",
          "depth": 4,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 132,
            "content": "        "
          },
          {
            "line": 133,
            "content": "        # 提取导入列表"
          },
          {
            "line": 134,
            "content": "        imports = self._extract_imports(content, language)"
          },
          {
            "line": 135,
            "content": "        "
          },
          {
            "line": 136,
            "content": "        # 提取类名和函数名（简化版本，取第一个找到的）"
          }
        ]
      },
      {
        "line_number": 135,
        "content": "        ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 133,
            "content": "        # 提取导入列表"
          },
          {
            "line": 134,
            "content": "        imports = self._extract_imports(content, language)"
          },
          {
            "line": 135,
            "content": "        "
          },
          {
            "line": 136,
            "content": "        # 提取类名和函数名（简化版本，取第一个找到的）"
          },
          {
            "line": 137,
            "content": "        class_name = self._extract_class_name(content, language)"
          }
        ]
      },
      {
        "line_number": 136,
        "content": "        # 提取类名和函数名（简化版本，取第一个找到的）",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 134,
            "content": "        imports = self._extract_imports(content, language)"
          },
          {
            "line": 135,
            "content": "        "
          },
          {
            "line": 136,
            "content": "        # 提取类名和函数名（简化版本，取第一个找到的）"
          },
          {
            "line": 137,
            "content": "        class_name = self._extract_class_name(content, language)"
          },
          {
            "line": 138,
            "content": "        function_name = self._extract_function_name(content, language)"
          }
        ]
      },
      {
        "line_number": 137,
        "content": "        class_name = self._extract_class_name(content, language)",
        "ast_features": {
          "node_type": "Assign",
          "depth": 4,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 135,
            "content": "        "
          },
          {
            "line": 136,
            "content": "        # 提取类名和函数名（简化版本，取第一个找到的）"
          },
          {
            "line": 137,
            "content": "        class_name = self._extract_class_name(content, language)"
          },
          {
            "line": 138,
            "content": "        function_name = self._extract_function_name(content, language)"
          },
          {
            "line": 139,
            "content": "        "
          }
        ]
      },
      {
        "line_number": 138,
        "content": "        function_name = self._extract_function_name(content, language)",
        "ast_features": {
          "node_type": "Assign",
          "depth": 4,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 136,
            "content": "        # 提取类名和函数名（简化版本，取第一个找到的）"
          },
          {
            "line": 137,
            "content": "        class_name = self._extract_class_name(content, language)"
          },
          {
            "line": 138,
            "content": "        function_name = self._extract_function_name(content, language)"
          },
          {
            "line": 139,
            "content": "        "
          },
          {
            "line": 140,
            "content": "        return {"
          }
        ]
      },
      {
        "line_number": 139,
        "content": "        ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 137,
            "content": "        class_name = self._extract_class_name(content, language)"
          },
          {
            "line": 138,
            "content": "        function_name = self._extract_function_name(content, language)"
          },
          {
            "line": 139,
            "content": "        "
          },
          {
            "line": 140,
            "content": "        return {"
          },
          {
            "line": 141,
            "content": "            \"file_name\": file_name,"
          }
        ]
      },
      {
        "line_number": 140,
        "content": "        return {",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 138,
            "content": "        function_name = self._extract_function_name(content, language)"
          },
          {
            "line": 139,
            "content": "        "
          },
          {
            "line": 140,
            "content": "        return {"
          },
          {
            "line": 141,
            "content": "            \"file_name\": file_name,"
          },
          {
            "line": 142,
            "content": "            \"imports\": imports,"
          }
        ]
      },
      {
        "line_number": 141,
        "content": "            \"file_name\": file_name,",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 139,
            "content": "        "
          },
          {
            "line": 140,
            "content": "        return {"
          },
          {
            "line": 141,
            "content": "            \"file_name\": file_name,"
          },
          {
            "line": 142,
            "content": "            \"imports\": imports,"
          },
          {
            "line": 143,
            "content": "            \"class_name\": class_name,"
          }
        ]
      },
      {
        "line_number": 142,
        "content": "            \"imports\": imports,",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 140,
            "content": "        return {"
          },
          {
            "line": 141,
            "content": "            \"file_name\": file_name,"
          },
          {
            "line": 142,
            "content": "            \"imports\": imports,"
          },
          {
            "line": 143,
            "content": "            \"class_name\": class_name,"
          },
          {
            "line": 144,
            "content": "            \"function_name\": function_name"
          }
        ]
      },
      {
        "line_number": 143,
        "content": "            \"class_name\": class_name,",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 141,
            "content": "            \"file_name\": file_name,"
          },
          {
            "line": 142,
            "content": "            \"imports\": imports,"
          },
          {
            "line": 143,
            "content": "            \"class_name\": class_name,"
          },
          {
            "line": 144,
            "content": "            \"function_name\": function_name"
          },
          {
            "line": 145,
            "content": "        }"
          }
        ]
      },
      {
        "line_number": 144,
        "content": "            \"function_name\": function_name",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 142,
            "content": "            \"imports\": imports,"
          },
          {
            "line": 143,
            "content": "            \"class_name\": class_name,"
          },
          {
            "line": 144,
            "content": "            \"function_name\": function_name"
          },
          {
            "line": 145,
            "content": "        }"
          },
          {
            "line": 146,
            "content": ""
          }
        ]
      },
      {
        "line_number": 145,
        "content": "        }",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 143,
            "content": "            \"class_name\": class_name,"
          },
          {
            "line": 144,
            "content": "            \"function_name\": function_name"
          },
          {
            "line": 145,
            "content": "        }"
          },
          {
            "line": 146,
            "content": ""
          },
          {
            "line": 147,
            "content": "    def _extract_imports(self, content: str, language: str) -> List[str]:"
          }
        ]
      },
      {
        "line_number": 146,
        "content": "",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 144,
            "content": "            \"function_name\": function_name"
          },
          {
            "line": 145,
            "content": "        }"
          },
          {
            "line": 146,
            "content": ""
          },
          {
            "line": 147,
            "content": "    def _extract_imports(self, content: str, language: str) -> List[str]:"
          },
          {
            "line": 148,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 147,
        "content": "    def _extract_imports(self, content: str, language: str) -> List[str]:",
        "ast_features": {
          "node_type": "FunctionDef",
          "depth": 1,
          "children_types": [
            "Name",
            "arguments"
          ]
        },
        "context_window": [
          {
            "line": 145,
            "content": "        }"
          },
          {
            "line": 146,
            "content": ""
          },
          {
            "line": 147,
            "content": "    def _extract_imports(self, content: str, language: str) -> List[str]:"
          },
          {
            "line": 148,
            "content": "        \"\"\""
          },
          {
            "line": 149,
            "content": "        提取导入列表"
          }
        ]
      },
      {
        "line_number": 148,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 146,
            "content": ""
          },
          {
            "line": 147,
            "content": "    def _extract_imports(self, content: str, language: str) -> List[str]:"
          },
          {
            "line": 148,
            "content": "        \"\"\""
          },
          {
            "line": 149,
            "content": "        提取导入列表"
          },
          {
            "line": 150,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 149,
        "content": "        提取导入列表",
        "ast_features": {
          "node_type": "Name",
          "depth": 1,
          "children_types": [
            "Load"
          ]
        },
        "context_window": [
          {
            "line": 147,
            "content": "    def _extract_imports(self, content: str, language: str) -> List[str]:"
          },
          {
            "line": 148,
            "content": "        \"\"\""
          },
          {
            "line": 149,
            "content": "        提取导入列表"
          },
          {
            "line": 150,
            "content": "        \"\"\""
          },
          {
            "line": 151,
            "content": "        imports = []"
          }
        ]
      },
      {
        "line_number": 150,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 148,
            "content": "        \"\"\""
          },
          {
            "line": 149,
            "content": "        提取导入列表"
          },
          {
            "line": 150,
            "content": "        \"\"\""
          },
          {
            "line": 151,
            "content": "        imports = []"
          },
          {
            "line": 152,
            "content": "        if language in self.import_patterns:"
          }
        ]
      },
      {
        "line_number": 151,
        "content": "        imports = []",
        "ast_features": {
          "node_type": "Assign",
          "depth": 2,
          "children_types": [
            "Name",
            "List"
          ]
        },
        "context_window": [
          {
            "line": 149,
            "content": "        提取导入列表"
          },
          {
            "line": 150,
            "content": "        \"\"\""
          },
          {
            "line": 151,
            "content": "        imports = []"
          },
          {
            "line": 152,
            "content": "        if language in self.import_patterns:"
          },
          {
            "line": 153,
            "content": "            for pattern in self.import_patterns[language]:"
          }
        ]
      },
      {
        "line_number": 152,
        "content": "        if language in self.import_patterns:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 150,
            "content": "        \"\"\""
          },
          {
            "line": 151,
            "content": "        imports = []"
          },
          {
            "line": 152,
            "content": "        if language in self.import_patterns:"
          },
          {
            "line": 153,
            "content": "            for pattern in self.import_patterns[language]:"
          },
          {
            "line": 154,
            "content": "                matches = re.findall(pattern, content)"
          }
        ]
      },
      {
        "line_number": 153,
        "content": "            for pattern in self.import_patterns[language]:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 151,
            "content": "        imports = []"
          },
          {
            "line": 152,
            "content": "        if language in self.import_patterns:"
          },
          {
            "line": 153,
            "content": "            for pattern in self.import_patterns[language]:"
          },
          {
            "line": 154,
            "content": "                matches = re.findall(pattern, content)"
          },
          {
            "line": 155,
            "content": "                imports.extend(matches)"
          }
        ]
      },
      {
        "line_number": 154,
        "content": "                matches = re.findall(pattern, content)",
        "ast_features": {
          "node_type": "Assign",
          "depth": 4,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 152,
            "content": "        if language in self.import_patterns:"
          },
          {
            "line": 153,
            "content": "            for pattern in self.import_patterns[language]:"
          },
          {
            "line": 154,
            "content": "                matches = re.findall(pattern, content)"
          },
          {
            "line": 155,
            "content": "                imports.extend(matches)"
          },
          {
            "line": 156,
            "content": "        return list(set(imports))  # 去重"
          }
        ]
      },
      {
        "line_number": 155,
        "content": "                imports.extend(matches)",
        "ast_features": {
          "node_type": "Call",
          "depth": 3,
          "children_types": [
            "Attribute",
            "Name"
          ]
        },
        "context_window": [
          {
            "line": 153,
            "content": "            for pattern in self.import_patterns[language]:"
          },
          {
            "line": 154,
            "content": "                matches = re.findall(pattern, content)"
          },
          {
            "line": 155,
            "content": "                imports.extend(matches)"
          },
          {
            "line": 156,
            "content": "        return list(set(imports))  # 去重"
          },
          {
            "line": 157,
            "content": ""
          }
        ]
      },
      {
        "line_number": 156,
        "content": "        return list(set(imports))  # 去重",
        "ast_features": {
          "node_type": "Return",
          "depth": 4,
          "children_types": [
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 154,
            "content": "                matches = re.findall(pattern, content)"
          },
          {
            "line": 155,
            "content": "                imports.extend(matches)"
          },
          {
            "line": 156,
            "content": "        return list(set(imports))  # 去重"
          },
          {
            "line": 157,
            "content": ""
          },
          {
            "line": 158,
            "content": "    def _extract_class_name(self, content: str, language: str) -> Optional[str]:"
          }
        ]
      },
      {
        "line_number": 157,
        "content": "",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 155,
            "content": "                imports.extend(matches)"
          },
          {
            "line": 156,
            "content": "        return list(set(imports))  # 去重"
          },
          {
            "line": 157,
            "content": ""
          },
          {
            "line": 158,
            "content": "    def _extract_class_name(self, content: str, language: str) -> Optional[str]:"
          },
          {
            "line": 159,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 158,
        "content": "    def _extract_class_name(self, content: str, language: str) -> Optional[str]:",
        "ast_features": {
          "node_type": "FunctionDef",
          "depth": 1,
          "children_types": [
            "Name",
            "arguments"
          ]
        },
        "context_window": [
          {
            "line": 156,
            "content": "        return list(set(imports))  # 去重"
          },
          {
            "line": 157,
            "content": ""
          },
          {
            "line": 158,
            "content": "    def _extract_class_name(self, content: str, language: str) -> Optional[str]:"
          },
          {
            "line": 159,
            "content": "        \"\"\""
          },
          {
            "line": 160,
            "content": "        提取类名"
          }
        ]
      },
      {
        "line_number": 159,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 157,
            "content": ""
          },
          {
            "line": 158,
            "content": "    def _extract_class_name(self, content: str, language: str) -> Optional[str]:"
          },
          {
            "line": 159,
            "content": "        \"\"\""
          },
          {
            "line": 160,
            "content": "        提取类名"
          },
          {
            "line": 161,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 160,
        "content": "        提取类名",
        "ast_features": {
          "node_type": "Name",
          "depth": 1,
          "children_types": [
            "Load"
          ]
        },
        "context_window": [
          {
            "line": 158,
            "content": "    def _extract_class_name(self, content: str, language: str) -> Optional[str]:"
          },
          {
            "line": 159,
            "content": "        \"\"\""
          },
          {
            "line": 160,
            "content": "        提取类名"
          },
          {
            "line": 161,
            "content": "        \"\"\""
          },
          {
            "line": 162,
            "content": "        patterns = {"
          }
        ]
      },
      {
        "line_number": 161,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 159,
            "content": "        \"\"\""
          },
          {
            "line": 160,
            "content": "        提取类名"
          },
          {
            "line": 161,
            "content": "        \"\"\""
          },
          {
            "line": 162,
            "content": "        patterns = {"
          },
          {
            "line": 163,
            "content": "            'Python': r'class\\s+(\\w+)',"
          }
        ]
      },
      {
        "line_number": 162,
        "content": "        patterns = {",
        "ast_features": {
          "node_type": "Assign",
          "depth": 1,
          "children_types": [
            "Name",
            "Constant"
          ]
        },
        "context_window": [
          {
            "line": 160,
            "content": "        提取类名"
          },
          {
            "line": 161,
            "content": "        \"\"\""
          },
          {
            "line": 162,
            "content": "        patterns = {"
          },
          {
            "line": 163,
            "content": "            'Python': r'class\\s+(\\w+)',"
          },
          {
            "line": 164,
            "content": "            'Java': r'(?:public\\s+)?class\\s+(\\w+)',"
          }
        ]
      },
      {
        "line_number": 163,
        "content": "            'Python': r'class\\s+(\\w+)',",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 161,
            "content": "        \"\"\""
          },
          {
            "line": 162,
            "content": "        patterns = {"
          },
          {
            "line": 163,
            "content": "            'Python': r'class\\s+(\\w+)',"
          },
          {
            "line": 164,
            "content": "            'Java': r'(?:public\\s+)?class\\s+(\\w+)',"
          },
          {
            "line": 165,
            "content": "            'JavaScript': r'class\\s+(\\w+)'"
          }
        ]
      },
      {
        "line_number": 164,
        "content": "            'Java': r'(?:public\\s+)?class\\s+(\\w+)',",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 162,
            "content": "        patterns = {"
          },
          {
            "line": 163,
            "content": "            'Python': r'class\\s+(\\w+)',"
          },
          {
            "line": 164,
            "content": "            'Java': r'(?:public\\s+)?class\\s+(\\w+)',"
          },
          {
            "line": 165,
            "content": "            'JavaScript': r'class\\s+(\\w+)'"
          },
          {
            "line": 166,
            "content": "        }"
          }
        ]
      },
      {
        "line_number": 165,
        "content": "            'JavaScript': r'class\\s+(\\w+)'",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 163,
            "content": "            'Python': r'class\\s+(\\w+)',"
          },
          {
            "line": 164,
            "content": "            'Java': r'(?:public\\s+)?class\\s+(\\w+)',"
          },
          {
            "line": 165,
            "content": "            'JavaScript': r'class\\s+(\\w+)'"
          },
          {
            "line": 166,
            "content": "        }"
          },
          {
            "line": 167,
            "content": "        "
          }
        ]
      },
      {
        "line_number": 166,
        "content": "        }",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 164,
            "content": "            'Java': r'(?:public\\s+)?class\\s+(\\w+)',"
          },
          {
            "line": 165,
            "content": "            'JavaScript': r'class\\s+(\\w+)'"
          },
          {
            "line": 166,
            "content": "        }"
          },
          {
            "line": 167,
            "content": "        "
          },
          {
            "line": 168,
            "content": "        if language in patterns:"
          }
        ]
      },
      {
        "line_number": 167,
        "content": "        ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 165,
            "content": "            'JavaScript': r'class\\s+(\\w+)'"
          },
          {
            "line": 166,
            "content": "        }"
          },
          {
            "line": 167,
            "content": "        "
          },
          {
            "line": 168,
            "content": "        if language in patterns:"
          },
          {
            "line": 169,
            "content": "            match = re.search(patterns[language], content)"
          }
        ]
      },
      {
        "line_number": 168,
        "content": "        if language in patterns:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 166,
            "content": "        }"
          },
          {
            "line": 167,
            "content": "        "
          },
          {
            "line": 168,
            "content": "        if language in patterns:"
          },
          {
            "line": 169,
            "content": "            match = re.search(patterns[language], content)"
          },
          {
            "line": 170,
            "content": "            if match:"
          }
        ]
      },
      {
        "line_number": 169,
        "content": "            match = re.search(patterns[language], content)",
        "ast_features": {
          "node_type": "Assign",
          "depth": 4,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 167,
            "content": "        "
          },
          {
            "line": 168,
            "content": "        if language in patterns:"
          },
          {
            "line": 169,
            "content": "            match = re.search(patterns[language], content)"
          },
          {
            "line": 170,
            "content": "            if match:"
          },
          {
            "line": 171,
            "content": "                return match.group(1)"
          }
        ]
      },
      {
        "line_number": 170,
        "content": "            if match:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 168,
            "content": "        if language in patterns:"
          },
          {
            "line": 169,
            "content": "            match = re.search(patterns[language], content)"
          },
          {
            "line": 170,
            "content": "            if match:"
          },
          {
            "line": 171,
            "content": "                return match.group(1)"
          },
          {
            "line": 172,
            "content": "        return None"
          }
        ]
      },
      {
        "line_number": 171,
        "content": "                return match.group(1)",
        "ast_features": {
          "node_type": "Return",
          "depth": 4,
          "children_types": [
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 169,
            "content": "            match = re.search(patterns[language], content)"
          },
          {
            "line": 170,
            "content": "            if match:"
          },
          {
            "line": 171,
            "content": "                return match.group(1)"
          },
          {
            "line": 172,
            "content": "        return None"
          },
          {
            "line": 173,
            "content": ""
          }
        ]
      },
      {
        "line_number": 172,
        "content": "        return None",
        "ast_features": {
          "node_type": "Return",
          "depth": 2,
          "children_types": [
            "Constant"
          ]
        },
        "context_window": [
          {
            "line": 170,
            "content": "            if match:"
          },
          {
            "line": 171,
            "content": "                return match.group(1)"
          },
          {
            "line": 172,
            "content": "        return None"
          },
          {
            "line": 173,
            "content": ""
          },
          {
            "line": 174,
            "content": "    def _extract_function_name(self, content: str, language: str) -> Optional[str]:"
          }
        ]
      },
      {
        "line_number": 173,
        "content": "",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 171,
            "content": "                return match.group(1)"
          },
          {
            "line": 172,
            "content": "        return None"
          },
          {
            "line": 173,
            "content": ""
          },
          {
            "line": 174,
            "content": "    def _extract_function_name(self, content: str, language: str) -> Optional[str]:"
          },
          {
            "line": 175,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 174,
        "content": "    def _extract_function_name(self, content: str, language: str) -> Optional[str]:",
        "ast_features": {
          "node_type": "FunctionDef",
          "depth": 1,
          "children_types": [
            "Name",
            "arguments"
          ]
        },
        "context_window": [
          {
            "line": 172,
            "content": "        return None"
          },
          {
            "line": 173,
            "content": ""
          },
          {
            "line": 174,
            "content": "    def _extract_function_name(self, content: str, language: str) -> Optional[str]:"
          },
          {
            "line": 175,
            "content": "        \"\"\""
          },
          {
            "line": 176,
            "content": "        提取函数名"
          }
        ]
      },
      {
        "line_number": 175,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 173,
            "content": ""
          },
          {
            "line": 174,
            "content": "    def _extract_function_name(self, content: str, language: str) -> Optional[str]:"
          },
          {
            "line": 175,
            "content": "        \"\"\""
          },
          {
            "line": 176,
            "content": "        提取函数名"
          },
          {
            "line": 177,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 176,
        "content": "        提取函数名",
        "ast_features": {
          "node_type": "Name",
          "depth": 1,
          "children_types": [
            "Load"
          ]
        },
        "context_window": [
          {
            "line": 174,
            "content": "    def _extract_function_name(self, content: str, language: str) -> Optional[str]:"
          },
          {
            "line": 175,
            "content": "        \"\"\""
          },
          {
            "line": 176,
            "content": "        提取函数名"
          },
          {
            "line": 177,
            "content": "        \"\"\""
          },
          {
            "line": 178,
            "content": "        patterns = {"
          }
        ]
      },
      {
        "line_number": 177,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 175,
            "content": "        \"\"\""
          },
          {
            "line": 176,
            "content": "        提取函数名"
          },
          {
            "line": 177,
            "content": "        \"\"\""
          },
          {
            "line": 178,
            "content": "        patterns = {"
          },
          {
            "line": 179,
            "content": "            'Python': r'def\\s+(\\w+)',"
          }
        ]
      },
      {
        "line_number": 178,
        "content": "        patterns = {",
        "ast_features": {
          "node_type": "Assign",
          "depth": 1,
          "children_types": [
            "Name",
            "Constant"
          ]
        },
        "context_window": [
          {
            "line": 176,
            "content": "        提取函数名"
          },
          {
            "line": 177,
            "content": "        \"\"\""
          },
          {
            "line": 178,
            "content": "        patterns = {"
          },
          {
            "line": 179,
            "content": "            'Python': r'def\\s+(\\w+)',"
          },
          {
            "line": 180,
            "content": "            'Java': r'(?:public|private|protected)?\\s*(?:static\\s+)?(?:\\w+\\s+)+(\\w+)\\s*\\(',"
          }
        ]
      },
      {
        "line_number": 179,
        "content": "            'Python': r'def\\s+(\\w+)',",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 177,
            "content": "        \"\"\""
          },
          {
            "line": 178,
            "content": "        patterns = {"
          },
          {
            "line": 179,
            "content": "            'Python': r'def\\s+(\\w+)',"
          },
          {
            "line": 180,
            "content": "            'Java': r'(?:public|private|protected)?\\s*(?:static\\s+)?(?:\\w+\\s+)+(\\w+)\\s*\\(',"
          },
          {
            "line": 181,
            "content": "            'JavaScript': r'function\\s+(\\w+)|(\\w+)\\s*=\\s*(?:function|\\([^)]*\\)\\s*=>)'"
          }
        ]
      },
      {
        "line_number": 180,
        "content": "            'Java': r'(?:public|private|protected)?\\s*(?:static\\s+)?(?:\\w+\\s+)+(\\w+)\\s*\\(',",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 178,
            "content": "        patterns = {"
          },
          {
            "line": 179,
            "content": "            'Python': r'def\\s+(\\w+)',"
          },
          {
            "line": 180,
            "content": "            'Java': r'(?:public|private|protected)?\\s*(?:static\\s+)?(?:\\w+\\s+)+(\\w+)\\s*\\(',"
          },
          {
            "line": 181,
            "content": "            'JavaScript': r'function\\s+(\\w+)|(\\w+)\\s*=\\s*(?:function|\\([^)]*\\)\\s*=>)'"
          },
          {
            "line": 182,
            "content": "        }"
          }
        ]
      },
      {
        "line_number": 181,
        "content": "            'JavaScript': r'function\\s+(\\w+)|(\\w+)\\s*=\\s*(?:function|\\([^)]*\\)\\s*=>)'",
        "ast_features": {
          "node_type": "Assign",
          "depth": 1,
          "children_types": [
            "Name",
            "Constant"
          ]
        },
        "context_window": [
          {
            "line": 179,
            "content": "            'Python': r'def\\s+(\\w+)',"
          },
          {
            "line": 180,
            "content": "            'Java': r'(?:public|private|protected)?\\s*(?:static\\s+)?(?:\\w+\\s+)+(\\w+)\\s*\\(',"
          },
          {
            "line": 181,
            "content": "            'JavaScript': r'function\\s+(\\w+)|(\\w+)\\s*=\\s*(?:function|\\([^)]*\\)\\s*=>)'"
          },
          {
            "line": 182,
            "content": "        }"
          },
          {
            "line": 183,
            "content": "        "
          }
        ]
      },
      {
        "line_number": 182,
        "content": "        }",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 180,
            "content": "            'Java': r'(?:public|private|protected)?\\s*(?:static\\s+)?(?:\\w+\\s+)+(\\w+)\\s*\\(',"
          },
          {
            "line": 181,
            "content": "            'JavaScript': r'function\\s+(\\w+)|(\\w+)\\s*=\\s*(?:function|\\([^)]*\\)\\s*=>)'"
          },
          {
            "line": 182,
            "content": "        }"
          },
          {
            "line": 183,
            "content": "        "
          },
          {
            "line": 184,
            "content": "        if language in patterns:"
          }
        ]
      },
      {
        "line_number": 183,
        "content": "        ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 181,
            "content": "            'JavaScript': r'function\\s+(\\w+)|(\\w+)\\s*=\\s*(?:function|\\([^)]*\\)\\s*=>)'"
          },
          {
            "line": 182,
            "content": "        }"
          },
          {
            "line": 183,
            "content": "        "
          },
          {
            "line": 184,
            "content": "        if language in patterns:"
          },
          {
            "line": 185,
            "content": "            match = re.search(patterns[language], content)"
          }
        ]
      },
      {
        "line_number": 184,
        "content": "        if language in patterns:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 182,
            "content": "        }"
          },
          {
            "line": 183,
            "content": "        "
          },
          {
            "line": 184,
            "content": "        if language in patterns:"
          },
          {
            "line": 185,
            "content": "            match = re.search(patterns[language], content)"
          },
          {
            "line": 186,
            "content": "            if match:"
          }
        ]
      },
      {
        "line_number": 185,
        "content": "            match = re.search(patterns[language], content)",
        "ast_features": {
          "node_type": "Assign",
          "depth": 4,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 183,
            "content": "        "
          },
          {
            "line": 184,
            "content": "        if language in patterns:"
          },
          {
            "line": 185,
            "content": "            match = re.search(patterns[language], content)"
          },
          {
            "line": 186,
            "content": "            if match:"
          },
          {
            "line": 187,
            "content": "                return match.group(1) or match.group(2)"
          }
        ]
      },
      {
        "line_number": 186,
        "content": "            if match:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 184,
            "content": "        if language in patterns:"
          },
          {
            "line": 185,
            "content": "            match = re.search(patterns[language], content)"
          },
          {
            "line": 186,
            "content": "            if match:"
          },
          {
            "line": 187,
            "content": "                return match.group(1) or match.group(2)"
          },
          {
            "line": 188,
            "content": "        return None"
          }
        ]
      },
      {
        "line_number": 187,
        "content": "                return match.group(1) or match.group(2)",
        "ast_features": {
          "node_type": "Return",
          "depth": 5,
          "children_types": [
            "BoolOp"
          ]
        },
        "context_window": [
          {
            "line": 185,
            "content": "            match = re.search(patterns[language], content)"
          },
          {
            "line": 186,
            "content": "            if match:"
          },
          {
            "line": 187,
            "content": "                return match.group(1) or match.group(2)"
          },
          {
            "line": 188,
            "content": "        return None"
          },
          {
            "line": 189,
            "content": ""
          }
        ]
      },
      {
        "line_number": 188,
        "content": "        return None",
        "ast_features": {
          "node_type": "Return",
          "depth": 2,
          "children_types": [
            "Constant"
          ]
        },
        "context_window": [
          {
            "line": 186,
            "content": "            if match:"
          },
          {
            "line": 187,
            "content": "                return match.group(1) or match.group(2)"
          },
          {
            "line": 188,
            "content": "        return None"
          },
          {
            "line": 189,
            "content": ""
          },
          {
            "line": 190,
            "content": "    def _parse_lines(self, content: str, language: str) -> List[Dict[str, Any]]:"
          }
        ]
      },
      {
        "line_number": 189,
        "content": "",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 187,
            "content": "                return match.group(1) or match.group(2)"
          },
          {
            "line": 188,
            "content": "        return None"
          },
          {
            "line": 189,
            "content": ""
          },
          {
            "line": 190,
            "content": "    def _parse_lines(self, content: str, language: str) -> List[Dict[str, Any]]:"
          },
          {
            "line": 191,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 190,
        "content": "    def _parse_lines(self, content: str, language: str) -> List[Dict[str, Any]]:",
        "ast_features": {
          "node_type": "FunctionDef",
          "depth": 1,
          "children_types": [
            "Name",
            "arguments"
          ]
        },
        "context_window": [
          {
            "line": 188,
            "content": "        return None"
          },
          {
            "line": 189,
            "content": ""
          },
          {
            "line": 190,
            "content": "    def _parse_lines(self, content: str, language: str) -> List[Dict[str, Any]]:"
          },
          {
            "line": 191,
            "content": "        \"\"\""
          },
          {
            "line": 192,
            "content": "        解析行级信息"
          }
        ]
      },
      {
        "line_number": 191,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 189,
            "content": ""
          },
          {
            "line": 190,
            "content": "    def _parse_lines(self, content: str, language: str) -> List[Dict[str, Any]]:"
          },
          {
            "line": 191,
            "content": "        \"\"\""
          },
          {
            "line": 192,
            "content": "        解析行级信息"
          },
          {
            "line": 193,
            "content": "        :param content: 文件内容"
          }
        ]
      },
      {
        "line_number": 192,
        "content": "        解析行级信息",
        "ast_features": {
          "node_type": "Name",
          "depth": 1,
          "children_types": [
            "Load"
          ]
        },
        "context_window": [
          {
            "line": 190,
            "content": "    def _parse_lines(self, content: str, language: str) -> List[Dict[str, Any]]:"
          },
          {
            "line": 191,
            "content": "        \"\"\""
          },
          {
            "line": 192,
            "content": "        解析行级信息"
          },
          {
            "line": 193,
            "content": "        :param content: 文件内容"
          },
          {
            "line": 194,
            "content": "        :param language: 编程语言"
          }
        ]
      },
      {
        "line_number": 193,
        "content": "        :param content: 文件内容",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 191,
            "content": "        \"\"\""
          },
          {
            "line": 192,
            "content": "        解析行级信息"
          },
          {
            "line": 193,
            "content": "        :param content: 文件内容"
          },
          {
            "line": 194,
            "content": "        :param language: 编程语言"
          },
          {
            "line": 195,
            "content": "        :return: 行级数据列表"
          }
        ]
      },
      {
        "line_number": 194,
        "content": "        :param language: 编程语言",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 192,
            "content": "        解析行级信息"
          },
          {
            "line": 193,
            "content": "        :param content: 文件内容"
          },
          {
            "line": 194,
            "content": "        :param language: 编程语言"
          },
          {
            "line": 195,
            "content": "        :return: 行级数据列表"
          },
          {
            "line": 196,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 195,
        "content": "        :return: 行级数据列表",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 193,
            "content": "        :param content: 文件内容"
          },
          {
            "line": 194,
            "content": "        :param language: 编程语言"
          },
          {
            "line": 195,
            "content": "        :return: 行级数据列表"
          },
          {
            "line": 196,
            "content": "        \"\"\""
          },
          {
            "line": 197,
            "content": "        lines = content.split('\\n')"
          }
        ]
      },
      {
        "line_number": 196,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 194,
            "content": "        :param language: 编程语言"
          },
          {
            "line": 195,
            "content": "        :return: 行级数据列表"
          },
          {
            "line": 196,
            "content": "        \"\"\""
          },
          {
            "line": 197,
            "content": "        lines = content.split('\\n')"
          },
          {
            "line": 198,
            "content": "        lines_data = []"
          }
        ]
      },
      {
        "line_number": 197,
        "content": "        lines = content.split('\\n')",
        "ast_features": {
          "node_type": "Assign",
          "depth": 4,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 195,
            "content": "        :return: 行级数据列表"
          },
          {
            "line": 196,
            "content": "        \"\"\""
          },
          {
            "line": 197,
            "content": "        lines = content.split('\\n')"
          },
          {
            "line": 198,
            "content": "        lines_data = []"
          },
          {
            "line": 199,
            "content": "        "
          }
        ]
      },
      {
        "line_number": 198,
        "content": "        lines_data = []",
        "ast_features": {
          "node_type": "Assign",
          "depth": 2,
          "children_types": [
            "Name",
            "List"
          ]
        },
        "context_window": [
          {
            "line": 196,
            "content": "        \"\"\""
          },
          {
            "line": 197,
            "content": "        lines = content.split('\\n')"
          },
          {
            "line": 198,
            "content": "        lines_data = []"
          },
          {
            "line": 199,
            "content": "        "
          },
          {
            "line": 200,
            "content": "        for i, line in enumerate(lines):"
          }
        ]
      },
      {
        "line_number": 199,
        "content": "        ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 197,
            "content": "        lines = content.split('\\n')"
          },
          {
            "line": 198,
            "content": "        lines_data = []"
          },
          {
            "line": 199,
            "content": "        "
          },
          {
            "line": 200,
            "content": "        for i, line in enumerate(lines):"
          },
          {
            "line": 201,
            "content": "            line_number = i + 1"
          }
        ]
      },
      {
        "line_number": 200,
        "content": "        for i, line in enumerate(lines):",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 198,
            "content": "        lines_data = []"
          },
          {
            "line": 199,
            "content": "        "
          },
          {
            "line": 200,
            "content": "        for i, line in enumerate(lines):"
          },
          {
            "line": 201,
            "content": "            line_number = i + 1"
          },
          {
            "line": 202,
            "content": "            "
          }
        ]
      },
      {
        "line_number": 201,
        "content": "            line_number = i + 1",
        "ast_features": {
          "node_type": "Assign",
          "depth": 3,
          "children_types": [
            "Name",
            "BinOp"
          ]
        },
        "context_window": [
          {
            "line": 199,
            "content": "        "
          },
          {
            "line": 200,
            "content": "        for i, line in enumerate(lines):"
          },
          {
            "line": 201,
            "content": "            line_number = i + 1"
          },
          {
            "line": 202,
            "content": "            "
          },
          {
            "line": 203,
            "content": "            # 提取AST特征"
          }
        ]
      },
      {
        "line_number": 202,
        "content": "            ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 200,
            "content": "        for i, line in enumerate(lines):"
          },
          {
            "line": 201,
            "content": "            line_number = i + 1"
          },
          {
            "line": 202,
            "content": "            "
          },
          {
            "line": 203,
            "content": "            # 提取AST特征"
          },
          {
            "line": 204,
            "content": "            ast_features = self._extract_ast_features(line, line_number, content, language)"
          }
        ]
      },
      {
        "line_number": 203,
        "content": "            # 提取AST特征",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 201,
            "content": "            line_number = i + 1"
          },
          {
            "line": 202,
            "content": "            "
          },
          {
            "line": 203,
            "content": "            # 提取AST特征"
          },
          {
            "line": 204,
            "content": "            ast_features = self._extract_ast_features(line, line_number, content, language)"
          },
          {
            "line": 205,
            "content": "            "
          }
        ]
      },
      {
        "line_number": 204,
        "content": "            ast_features = self._extract_ast_features(line, line_number, content, language)",
        "ast_features": {
          "node_type": "Assign",
          "depth": 4,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 202,
            "content": "            "
          },
          {
            "line": 203,
            "content": "            # 提取AST特征"
          },
          {
            "line": 204,
            "content": "            ast_features = self._extract_ast_features(line, line_number, content, language)"
          },
          {
            "line": 205,
            "content": "            "
          },
          {
            "line": 206,
            "content": "            # 构建上下文窗口"
          }
        ]
      },
      {
        "line_number": 205,
        "content": "            ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 203,
            "content": "            # 提取AST特征"
          },
          {
            "line": 204,
            "content": "            ast_features = self._extract_ast_features(line, line_number, content, language)"
          },
          {
            "line": 205,
            "content": "            "
          },
          {
            "line": 206,
            "content": "            # 构建上下文窗口"
          },
          {
            "line": 207,
            "content": "            context_window = self._build_context_window(lines, i)"
          }
        ]
      },
      {
        "line_number": 206,
        "content": "            # 构建上下文窗口",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 204,
            "content": "            ast_features = self._extract_ast_features(line, line_number, content, language)"
          },
          {
            "line": 205,
            "content": "            "
          },
          {
            "line": 206,
            "content": "            # 构建上下文窗口"
          },
          {
            "line": 207,
            "content": "            context_window = self._build_context_window(lines, i)"
          },
          {
            "line": 208,
            "content": "            "
          }
        ]
      },
      {
        "line_number": 207,
        "content": "            context_window = self._build_context_window(lines, i)",
        "ast_features": {
          "node_type": "Assign",
          "depth": 4,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 205,
            "content": "            "
          },
          {
            "line": 206,
            "content": "            # 构建上下文窗口"
          },
          {
            "line": 207,
            "content": "            context_window = self._build_context_window(lines, i)"
          },
          {
            "line": 208,
            "content": "            "
          },
          {
            "line": 209,
            "content": "            line_data = {"
          }
        ]
      },
      {
        "line_number": 208,
        "content": "            ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 206,
            "content": "            # 构建上下文窗口"
          },
          {
            "line": 207,
            "content": "            context_window = self._build_context_window(lines, i)"
          },
          {
            "line": 208,
            "content": "            "
          },
          {
            "line": 209,
            "content": "            line_data = {"
          },
          {
            "line": 210,
            "content": "                \"line_number\": line_number,"
          }
        ]
      },
      {
        "line_number": 209,
        "content": "            line_data = {",
        "ast_features": {
          "node_type": "Assign",
          "depth": 1,
          "children_types": [
            "Name",
            "Constant"
          ]
        },
        "context_window": [
          {
            "line": 207,
            "content": "            context_window = self._build_context_window(lines, i)"
          },
          {
            "line": 208,
            "content": "            "
          },
          {
            "line": 209,
            "content": "            line_data = {"
          },
          {
            "line": 210,
            "content": "                \"line_number\": line_number,"
          },
          {
            "line": 211,
            "content": "                \"content\": line,"
          }
        ]
      },
      {
        "line_number": 210,
        "content": "                \"line_number\": line_number,",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 208,
            "content": "            "
          },
          {
            "line": 209,
            "content": "            line_data = {"
          },
          {
            "line": 210,
            "content": "                \"line_number\": line_number,"
          },
          {
            "line": 211,
            "content": "                \"content\": line,"
          },
          {
            "line": 212,
            "content": "                \"ast_features\": ast_features,"
          }
        ]
      },
      {
        "line_number": 211,
        "content": "                \"content\": line,",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 209,
            "content": "            line_data = {"
          },
          {
            "line": 210,
            "content": "                \"line_number\": line_number,"
          },
          {
            "line": 211,
            "content": "                \"content\": line,"
          },
          {
            "line": 212,
            "content": "                \"ast_features\": ast_features,"
          },
          {
            "line": 213,
            "content": "                \"context_window\": context_window"
          }
        ]
      },
      {
        "line_number": 212,
        "content": "                \"ast_features\": ast_features,",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 210,
            "content": "                \"line_number\": line_number,"
          },
          {
            "line": 211,
            "content": "                \"content\": line,"
          },
          {
            "line": 212,
            "content": "                \"ast_features\": ast_features,"
          },
          {
            "line": 213,
            "content": "                \"context_window\": context_window"
          },
          {
            "line": 214,
            "content": "            }"
          }
        ]
      },
      {
        "line_number": 213,
        "content": "                \"context_window\": context_window",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 211,
            "content": "                \"content\": line,"
          },
          {
            "line": 212,
            "content": "                \"ast_features\": ast_features,"
          },
          {
            "line": 213,
            "content": "                \"context_window\": context_window"
          },
          {
            "line": 214,
            "content": "            }"
          },
          {
            "line": 215,
            "content": "            "
          }
        ]
      },
      {
        "line_number": 214,
        "content": "            }",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 212,
            "content": "                \"ast_features\": ast_features,"
          },
          {
            "line": 213,
            "content": "                \"context_window\": context_window"
          },
          {
            "line": 214,
            "content": "            }"
          },
          {
            "line": 215,
            "content": "            "
          },
          {
            "line": 216,
            "content": "            lines_data.append(line_data)"
          }
        ]
      },
      {
        "line_number": 215,
        "content": "            ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 213,
            "content": "                \"context_window\": context_window"
          },
          {
            "line": 214,
            "content": "            }"
          },
          {
            "line": 215,
            "content": "            "
          },
          {
            "line": 216,
            "content": "            lines_data.append(line_data)"
          },
          {
            "line": 217,
            "content": "        "
          }
        ]
      },
      {
        "line_number": 216,
        "content": "            lines_data.append(line_data)",
        "ast_features": {
          "node_type": "Call",
          "depth": 3,
          "children_types": [
            "Attribute",
            "Name"
          ]
        },
        "context_window": [
          {
            "line": 214,
            "content": "            }"
          },
          {
            "line": 215,
            "content": "            "
          },
          {
            "line": 216,
            "content": "            lines_data.append(line_data)"
          },
          {
            "line": 217,
            "content": "        "
          },
          {
            "line": 218,
            "content": "        return lines_data"
          }
        ]
      },
      {
        "line_number": 217,
        "content": "        ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 215,
            "content": "            "
          },
          {
            "line": 216,
            "content": "            lines_data.append(line_data)"
          },
          {
            "line": 217,
            "content": "        "
          },
          {
            "line": 218,
            "content": "        return lines_data"
          },
          {
            "line": 219,
            "content": ""
          }
        ]
      },
      {
        "line_number": 218,
        "content": "        return lines_data",
        "ast_features": {
          "node_type": "Return",
          "depth": 2,
          "children_types": [
            "Name"
          ]
        },
        "context_window": [
          {
            "line": 216,
            "content": "            lines_data.append(line_data)"
          },
          {
            "line": 217,
            "content": "        "
          },
          {
            "line": 218,
            "content": "        return lines_data"
          },
          {
            "line": 219,
            "content": ""
          },
          {
            "line": 220,
            "content": "    def _extract_ast_features(self, line_content: str, line_num: int, "
          }
        ]
      },
      {
        "line_number": 219,
        "content": "",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 217,
            "content": "        "
          },
          {
            "line": 218,
            "content": "        return lines_data"
          },
          {
            "line": 219,
            "content": ""
          },
          {
            "line": 220,
            "content": "    def _extract_ast_features(self, line_content: str, line_num: int, "
          },
          {
            "line": 221,
            "content": "                             full_content: str, language: str) -> Dict[str, Any]:"
          }
        ]
      },
      {
        "line_number": 220,
        "content": "    def _extract_ast_features(self, line_content: str, line_num: int, ",
        "ast_features": {
          "node_type": "FunctionDef",
          "depth": 1,
          "children_types": [
            "Name",
            "arguments"
          ]
        },
        "context_window": [
          {
            "line": 218,
            "content": "        return lines_data"
          },
          {
            "line": 219,
            "content": ""
          },
          {
            "line": 220,
            "content": "    def _extract_ast_features(self, line_content: str, line_num: int, "
          },
          {
            "line": 221,
            "content": "                             full_content: str, language: str) -> Dict[str, Any]:"
          },
          {
            "line": 222,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 221,
        "content": "                             full_content: str, language: str) -> Dict[str, Any]:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 219,
            "content": ""
          },
          {
            "line": 220,
            "content": "    def _extract_ast_features(self, line_content: str, line_num: int, "
          },
          {
            "line": 221,
            "content": "                             full_content: str, language: str) -> Dict[str, Any]:"
          },
          {
            "line": 222,
            "content": "        \"\"\""
          },
          {
            "line": 223,
            "content": "        提取行的AST特征"
          }
        ]
      },
      {
        "line_number": 222,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 220,
            "content": "    def _extract_ast_features(self, line_content: str, line_num: int, "
          },
          {
            "line": 221,
            "content": "                             full_content: str, language: str) -> Dict[str, Any]:"
          },
          {
            "line": 222,
            "content": "        \"\"\""
          },
          {
            "line": 223,
            "content": "        提取行的AST特征"
          },
          {
            "line": 224,
            "content": "        :param line_content: 当前行内容"
          }
        ]
      },
      {
        "line_number": 223,
        "content": "        提取行的AST特征",
        "ast_features": {
          "node_type": "Name",
          "depth": 1,
          "children_types": [
            "Load"
          ]
        },
        "context_window": [
          {
            "line": 221,
            "content": "                             full_content: str, language: str) -> Dict[str, Any]:"
          },
          {
            "line": 222,
            "content": "        \"\"\""
          },
          {
            "line": 223,
            "content": "        提取行的AST特征"
          },
          {
            "line": 224,
            "content": "        :param line_content: 当前行内容"
          },
          {
            "line": 225,
            "content": "        :param line_num: 行号"
          }
        ]
      },
      {
        "line_number": 224,
        "content": "        :param line_content: 当前行内容",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 222,
            "content": "        \"\"\""
          },
          {
            "line": 223,
            "content": "        提取行的AST特征"
          },
          {
            "line": 224,
            "content": "        :param line_content: 当前行内容"
          },
          {
            "line": 225,
            "content": "        :param line_num: 行号"
          },
          {
            "line": 226,
            "content": "        :param full_content: 完整文件内容"
          }
        ]
      },
      {
        "line_number": 225,
        "content": "        :param line_num: 行号",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 223,
            "content": "        提取行的AST特征"
          },
          {
            "line": 224,
            "content": "        :param line_content: 当前行内容"
          },
          {
            "line": 225,
            "content": "        :param line_num: 行号"
          },
          {
            "line": 226,
            "content": "        :param full_content: 完整文件内容"
          },
          {
            "line": 227,
            "content": "        :param language: 编程语言"
          }
        ]
      },
      {
        "line_number": 226,
        "content": "        :param full_content: 完整文件内容",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 224,
            "content": "        :param line_content: 当前行内容"
          },
          {
            "line": 225,
            "content": "        :param line_num: 行号"
          },
          {
            "line": 226,
            "content": "        :param full_content: 完整文件内容"
          },
          {
            "line": 227,
            "content": "        :param language: 编程语言"
          },
          {
            "line": 228,
            "content": "        :return: AST特征字典"
          }
        ]
      },
      {
        "line_number": 227,
        "content": "        :param language: 编程语言",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 225,
            "content": "        :param line_num: 行号"
          },
          {
            "line": 226,
            "content": "        :param full_content: 完整文件内容"
          },
          {
            "line": 227,
            "content": "        :param language: 编程语言"
          },
          {
            "line": 228,
            "content": "        :return: AST特征字典"
          },
          {
            "line": 229,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 228,
        "content": "        :return: AST特征字典",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 226,
            "content": "        :param full_content: 完整文件内容"
          },
          {
            "line": 227,
            "content": "        :param language: 编程语言"
          },
          {
            "line": 228,
            "content": "        :return: AST特征字典"
          },
          {
            "line": 229,
            "content": "        \"\"\""
          },
          {
            "line": 230,
            "content": "        if language == 'Python':"
          }
        ]
      },
      {
        "line_number": 229,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 227,
            "content": "        :param language: 编程语言"
          },
          {
            "line": 228,
            "content": "        :return: AST特征字典"
          },
          {
            "line": 229,
            "content": "        \"\"\""
          },
          {
            "line": 230,
            "content": "        if language == 'Python':"
          },
          {
            "line": 231,
            "content": "            return self._extract_python_ast_features(line_content, full_content)"
          }
        ]
      },
      {
        "line_number": 230,
        "content": "        if language == 'Python':",
        "ast_features": {
          "node_type": "Assign",
          "depth": 1,
          "children_types": [
            "Name",
            "Constant"
          ]
        },
        "context_window": [
          {
            "line": 228,
            "content": "        :return: AST特征字典"
          },
          {
            "line": 229,
            "content": "        \"\"\""
          },
          {
            "line": 230,
            "content": "        if language == 'Python':"
          },
          {
            "line": 231,
            "content": "            return self._extract_python_ast_features(line_content, full_content)"
          },
          {
            "line": 232,
            "content": "        elif language == 'Java':"
          }
        ]
      },
      {
        "line_number": 231,
        "content": "            return self._extract_python_ast_features(line_content, full_content)",
        "ast_features": {
          "node_type": "Return",
          "depth": 4,
          "children_types": [
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 229,
            "content": "        \"\"\""
          },
          {
            "line": 230,
            "content": "        if language == 'Python':"
          },
          {
            "line": 231,
            "content": "            return self._extract_python_ast_features(line_content, full_content)"
          },
          {
            "line": 232,
            "content": "        elif language == 'Java':"
          },
          {
            "line": 233,
            "content": "            return self._extract_java_ast_features(line_content)"
          }
        ]
      },
      {
        "line_number": 232,
        "content": "        elif language == 'Java':",
        "ast_features": {
          "node_type": "Assign",
          "depth": 1,
          "children_types": [
            "Name",
            "Constant"
          ]
        },
        "context_window": [
          {
            "line": 230,
            "content": "        if language == 'Python':"
          },
          {
            "line": 231,
            "content": "            return self._extract_python_ast_features(line_content, full_content)"
          },
          {
            "line": 232,
            "content": "        elif language == 'Java':"
          },
          {
            "line": 233,
            "content": "            return self._extract_java_ast_features(line_content)"
          },
          {
            "line": 234,
            "content": "        elif language == 'JavaScript':"
          }
        ]
      },
      {
        "line_number": 233,
        "content": "            return self._extract_java_ast_features(line_content)",
        "ast_features": {
          "node_type": "Return",
          "depth": 4,
          "children_types": [
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 231,
            "content": "            return self._extract_python_ast_features(line_content, full_content)"
          },
          {
            "line": 232,
            "content": "        elif language == 'Java':"
          },
          {
            "line": 233,
            "content": "            return self._extract_java_ast_features(line_content)"
          },
          {
            "line": 234,
            "content": "        elif language == 'JavaScript':"
          },
          {
            "line": 235,
            "content": "            return self._extract_javascript_ast_features(line_content)"
          }
        ]
      },
      {
        "line_number": 234,
        "content": "        elif language == 'JavaScript':",
        "ast_features": {
          "node_type": "Assign",
          "depth": 1,
          "children_types": [
            "Name",
            "Constant"
          ]
        },
        "context_window": [
          {
            "line": 232,
            "content": "        elif language == 'Java':"
          },
          {
            "line": 233,
            "content": "            return self._extract_java_ast_features(line_content)"
          },
          {
            "line": 234,
            "content": "        elif language == 'JavaScript':"
          },
          {
            "line": 235,
            "content": "            return self._extract_javascript_ast_features(line_content)"
          },
          {
            "line": 236,
            "content": "        else:"
          }
        ]
      },
      {
        "line_number": 235,
        "content": "            return self._extract_javascript_ast_features(line_content)",
        "ast_features": {
          "node_type": "Return",
          "depth": 4,
          "children_types": [
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 233,
            "content": "            return self._extract_java_ast_features(line_content)"
          },
          {
            "line": 234,
            "content": "        elif language == 'JavaScript':"
          },
          {
            "line": 235,
            "content": "            return self._extract_javascript_ast_features(line_content)"
          },
          {
            "line": 236,
            "content": "        else:"
          },
          {
            "line": 237,
            "content": "            return {"
          }
        ]
      },
      {
        "line_number": 236,
        "content": "        else:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 234,
            "content": "        elif language == 'JavaScript':"
          },
          {
            "line": 235,
            "content": "            return self._extract_javascript_ast_features(line_content)"
          },
          {
            "line": 236,
            "content": "        else:"
          },
          {
            "line": 237,
            "content": "            return {"
          },
          {
            "line": 238,
            "content": "                \"node_type\": \"unknown\","
          }
        ]
      },
      {
        "line_number": 237,
        "content": "            return {",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 235,
            "content": "            return self._extract_javascript_ast_features(line_content)"
          },
          {
            "line": 236,
            "content": "        else:"
          },
          {
            "line": 237,
            "content": "            return {"
          },
          {
            "line": 238,
            "content": "                \"node_type\": \"unknown\","
          },
          {
            "line": 239,
            "content": "                \"depth\": 0,"
          }
        ]
      },
      {
        "line_number": 238,
        "content": "                \"node_type\": \"unknown\",",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 236,
            "content": "        else:"
          },
          {
            "line": 237,
            "content": "            return {"
          },
          {
            "line": 238,
            "content": "                \"node_type\": \"unknown\","
          },
          {
            "line": 239,
            "content": "                \"depth\": 0,"
          },
          {
            "line": 240,
            "content": "                \"children_types\": []"
          }
        ]
      },
      {
        "line_number": 239,
        "content": "                \"depth\": 0,",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 237,
            "content": "            return {"
          },
          {
            "line": 238,
            "content": "                \"node_type\": \"unknown\","
          },
          {
            "line": 239,
            "content": "                \"depth\": 0,"
          },
          {
            "line": 240,
            "content": "                \"children_types\": []"
          },
          {
            "line": 241,
            "content": "            }"
          }
        ]
      },
      {
        "line_number": 240,
        "content": "                \"children_types\": []",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 238,
            "content": "                \"node_type\": \"unknown\","
          },
          {
            "line": 239,
            "content": "                \"depth\": 0,"
          },
          {
            "line": 240,
            "content": "                \"children_types\": []"
          },
          {
            "line": 241,
            "content": "            }"
          },
          {
            "line": 242,
            "content": ""
          }
        ]
      },
      {
        "line_number": 241,
        "content": "            }",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 239,
            "content": "                \"depth\": 0,"
          },
          {
            "line": 240,
            "content": "                \"children_types\": []"
          },
          {
            "line": 241,
            "content": "            }"
          },
          {
            "line": 242,
            "content": ""
          },
          {
            "line": 243,
            "content": "    def _extract_python_ast_features(self, line_content: str, full_content: str) -> Dict[str, Any]:"
          }
        ]
      },
      {
        "line_number": 242,
        "content": "",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 240,
            "content": "                \"children_types\": []"
          },
          {
            "line": 241,
            "content": "            }"
          },
          {
            "line": 242,
            "content": ""
          },
          {
            "line": 243,
            "content": "    def _extract_python_ast_features(self, line_content: str, full_content: str) -> Dict[str, Any]:"
          },
          {
            "line": 244,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 243,
        "content": "    def _extract_python_ast_features(self, line_content: str, full_content: str) -> Dict[str, Any]:",
        "ast_features": {
          "node_type": "FunctionDef",
          "depth": 1,
          "children_types": [
            "Name",
            "arguments"
          ]
        },
        "context_window": [
          {
            "line": 241,
            "content": "            }"
          },
          {
            "line": 242,
            "content": ""
          },
          {
            "line": 243,
            "content": "    def _extract_python_ast_features(self, line_content: str, full_content: str) -> Dict[str, Any]:"
          },
          {
            "line": 244,
            "content": "        \"\"\""
          },
          {
            "line": 245,
            "content": "        提取Python AST特征"
          }
        ]
      },
      {
        "line_number": 244,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 242,
            "content": ""
          },
          {
            "line": 243,
            "content": "    def _extract_python_ast_features(self, line_content: str, full_content: str) -> Dict[str, Any]:"
          },
          {
            "line": 244,
            "content": "        \"\"\""
          },
          {
            "line": 245,
            "content": "        提取Python AST特征"
          },
          {
            "line": 246,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 245,
        "content": "        提取Python AST特征",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 243,
            "content": "    def _extract_python_ast_features(self, line_content: str, full_content: str) -> Dict[str, Any]:"
          },
          {
            "line": 244,
            "content": "        \"\"\""
          },
          {
            "line": 245,
            "content": "        提取Python AST特征"
          },
          {
            "line": 246,
            "content": "        \"\"\""
          },
          {
            "line": 247,
            "content": "        try:"
          }
        ]
      },
      {
        "line_number": 246,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 244,
            "content": "        \"\"\""
          },
          {
            "line": 245,
            "content": "        提取Python AST特征"
          },
          {
            "line": 246,
            "content": "        \"\"\""
          },
          {
            "line": 247,
            "content": "        try:"
          },
          {
            "line": 248,
            "content": "            # 尝试解析单行"
          }
        ]
      },
      {
        "line_number": 247,
        "content": "        try:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 245,
            "content": "        提取Python AST特征"
          },
          {
            "line": 246,
            "content": "        \"\"\""
          },
          {
            "line": 247,
            "content": "        try:"
          },
          {
            "line": 248,
            "content": "            # 尝试解析单行"
          },
          {
            "line": 249,
            "content": "            if line_content.strip():"
          }
        ]
      },
      {
        "line_number": 248,
        "content": "            # 尝试解析单行",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 246,
            "content": "        \"\"\""
          },
          {
            "line": 247,
            "content": "        try:"
          },
          {
            "line": 248,
            "content": "            # 尝试解析单行"
          },
          {
            "line": 249,
            "content": "            if line_content.strip():"
          },
          {
            "line": 250,
            "content": "                # 对于单行，我们尝试解析为表达式或语句"
          }
        ]
      },
      {
        "line_number": 249,
        "content": "            if line_content.strip():",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 247,
            "content": "        try:"
          },
          {
            "line": 248,
            "content": "            # 尝试解析单行"
          },
          {
            "line": 249,
            "content": "            if line_content.strip():"
          },
          {
            "line": 250,
            "content": "                # 对于单行，我们尝试解析为表达式或语句"
          },
          {
            "line": 251,
            "content": "                try:"
          }
        ]
      },
      {
        "line_number": 250,
        "content": "                # 对于单行，我们尝试解析为表达式或语句",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 248,
            "content": "            # 尝试解析单行"
          },
          {
            "line": 249,
            "content": "            if line_content.strip():"
          },
          {
            "line": 250,
            "content": "                # 对于单行，我们尝试解析为表达式或语句"
          },
          {
            "line": 251,
            "content": "                try:"
          },
          {
            "line": 252,
            "content": "                    # 先尝试作为表达式解析"
          }
        ]
      },
      {
        "line_number": 251,
        "content": "                try:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 249,
            "content": "            if line_content.strip():"
          },
          {
            "line": 250,
            "content": "                # 对于单行，我们尝试解析为表达式或语句"
          },
          {
            "line": 251,
            "content": "                try:"
          },
          {
            "line": 252,
            "content": "                    # 先尝试作为表达式解析"
          },
          {
            "line": 253,
            "content": "                    tree = ast.parse(line_content.strip(), mode='eval')"
          }
        ]
      },
      {
        "line_number": 252,
        "content": "                    # 先尝试作为表达式解析",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 250,
            "content": "                # 对于单行，我们尝试解析为表达式或语句"
          },
          {
            "line": 251,
            "content": "                try:"
          },
          {
            "line": 252,
            "content": "                    # 先尝试作为表达式解析"
          },
          {
            "line": 253,
            "content": "                    tree = ast.parse(line_content.strip(), mode='eval')"
          },
          {
            "line": 254,
            "content": "                    node = tree.body"
          }
        ]
      },
      {
        "line_number": 253,
        "content": "                    tree = ast.parse(line_content.strip(), mode='eval')",
        "ast_features": {
          "node_type": "Assign",
          "depth": 5,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 251,
            "content": "                try:"
          },
          {
            "line": 252,
            "content": "                    # 先尝试作为表达式解析"
          },
          {
            "line": 253,
            "content": "                    tree = ast.parse(line_content.strip(), mode='eval')"
          },
          {
            "line": 254,
            "content": "                    node = tree.body"
          },
          {
            "line": 255,
            "content": "                except SyntaxError:"
          }
        ]
      },
      {
        "line_number": 254,
        "content": "                    node = tree.body",
        "ast_features": {
          "node_type": "Assign",
          "depth": 3,
          "children_types": [
            "Name",
            "Attribute"
          ]
        },
        "context_window": [
          {
            "line": 252,
            "content": "                    # 先尝试作为表达式解析"
          },
          {
            "line": 253,
            "content": "                    tree = ast.parse(line_content.strip(), mode='eval')"
          },
          {
            "line": 254,
            "content": "                    node = tree.body"
          },
          {
            "line": 255,
            "content": "                except SyntaxError:"
          },
          {
            "line": 256,
            "content": "                    try:"
          }
        ]
      },
      {
        "line_number": 255,
        "content": "                except SyntaxError:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 253,
            "content": "                    tree = ast.parse(line_content.strip(), mode='eval')"
          },
          {
            "line": 254,
            "content": "                    node = tree.body"
          },
          {
            "line": 255,
            "content": "                except SyntaxError:"
          },
          {
            "line": 256,
            "content": "                    try:"
          },
          {
            "line": 257,
            "content": "                        # 再尝试作为语句解析"
          }
        ]
      },
      {
        "line_number": 256,
        "content": "                    try:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 254,
            "content": "                    node = tree.body"
          },
          {
            "line": 255,
            "content": "                except SyntaxError:"
          },
          {
            "line": 256,
            "content": "                    try:"
          },
          {
            "line": 257,
            "content": "                        # 再尝试作为语句解析"
          },
          {
            "line": 258,
            "content": "                        tree = ast.parse(line_content.strip(), mode='exec')"
          }
        ]
      },
      {
        "line_number": 257,
        "content": "                        # 再尝试作为语句解析",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 255,
            "content": "                except SyntaxError:"
          },
          {
            "line": 256,
            "content": "                    try:"
          },
          {
            "line": 257,
            "content": "                        # 再尝试作为语句解析"
          },
          {
            "line": 258,
            "content": "                        tree = ast.parse(line_content.strip(), mode='exec')"
          },
          {
            "line": 259,
            "content": "                        if tree.body:"
          }
        ]
      },
      {
        "line_number": 258,
        "content": "                        tree = ast.parse(line_content.strip(), mode='exec')",
        "ast_features": {
          "node_type": "Assign",
          "depth": 5,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 256,
            "content": "                    try:"
          },
          {
            "line": 257,
            "content": "                        # 再尝试作为语句解析"
          },
          {
            "line": 258,
            "content": "                        tree = ast.parse(line_content.strip(), mode='exec')"
          },
          {
            "line": 259,
            "content": "                        if tree.body:"
          },
          {
            "line": 260,
            "content": "                            node = tree.body[0]"
          }
        ]
      },
      {
        "line_number": 259,
        "content": "                        if tree.body:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 257,
            "content": "                        # 再尝试作为语句解析"
          },
          {
            "line": 258,
            "content": "                        tree = ast.parse(line_content.strip(), mode='exec')"
          },
          {
            "line": 259,
            "content": "                        if tree.body:"
          },
          {
            "line": 260,
            "content": "                            node = tree.body[0]"
          },
          {
            "line": 261,
            "content": "                        else:"
          }
        ]
      },
      {
        "line_number": 260,
        "content": "                            node = tree.body[0]",
        "ast_features": {
          "node_type": "Assign",
          "depth": 4,
          "children_types": [
            "Name",
            "Subscript"
          ]
        },
        "context_window": [
          {
            "line": 258,
            "content": "                        tree = ast.parse(line_content.strip(), mode='exec')"
          },
          {
            "line": 259,
            "content": "                        if tree.body:"
          },
          {
            "line": 260,
            "content": "                            node = tree.body[0]"
          },
          {
            "line": 261,
            "content": "                        else:"
          },
          {
            "line": 262,
            "content": "                            raise SyntaxError(\"Empty statement\")"
          }
        ]
      },
      {
        "line_number": 261,
        "content": "                        else:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 259,
            "content": "                        if tree.body:"
          },
          {
            "line": 260,
            "content": "                            node = tree.body[0]"
          },
          {
            "line": 261,
            "content": "                        else:"
          },
          {
            "line": 262,
            "content": "                            raise SyntaxError(\"Empty statement\")"
          },
          {
            "line": 263,
            "content": "                    except SyntaxError:"
          }
        ]
      },
      {
        "line_number": 262,
        "content": "                            raise SyntaxError(\"Empty statement\")",
        "ast_features": {
          "node_type": "Raise",
          "depth": 3,
          "children_types": [
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 260,
            "content": "                            node = tree.body[0]"
          },
          {
            "line": 261,
            "content": "                        else:"
          },
          {
            "line": 262,
            "content": "                            raise SyntaxError(\"Empty statement\")"
          },
          {
            "line": 263,
            "content": "                    except SyntaxError:"
          },
          {
            "line": 264,
            "content": "                        # 如果单行解析失败，尝试在完整文件上下文中找到对应节点"
          }
        ]
      },
      {
        "line_number": 263,
        "content": "                    except SyntaxError:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 261,
            "content": "                        else:"
          },
          {
            "line": 262,
            "content": "                            raise SyntaxError(\"Empty statement\")"
          },
          {
            "line": 263,
            "content": "                    except SyntaxError:"
          },
          {
            "line": 264,
            "content": "                        # 如果单行解析失败，尝试在完整文件上下文中找到对应节点"
          },
          {
            "line": 265,
            "content": "                        return self._fallback_ast_analysis(line_content)"
          }
        ]
      },
      {
        "line_number": 264,
        "content": "                        # 如果单行解析失败，尝试在完整文件上下文中找到对应节点",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 262,
            "content": "                            raise SyntaxError(\"Empty statement\")"
          },
          {
            "line": 263,
            "content": "                    except SyntaxError:"
          },
          {
            "line": 264,
            "content": "                        # 如果单行解析失败，尝试在完整文件上下文中找到对应节点"
          },
          {
            "line": 265,
            "content": "                        return self._fallback_ast_analysis(line_content)"
          },
          {
            "line": 266,
            "content": "                "
          }
        ]
      },
      {
        "line_number": 265,
        "content": "                        return self._fallback_ast_analysis(line_content)",
        "ast_features": {
          "node_type": "Return",
          "depth": 4,
          "children_types": [
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 263,
            "content": "                    except SyntaxError:"
          },
          {
            "line": 264,
            "content": "                        # 如果单行解析失败，尝试在完整文件上下文中找到对应节点"
          },
          {
            "line": 265,
            "content": "                        return self._fallback_ast_analysis(line_content)"
          },
          {
            "line": 266,
            "content": "                "
          },
          {
            "line": 267,
            "content": "                return {"
          }
        ]
      },
      {
        "line_number": 266,
        "content": "                ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 264,
            "content": "                        # 如果单行解析失败，尝试在完整文件上下文中找到对应节点"
          },
          {
            "line": 265,
            "content": "                        return self._fallback_ast_analysis(line_content)"
          },
          {
            "line": 266,
            "content": "                "
          },
          {
            "line": 267,
            "content": "                return {"
          },
          {
            "line": 268,
            "content": "                    \"node_type\": type(node).__name__,"
          }
        ]
      },
      {
        "line_number": 267,
        "content": "                return {",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 265,
            "content": "                        return self._fallback_ast_analysis(line_content)"
          },
          {
            "line": 266,
            "content": "                "
          },
          {
            "line": 267,
            "content": "                return {"
          },
          {
            "line": 268,
            "content": "                    \"node_type\": type(node).__name__,"
          },
          {
            "line": 269,
            "content": "                    \"depth\": self._calculate_ast_depth(node),"
          }
        ]
      },
      {
        "line_number": 268,
        "content": "                    \"node_type\": type(node).__name__,",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 266,
            "content": "                "
          },
          {
            "line": 267,
            "content": "                return {"
          },
          {
            "line": 268,
            "content": "                    \"node_type\": type(node).__name__,"
          },
          {
            "line": 269,
            "content": "                    \"depth\": self._calculate_ast_depth(node),"
          },
          {
            "line": 270,
            "content": "                    \"children_types\": [type(child).__name__ for child in ast.iter_child_nodes(node)]"
          }
        ]
      },
      {
        "line_number": 269,
        "content": "                    \"depth\": self._calculate_ast_depth(node),",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 267,
            "content": "                return {"
          },
          {
            "line": 268,
            "content": "                    \"node_type\": type(node).__name__,"
          },
          {
            "line": 269,
            "content": "                    \"depth\": self._calculate_ast_depth(node),"
          },
          {
            "line": 270,
            "content": "                    \"children_types\": [type(child).__name__ for child in ast.iter_child_nodes(node)]"
          },
          {
            "line": 271,
            "content": "                }"
          }
        ]
      },
      {
        "line_number": 270,
        "content": "                    \"children_types\": [type(child).__name__ for child in ast.iter_child_nodes(node)]",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 268,
            "content": "                    \"node_type\": type(node).__name__,"
          },
          {
            "line": 269,
            "content": "                    \"depth\": self._calculate_ast_depth(node),"
          },
          {
            "line": 270,
            "content": "                    \"children_types\": [type(child).__name__ for child in ast.iter_child_nodes(node)]"
          },
          {
            "line": 271,
            "content": "                }"
          },
          {
            "line": 272,
            "content": "            else:"
          }
        ]
      },
      {
        "line_number": 271,
        "content": "                }",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 269,
            "content": "                    \"depth\": self._calculate_ast_depth(node),"
          },
          {
            "line": 270,
            "content": "                    \"children_types\": [type(child).__name__ for child in ast.iter_child_nodes(node)]"
          },
          {
            "line": 271,
            "content": "                }"
          },
          {
            "line": 272,
            "content": "            else:"
          },
          {
            "line": 273,
            "content": "                return {"
          }
        ]
      },
      {
        "line_number": 272,
        "content": "            else:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 270,
            "content": "                    \"children_types\": [type(child).__name__ for child in ast.iter_child_nodes(node)]"
          },
          {
            "line": 271,
            "content": "                }"
          },
          {
            "line": 272,
            "content": "            else:"
          },
          {
            "line": 273,
            "content": "                return {"
          },
          {
            "line": 274,
            "content": "                    \"node_type\": \"Empty\","
          }
        ]
      },
      {
        "line_number": 273,
        "content": "                return {",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 271,
            "content": "                }"
          },
          {
            "line": 272,
            "content": "            else:"
          },
          {
            "line": 273,
            "content": "                return {"
          },
          {
            "line": 274,
            "content": "                    \"node_type\": \"Empty\","
          },
          {
            "line": 275,
            "content": "                    \"depth\": 0,"
          }
        ]
      },
      {
        "line_number": 274,
        "content": "                    \"node_type\": \"Empty\",",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 272,
            "content": "            else:"
          },
          {
            "line": 273,
            "content": "                return {"
          },
          {
            "line": 274,
            "content": "                    \"node_type\": \"Empty\","
          },
          {
            "line": 275,
            "content": "                    \"depth\": 0,"
          },
          {
            "line": 276,
            "content": "                    \"children_types\": []"
          }
        ]
      },
      {
        "line_number": 275,
        "content": "                    \"depth\": 0,",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 273,
            "content": "                return {"
          },
          {
            "line": 274,
            "content": "                    \"node_type\": \"Empty\","
          },
          {
            "line": 275,
            "content": "                    \"depth\": 0,"
          },
          {
            "line": 276,
            "content": "                    \"children_types\": []"
          },
          {
            "line": 277,
            "content": "                }"
          }
        ]
      },
      {
        "line_number": 276,
        "content": "                    \"children_types\": []",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 274,
            "content": "                    \"node_type\": \"Empty\","
          },
          {
            "line": 275,
            "content": "                    \"depth\": 0,"
          },
          {
            "line": 276,
            "content": "                    \"children_types\": []"
          },
          {
            "line": 277,
            "content": "                }"
          },
          {
            "line": 278,
            "content": "        except Exception as e:"
          }
        ]
      },
      {
        "line_number": 277,
        "content": "                }",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 275,
            "content": "                    \"depth\": 0,"
          },
          {
            "line": 276,
            "content": "                    \"children_types\": []"
          },
          {
            "line": 277,
            "content": "                }"
          },
          {
            "line": 278,
            "content": "        except Exception as e:"
          },
          {
            "line": 279,
            "content": "            logger.debug(f\"AST parsing error for line '{line_content}': {str(e)}\")"
          }
        ]
      },
      {
        "line_number": 278,
        "content": "        except Exception as e:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 276,
            "content": "                    \"children_types\": []"
          },
          {
            "line": 277,
            "content": "                }"
          },
          {
            "line": 278,
            "content": "        except Exception as e:"
          },
          {
            "line": 279,
            "content": "            logger.debug(f\"AST parsing error for line '{line_content}': {str(e)}\")"
          },
          {
            "line": 280,
            "content": "            return self._fallback_ast_analysis(line_content)"
          }
        ]
      },
      {
        "line_number": 279,
        "content": "            logger.debug(f\"AST parsing error for line '{line_content}': {str(e)}\")",
        "ast_features": {
          "node_type": "Call",
          "depth": 5,
          "children_types": [
            "Attribute",
            "JoinedStr"
          ]
        },
        "context_window": [
          {
            "line": 277,
            "content": "                }"
          },
          {
            "line": 278,
            "content": "        except Exception as e:"
          },
          {
            "line": 279,
            "content": "            logger.debug(f\"AST parsing error for line '{line_content}': {str(e)}\")"
          },
          {
            "line": 280,
            "content": "            return self._fallback_ast_analysis(line_content)"
          },
          {
            "line": 281,
            "content": ""
          }
        ]
      },
      {
        "line_number": 280,
        "content": "            return self._fallback_ast_analysis(line_content)",
        "ast_features": {
          "node_type": "Return",
          "depth": 4,
          "children_types": [
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 278,
            "content": "        except Exception as e:"
          },
          {
            "line": 279,
            "content": "            logger.debug(f\"AST parsing error for line '{line_content}': {str(e)}\")"
          },
          {
            "line": 280,
            "content": "            return self._fallback_ast_analysis(line_content)"
          },
          {
            "line": 281,
            "content": ""
          },
          {
            "line": 282,
            "content": "    def _extract_java_ast_features(self, line_content: str) -> Dict[str, Any]:"
          }
        ]
      },
      {
        "line_number": 281,
        "content": "",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 279,
            "content": "            logger.debug(f\"AST parsing error for line '{line_content}': {str(e)}\")"
          },
          {
            "line": 280,
            "content": "            return self._fallback_ast_analysis(line_content)"
          },
          {
            "line": 281,
            "content": ""
          },
          {
            "line": 282,
            "content": "    def _extract_java_ast_features(self, line_content: str) -> Dict[str, Any]:"
          },
          {
            "line": 283,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 282,
        "content": "    def _extract_java_ast_features(self, line_content: str) -> Dict[str, Any]:",
        "ast_features": {
          "node_type": "FunctionDef",
          "depth": 1,
          "children_types": [
            "Name",
            "arguments"
          ]
        },
        "context_window": [
          {
            "line": 280,
            "content": "            return self._fallback_ast_analysis(line_content)"
          },
          {
            "line": 281,
            "content": ""
          },
          {
            "line": 282,
            "content": "    def _extract_java_ast_features(self, line_content: str) -> Dict[str, Any]:"
          },
          {
            "line": 283,
            "content": "        \"\"\""
          },
          {
            "line": 284,
            "content": "        提取Java AST特征（简化版本）"
          }
        ]
      },
      {
        "line_number": 283,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 281,
            "content": ""
          },
          {
            "line": 282,
            "content": "    def _extract_java_ast_features(self, line_content: str) -> Dict[str, Any]:"
          },
          {
            "line": 283,
            "content": "        \"\"\""
          },
          {
            "line": 284,
            "content": "        提取Java AST特征（简化版本）"
          },
          {
            "line": 285,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 284,
        "content": "        提取Java AST特征（简化版本）",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 282,
            "content": "    def _extract_java_ast_features(self, line_content: str) -> Dict[str, Any]:"
          },
          {
            "line": 283,
            "content": "        \"\"\""
          },
          {
            "line": 284,
            "content": "        提取Java AST特征（简化版本）"
          },
          {
            "line": 285,
            "content": "        \"\"\""
          },
          {
            "line": 286,
            "content": "        # 由于Java AST解析较复杂，这里实现基础的模式匹配"
          }
        ]
      },
      {
        "line_number": 285,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 283,
            "content": "        \"\"\""
          },
          {
            "line": 284,
            "content": "        提取Java AST特征（简化版本）"
          },
          {
            "line": 285,
            "content": "        \"\"\""
          },
          {
            "line": 286,
            "content": "        # 由于Java AST解析较复杂，这里实现基础的模式匹配"
          },
          {
            "line": 287,
            "content": "        line = line_content.strip()"
          }
        ]
      },
      {
        "line_number": 286,
        "content": "        # 由于Java AST解析较复杂，这里实现基础的模式匹配",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 284,
            "content": "        提取Java AST特征（简化版本）"
          },
          {
            "line": 285,
            "content": "        \"\"\""
          },
          {
            "line": 286,
            "content": "        # 由于Java AST解析较复杂，这里实现基础的模式匹配"
          },
          {
            "line": 287,
            "content": "        line = line_content.strip()"
          },
          {
            "line": 288,
            "content": "        "
          }
        ]
      },
      {
        "line_number": 287,
        "content": "        line = line_content.strip()",
        "ast_features": {
          "node_type": "Assign",
          "depth": 4,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 285,
            "content": "        \"\"\""
          },
          {
            "line": 286,
            "content": "        # 由于Java AST解析较复杂，这里实现基础的模式匹配"
          },
          {
            "line": 287,
            "content": "        line = line_content.strip()"
          },
          {
            "line": 288,
            "content": "        "
          },
          {
            "line": 289,
            "content": "        if not line or line.startswith('//') or line.startswith('/*'):"
          }
        ]
      },
      {
        "line_number": 288,
        "content": "        ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 286,
            "content": "        # 由于Java AST解析较复杂，这里实现基础的模式匹配"
          },
          {
            "line": 287,
            "content": "        line = line_content.strip()"
          },
          {
            "line": 288,
            "content": "        "
          },
          {
            "line": 289,
            "content": "        if not line or line.startswith('//') or line.startswith('/*'):"
          },
          {
            "line": 290,
            "content": "            return {\"node_type\": \"Comment\", \"depth\": 0, \"children_types\": []}"
          }
        ]
      },
      {
        "line_number": 289,
        "content": "        if not line or line.startswith('//') or line.startswith('/*'):",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 287,
            "content": "        line = line_content.strip()"
          },
          {
            "line": 288,
            "content": "        "
          },
          {
            "line": 289,
            "content": "        if not line or line.startswith('//') or line.startswith('/*'):"
          },
          {
            "line": 290,
            "content": "            return {\"node_type\": \"Comment\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 291,
            "content": "        elif line.startswith('import '):"
          }
        ]
      },
      {
        "line_number": 290,
        "content": "            return {\"node_type\": \"Comment\", \"depth\": 0, \"children_types\": []}",
        "ast_features": {
          "node_type": "Return",
          "depth": 3,
          "children_types": [
            "Dict"
          ]
        },
        "context_window": [
          {
            "line": 288,
            "content": "        "
          },
          {
            "line": 289,
            "content": "        if not line or line.startswith('//') or line.startswith('/*'):"
          },
          {
            "line": 290,
            "content": "            return {\"node_type\": \"Comment\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 291,
            "content": "        elif line.startswith('import '):"
          },
          {
            "line": 292,
            "content": "            return {\"node_type\": \"ImportDeclaration\", \"depth\": 0, \"children_types\": []}"
          }
        ]
      },
      {
        "line_number": 291,
        "content": "        elif line.startswith('import '):",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 289,
            "content": "        if not line or line.startswith('//') or line.startswith('/*'):"
          },
          {
            "line": 290,
            "content": "            return {\"node_type\": \"Comment\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 291,
            "content": "        elif line.startswith('import '):"
          },
          {
            "line": 292,
            "content": "            return {\"node_type\": \"ImportDeclaration\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 293,
            "content": "        elif 'class ' in line:"
          }
        ]
      },
      {
        "line_number": 292,
        "content": "            return {\"node_type\": \"ImportDeclaration\", \"depth\": 0, \"children_types\": []}",
        "ast_features": {
          "node_type": "Return",
          "depth": 3,
          "children_types": [
            "Dict"
          ]
        },
        "context_window": [
          {
            "line": 290,
            "content": "            return {\"node_type\": \"Comment\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 291,
            "content": "        elif line.startswith('import '):"
          },
          {
            "line": 292,
            "content": "            return {\"node_type\": \"ImportDeclaration\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 293,
            "content": "        elif 'class ' in line:"
          },
          {
            "line": 294,
            "content": "            return {\"node_type\": \"ClassDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\"]}"
          }
        ]
      },
      {
        "line_number": 293,
        "content": "        elif 'class ' in line:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 291,
            "content": "        elif line.startswith('import '):"
          },
          {
            "line": 292,
            "content": "            return {\"node_type\": \"ImportDeclaration\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 293,
            "content": "        elif 'class ' in line:"
          },
          {
            "line": 294,
            "content": "            return {\"node_type\": \"ClassDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\"]}"
          },
          {
            "line": 295,
            "content": "        elif re.search(r'\\b(public|private|protected)\\s+.*\\(.*\\)', line):"
          }
        ]
      },
      {
        "line_number": 294,
        "content": "            return {\"node_type\": \"ClassDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\"]}",
        "ast_features": {
          "node_type": "Return",
          "depth": 4,
          "children_types": [
            "Dict"
          ]
        },
        "context_window": [
          {
            "line": 292,
            "content": "            return {\"node_type\": \"ImportDeclaration\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 293,
            "content": "        elif 'class ' in line:"
          },
          {
            "line": 294,
            "content": "            return {\"node_type\": \"ClassDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\"]}"
          },
          {
            "line": 295,
            "content": "        elif re.search(r'\\b(public|private|protected)\\s+.*\\(.*\\)', line):"
          },
          {
            "line": 296,
            "content": "            return {\"node_type\": \"MethodDeclaration\", \"depth\": 2, \"children_types\": [\"Identifier\", \"Parameters\"]}"
          }
        ]
      },
      {
        "line_number": 295,
        "content": "        elif re.search(r'\\b(public|private|protected)\\s+.*\\(.*\\)', line):",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 293,
            "content": "        elif 'class ' in line:"
          },
          {
            "line": 294,
            "content": "            return {\"node_type\": \"ClassDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\"]}"
          },
          {
            "line": 295,
            "content": "        elif re.search(r'\\b(public|private|protected)\\s+.*\\(.*\\)', line):"
          },
          {
            "line": 296,
            "content": "            return {\"node_type\": \"MethodDeclaration\", \"depth\": 2, \"children_types\": [\"Identifier\", \"Parameters\"]}"
          },
          {
            "line": 297,
            "content": "        elif '=' in line and ';' in line:"
          }
        ]
      },
      {
        "line_number": 296,
        "content": "            return {\"node_type\": \"MethodDeclaration\", \"depth\": 2, \"children_types\": [\"Identifier\", \"Parameters\"]}",
        "ast_features": {
          "node_type": "Return",
          "depth": 4,
          "children_types": [
            "Dict"
          ]
        },
        "context_window": [
          {
            "line": 294,
            "content": "            return {\"node_type\": \"ClassDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\"]}"
          },
          {
            "line": 295,
            "content": "        elif re.search(r'\\b(public|private|protected)\\s+.*\\(.*\\)', line):"
          },
          {
            "line": 296,
            "content": "            return {\"node_type\": \"MethodDeclaration\", \"depth\": 2, \"children_types\": [\"Identifier\", \"Parameters\"]}"
          },
          {
            "line": 297,
            "content": "        elif '=' in line and ';' in line:"
          },
          {
            "line": 298,
            "content": "            return {\"node_type\": \"VariableDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\", \"Expression\"]}"
          }
        ]
      },
      {
        "line_number": 297,
        "content": "        elif '=' in line and ';' in line:",
        "ast_features": {
          "node_type": "Assign",
          "depth": 1,
          "children_types": [
            "Name",
            "Constant"
          ]
        },
        "context_window": [
          {
            "line": 295,
            "content": "        elif re.search(r'\\b(public|private|protected)\\s+.*\\(.*\\)', line):"
          },
          {
            "line": 296,
            "content": "            return {\"node_type\": \"MethodDeclaration\", \"depth\": 2, \"children_types\": [\"Identifier\", \"Parameters\"]}"
          },
          {
            "line": 297,
            "content": "        elif '=' in line and ';' in line:"
          },
          {
            "line": 298,
            "content": "            return {\"node_type\": \"VariableDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\", \"Expression\"]}"
          },
          {
            "line": 299,
            "content": "        elif line.endswith(';'):"
          }
        ]
      },
      {
        "line_number": 298,
        "content": "            return {\"node_type\": \"VariableDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\", \"Expression\"]}",
        "ast_features": {
          "node_type": "Return",
          "depth": 4,
          "children_types": [
            "Dict"
          ]
        },
        "context_window": [
          {
            "line": 296,
            "content": "            return {\"node_type\": \"MethodDeclaration\", \"depth\": 2, \"children_types\": [\"Identifier\", \"Parameters\"]}"
          },
          {
            "line": 297,
            "content": "        elif '=' in line and ';' in line:"
          },
          {
            "line": 298,
            "content": "            return {\"node_type\": \"VariableDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\", \"Expression\"]}"
          },
          {
            "line": 299,
            "content": "        elif line.endswith(';'):"
          },
          {
            "line": 300,
            "content": "            return {\"node_type\": \"ExpressionStatement\", \"depth\": 1, \"children_types\": [\"Expression\"]}"
          }
        ]
      },
      {
        "line_number": 299,
        "content": "        elif line.endswith(';'):",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 297,
            "content": "        elif '=' in line and ';' in line:"
          },
          {
            "line": 298,
            "content": "            return {\"node_type\": \"VariableDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\", \"Expression\"]}"
          },
          {
            "line": 299,
            "content": "        elif line.endswith(';'):"
          },
          {
            "line": 300,
            "content": "            return {\"node_type\": \"ExpressionStatement\", \"depth\": 1, \"children_types\": [\"Expression\"]}"
          },
          {
            "line": 301,
            "content": "        else:"
          }
        ]
      },
      {
        "line_number": 300,
        "content": "            return {\"node_type\": \"ExpressionStatement\", \"depth\": 1, \"children_types\": [\"Expression\"]}",
        "ast_features": {
          "node_type": "Return",
          "depth": 4,
          "children_types": [
            "Dict"
          ]
        },
        "context_window": [
          {
            "line": 298,
            "content": "            return {\"node_type\": \"VariableDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\", \"Expression\"]}"
          },
          {
            "line": 299,
            "content": "        elif line.endswith(';'):"
          },
          {
            "line": 300,
            "content": "            return {\"node_type\": \"ExpressionStatement\", \"depth\": 1, \"children_types\": [\"Expression\"]}"
          },
          {
            "line": 301,
            "content": "        else:"
          },
          {
            "line": 302,
            "content": "            return {\"node_type\": \"unknown\", \"depth\": 0, \"children_types\": []}"
          }
        ]
      },
      {
        "line_number": 301,
        "content": "        else:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 299,
            "content": "        elif line.endswith(';'):"
          },
          {
            "line": 300,
            "content": "            return {\"node_type\": \"ExpressionStatement\", \"depth\": 1, \"children_types\": [\"Expression\"]}"
          },
          {
            "line": 301,
            "content": "        else:"
          },
          {
            "line": 302,
            "content": "            return {\"node_type\": \"unknown\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 303,
            "content": ""
          }
        ]
      },
      {
        "line_number": 302,
        "content": "            return {\"node_type\": \"unknown\", \"depth\": 0, \"children_types\": []}",
        "ast_features": {
          "node_type": "Return",
          "depth": 3,
          "children_types": [
            "Dict"
          ]
        },
        "context_window": [
          {
            "line": 300,
            "content": "            return {\"node_type\": \"ExpressionStatement\", \"depth\": 1, \"children_types\": [\"Expression\"]}"
          },
          {
            "line": 301,
            "content": "        else:"
          },
          {
            "line": 302,
            "content": "            return {\"node_type\": \"unknown\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 303,
            "content": ""
          },
          {
            "line": 304,
            "content": "    def _extract_javascript_ast_features(self, line_content: str) -> Dict[str, Any]:"
          }
        ]
      },
      {
        "line_number": 303,
        "content": "",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 301,
            "content": "        else:"
          },
          {
            "line": 302,
            "content": "            return {\"node_type\": \"unknown\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 303,
            "content": ""
          },
          {
            "line": 304,
            "content": "    def _extract_javascript_ast_features(self, line_content: str) -> Dict[str, Any]:"
          },
          {
            "line": 305,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 304,
        "content": "    def _extract_javascript_ast_features(self, line_content: str) -> Dict[str, Any]:",
        "ast_features": {
          "node_type": "FunctionDef",
          "depth": 1,
          "children_types": [
            "Name",
            "arguments"
          ]
        },
        "context_window": [
          {
            "line": 302,
            "content": "            return {\"node_type\": \"unknown\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 303,
            "content": ""
          },
          {
            "line": 304,
            "content": "    def _extract_javascript_ast_features(self, line_content: str) -> Dict[str, Any]:"
          },
          {
            "line": 305,
            "content": "        \"\"\""
          },
          {
            "line": 306,
            "content": "        提取JavaScript AST特征（简化版本）"
          }
        ]
      },
      {
        "line_number": 305,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 303,
            "content": ""
          },
          {
            "line": 304,
            "content": "    def _extract_javascript_ast_features(self, line_content: str) -> Dict[str, Any]:"
          },
          {
            "line": 305,
            "content": "        \"\"\""
          },
          {
            "line": 306,
            "content": "        提取JavaScript AST特征（简化版本）"
          },
          {
            "line": 307,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 306,
        "content": "        提取JavaScript AST特征（简化版本）",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 304,
            "content": "    def _extract_javascript_ast_features(self, line_content: str) -> Dict[str, Any]:"
          },
          {
            "line": 305,
            "content": "        \"\"\""
          },
          {
            "line": 306,
            "content": "        提取JavaScript AST特征（简化版本）"
          },
          {
            "line": 307,
            "content": "        \"\"\""
          },
          {
            "line": 308,
            "content": "        line = line_content.strip()"
          }
        ]
      },
      {
        "line_number": 307,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 305,
            "content": "        \"\"\""
          },
          {
            "line": 306,
            "content": "        提取JavaScript AST特征（简化版本）"
          },
          {
            "line": 307,
            "content": "        \"\"\""
          },
          {
            "line": 308,
            "content": "        line = line_content.strip()"
          },
          {
            "line": 309,
            "content": "        "
          }
        ]
      },
      {
        "line_number": 308,
        "content": "        line = line_content.strip()",
        "ast_features": {
          "node_type": "Assign",
          "depth": 4,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 306,
            "content": "        提取JavaScript AST特征（简化版本）"
          },
          {
            "line": 307,
            "content": "        \"\"\""
          },
          {
            "line": 308,
            "content": "        line = line_content.strip()"
          },
          {
            "line": 309,
            "content": "        "
          },
          {
            "line": 310,
            "content": "        if not line or line.startswith('//') or line.startswith('/*'):"
          }
        ]
      },
      {
        "line_number": 309,
        "content": "        ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 307,
            "content": "        \"\"\""
          },
          {
            "line": 308,
            "content": "        line = line_content.strip()"
          },
          {
            "line": 309,
            "content": "        "
          },
          {
            "line": 310,
            "content": "        if not line or line.startswith('//') or line.startswith('/*'):"
          },
          {
            "line": 311,
            "content": "            return {\"node_type\": \"Comment\", \"depth\": 0, \"children_types\": []}"
          }
        ]
      },
      {
        "line_number": 310,
        "content": "        if not line or line.startswith('//') or line.startswith('/*'):",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 308,
            "content": "        line = line_content.strip()"
          },
          {
            "line": 309,
            "content": "        "
          },
          {
            "line": 310,
            "content": "        if not line or line.startswith('//') or line.startswith('/*'):"
          },
          {
            "line": 311,
            "content": "            return {\"node_type\": \"Comment\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 312,
            "content": "        elif line.startswith('import ') or line.startswith('const ') and 'require(' in line:"
          }
        ]
      },
      {
        "line_number": 311,
        "content": "            return {\"node_type\": \"Comment\", \"depth\": 0, \"children_types\": []}",
        "ast_features": {
          "node_type": "Return",
          "depth": 3,
          "children_types": [
            "Dict"
          ]
        },
        "context_window": [
          {
            "line": 309,
            "content": "        "
          },
          {
            "line": 310,
            "content": "        if not line or line.startswith('//') or line.startswith('/*'):"
          },
          {
            "line": 311,
            "content": "            return {\"node_type\": \"Comment\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 312,
            "content": "        elif line.startswith('import ') or line.startswith('const ') and 'require(' in line:"
          },
          {
            "line": 313,
            "content": "            return {\"node_type\": \"ImportDeclaration\", \"depth\": 0, \"children_types\": []}"
          }
        ]
      },
      {
        "line_number": 312,
        "content": "        elif line.startswith('import ') or line.startswith('const ') and 'require(' in line:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 310,
            "content": "        if not line or line.startswith('//') or line.startswith('/*'):"
          },
          {
            "line": 311,
            "content": "            return {\"node_type\": \"Comment\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 312,
            "content": "        elif line.startswith('import ') or line.startswith('const ') and 'require(' in line:"
          },
          {
            "line": 313,
            "content": "            return {\"node_type\": \"ImportDeclaration\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 314,
            "content": "        elif line.startswith('function '):"
          }
        ]
      },
      {
        "line_number": 313,
        "content": "            return {\"node_type\": \"ImportDeclaration\", \"depth\": 0, \"children_types\": []}",
        "ast_features": {
          "node_type": "Return",
          "depth": 3,
          "children_types": [
            "Dict"
          ]
        },
        "context_window": [
          {
            "line": 311,
            "content": "            return {\"node_type\": \"Comment\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 312,
            "content": "        elif line.startswith('import ') or line.startswith('const ') and 'require(' in line:"
          },
          {
            "line": 313,
            "content": "            return {\"node_type\": \"ImportDeclaration\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 314,
            "content": "        elif line.startswith('function '):"
          },
          {
            "line": 315,
            "content": "            return {\"node_type\": \"FunctionDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\", \"Parameters\"]}"
          }
        ]
      },
      {
        "line_number": 314,
        "content": "        elif line.startswith('function '):",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 312,
            "content": "        elif line.startswith('import ') or line.startswith('const ') and 'require(' in line:"
          },
          {
            "line": 313,
            "content": "            return {\"node_type\": \"ImportDeclaration\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 314,
            "content": "        elif line.startswith('function '):"
          },
          {
            "line": 315,
            "content": "            return {\"node_type\": \"FunctionDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\", \"Parameters\"]}"
          },
          {
            "line": 316,
            "content": "        elif '=>' in line:"
          }
        ]
      },
      {
        "line_number": 315,
        "content": "            return {\"node_type\": \"FunctionDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\", \"Parameters\"]}",
        "ast_features": {
          "node_type": "Return",
          "depth": 4,
          "children_types": [
            "Dict"
          ]
        },
        "context_window": [
          {
            "line": 313,
            "content": "            return {\"node_type\": \"ImportDeclaration\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 314,
            "content": "        elif line.startswith('function '):"
          },
          {
            "line": 315,
            "content": "            return {\"node_type\": \"FunctionDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\", \"Parameters\"]}"
          },
          {
            "line": 316,
            "content": "        elif '=>' in line:"
          },
          {
            "line": 317,
            "content": "            return {\"node_type\": \"ArrowFunctionExpression\", \"depth\": 1, \"children_types\": [\"Parameters\", \"BlockStatement\"]}"
          }
        ]
      },
      {
        "line_number": 316,
        "content": "        elif '=>' in line:",
        "ast_features": {
          "node_type": "Assign",
          "depth": 1,
          "children_types": [
            "Name",
            "Constant"
          ]
        },
        "context_window": [
          {
            "line": 314,
            "content": "        elif line.startswith('function '):"
          },
          {
            "line": 315,
            "content": "            return {\"node_type\": \"FunctionDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\", \"Parameters\"]}"
          },
          {
            "line": 316,
            "content": "        elif '=>' in line:"
          },
          {
            "line": 317,
            "content": "            return {\"node_type\": \"ArrowFunctionExpression\", \"depth\": 1, \"children_types\": [\"Parameters\", \"BlockStatement\"]}"
          },
          {
            "line": 318,
            "content": "        elif 'class ' in line:"
          }
        ]
      },
      {
        "line_number": 317,
        "content": "            return {\"node_type\": \"ArrowFunctionExpression\", \"depth\": 1, \"children_types\": [\"Parameters\", \"BlockStatement\"]}",
        "ast_features": {
          "node_type": "Return",
          "depth": 4,
          "children_types": [
            "Dict"
          ]
        },
        "context_window": [
          {
            "line": 315,
            "content": "            return {\"node_type\": \"FunctionDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\", \"Parameters\"]}"
          },
          {
            "line": 316,
            "content": "        elif '=>' in line:"
          },
          {
            "line": 317,
            "content": "            return {\"node_type\": \"ArrowFunctionExpression\", \"depth\": 1, \"children_types\": [\"Parameters\", \"BlockStatement\"]}"
          },
          {
            "line": 318,
            "content": "        elif 'class ' in line:"
          },
          {
            "line": 319,
            "content": "            return {\"node_type\": \"ClassDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\"]}"
          }
        ]
      },
      {
        "line_number": 318,
        "content": "        elif 'class ' in line:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 316,
            "content": "        elif '=>' in line:"
          },
          {
            "line": 317,
            "content": "            return {\"node_type\": \"ArrowFunctionExpression\", \"depth\": 1, \"children_types\": [\"Parameters\", \"BlockStatement\"]}"
          },
          {
            "line": 318,
            "content": "        elif 'class ' in line:"
          },
          {
            "line": 319,
            "content": "            return {\"node_type\": \"ClassDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\"]}"
          },
          {
            "line": 320,
            "content": "        elif re.search(r'\\b(var|let|const)\\s+\\w+', line):"
          }
        ]
      },
      {
        "line_number": 319,
        "content": "            return {\"node_type\": \"ClassDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\"]}",
        "ast_features": {
          "node_type": "Return",
          "depth": 4,
          "children_types": [
            "Dict"
          ]
        },
        "context_window": [
          {
            "line": 317,
            "content": "            return {\"node_type\": \"ArrowFunctionExpression\", \"depth\": 1, \"children_types\": [\"Parameters\", \"BlockStatement\"]}"
          },
          {
            "line": 318,
            "content": "        elif 'class ' in line:"
          },
          {
            "line": 319,
            "content": "            return {\"node_type\": \"ClassDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\"]}"
          },
          {
            "line": 320,
            "content": "        elif re.search(r'\\b(var|let|const)\\s+\\w+', line):"
          },
          {
            "line": 321,
            "content": "            return {\"node_type\": \"VariableDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\"]}"
          }
        ]
      },
      {
        "line_number": 320,
        "content": "        elif re.search(r'\\b(var|let|const)\\s+\\w+', line):",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 318,
            "content": "        elif 'class ' in line:"
          },
          {
            "line": 319,
            "content": "            return {\"node_type\": \"ClassDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\"]}"
          },
          {
            "line": 320,
            "content": "        elif re.search(r'\\b(var|let|const)\\s+\\w+', line):"
          },
          {
            "line": 321,
            "content": "            return {\"node_type\": \"VariableDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\"]}"
          },
          {
            "line": 322,
            "content": "        else:"
          }
        ]
      },
      {
        "line_number": 321,
        "content": "            return {\"node_type\": \"VariableDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\"]}",
        "ast_features": {
          "node_type": "Return",
          "depth": 4,
          "children_types": [
            "Dict"
          ]
        },
        "context_window": [
          {
            "line": 319,
            "content": "            return {\"node_type\": \"ClassDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\"]}"
          },
          {
            "line": 320,
            "content": "        elif re.search(r'\\b(var|let|const)\\s+\\w+', line):"
          },
          {
            "line": 321,
            "content": "            return {\"node_type\": \"VariableDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\"]}"
          },
          {
            "line": 322,
            "content": "        else:"
          },
          {
            "line": 323,
            "content": "            return {\"node_type\": \"unknown\", \"depth\": 0, \"children_types\": []}"
          }
        ]
      },
      {
        "line_number": 322,
        "content": "        else:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 320,
            "content": "        elif re.search(r'\\b(var|let|const)\\s+\\w+', line):"
          },
          {
            "line": 321,
            "content": "            return {\"node_type\": \"VariableDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\"]}"
          },
          {
            "line": 322,
            "content": "        else:"
          },
          {
            "line": 323,
            "content": "            return {\"node_type\": \"unknown\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 324,
            "content": ""
          }
        ]
      },
      {
        "line_number": 323,
        "content": "            return {\"node_type\": \"unknown\", \"depth\": 0, \"children_types\": []}",
        "ast_features": {
          "node_type": "Return",
          "depth": 3,
          "children_types": [
            "Dict"
          ]
        },
        "context_window": [
          {
            "line": 321,
            "content": "            return {\"node_type\": \"VariableDeclaration\", \"depth\": 1, \"children_types\": [\"Identifier\"]}"
          },
          {
            "line": 322,
            "content": "        else:"
          },
          {
            "line": 323,
            "content": "            return {\"node_type\": \"unknown\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 324,
            "content": ""
          },
          {
            "line": 325,
            "content": "    def _fallback_ast_analysis(self, line_content: str) -> Dict[str, Any]:"
          }
        ]
      },
      {
        "line_number": 324,
        "content": "",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 322,
            "content": "        else:"
          },
          {
            "line": 323,
            "content": "            return {\"node_type\": \"unknown\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 324,
            "content": ""
          },
          {
            "line": 325,
            "content": "    def _fallback_ast_analysis(self, line_content: str) -> Dict[str, Any]:"
          },
          {
            "line": 326,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 325,
        "content": "    def _fallback_ast_analysis(self, line_content: str) -> Dict[str, Any]:",
        "ast_features": {
          "node_type": "FunctionDef",
          "depth": 1,
          "children_types": [
            "Name",
            "arguments"
          ]
        },
        "context_window": [
          {
            "line": 323,
            "content": "            return {\"node_type\": \"unknown\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 324,
            "content": ""
          },
          {
            "line": 325,
            "content": "    def _fallback_ast_analysis(self, line_content: str) -> Dict[str, Any]:"
          },
          {
            "line": 326,
            "content": "        \"\"\""
          },
          {
            "line": 327,
            "content": "        AST解析失败时的降级分析"
          }
        ]
      },
      {
        "line_number": 326,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 324,
            "content": ""
          },
          {
            "line": 325,
            "content": "    def _fallback_ast_analysis(self, line_content: str) -> Dict[str, Any]:"
          },
          {
            "line": 326,
            "content": "        \"\"\""
          },
          {
            "line": 327,
            "content": "        AST解析失败时的降级分析"
          },
          {
            "line": 328,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 327,
        "content": "        AST解析失败时的降级分析",
        "ast_features": {
          "node_type": "Name",
          "depth": 1,
          "children_types": [
            "Load"
          ]
        },
        "context_window": [
          {
            "line": 325,
            "content": "    def _fallback_ast_analysis(self, line_content: str) -> Dict[str, Any]:"
          },
          {
            "line": 326,
            "content": "        \"\"\""
          },
          {
            "line": 327,
            "content": "        AST解析失败时的降级分析"
          },
          {
            "line": 328,
            "content": "        \"\"\""
          },
          {
            "line": 329,
            "content": "        line = line_content.strip()"
          }
        ]
      },
      {
        "line_number": 328,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 326,
            "content": "        \"\"\""
          },
          {
            "line": 327,
            "content": "        AST解析失败时的降级分析"
          },
          {
            "line": 328,
            "content": "        \"\"\""
          },
          {
            "line": 329,
            "content": "        line = line_content.strip()"
          },
          {
            "line": 330,
            "content": "        "
          }
        ]
      },
      {
        "line_number": 329,
        "content": "        line = line_content.strip()",
        "ast_features": {
          "node_type": "Assign",
          "depth": 4,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 327,
            "content": "        AST解析失败时的降级分析"
          },
          {
            "line": 328,
            "content": "        \"\"\""
          },
          {
            "line": 329,
            "content": "        line = line_content.strip()"
          },
          {
            "line": 330,
            "content": "        "
          },
          {
            "line": 331,
            "content": "        # 基础的语法模式匹配"
          }
        ]
      },
      {
        "line_number": 330,
        "content": "        ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 328,
            "content": "        \"\"\""
          },
          {
            "line": 329,
            "content": "        line = line_content.strip()"
          },
          {
            "line": 330,
            "content": "        "
          },
          {
            "line": 331,
            "content": "        # 基础的语法模式匹配"
          },
          {
            "line": 332,
            "content": "        if not line:"
          }
        ]
      },
      {
        "line_number": 331,
        "content": "        # 基础的语法模式匹配",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 329,
            "content": "        line = line_content.strip()"
          },
          {
            "line": 330,
            "content": "        "
          },
          {
            "line": 331,
            "content": "        # 基础的语法模式匹配"
          },
          {
            "line": 332,
            "content": "        if not line:"
          },
          {
            "line": 333,
            "content": "            return {\"node_type\": \"Empty\", \"depth\": 0, \"children_types\": []}"
          }
        ]
      },
      {
        "line_number": 332,
        "content": "        if not line:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 330,
            "content": "        "
          },
          {
            "line": 331,
            "content": "        # 基础的语法模式匹配"
          },
          {
            "line": 332,
            "content": "        if not line:"
          },
          {
            "line": 333,
            "content": "            return {\"node_type\": \"Empty\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 334,
            "content": "        elif line.startswith('#'):"
          }
        ]
      },
      {
        "line_number": 333,
        "content": "            return {\"node_type\": \"Empty\", \"depth\": 0, \"children_types\": []}",
        "ast_features": {
          "node_type": "Return",
          "depth": 3,
          "children_types": [
            "Dict"
          ]
        },
        "context_window": [
          {
            "line": 331,
            "content": "        # 基础的语法模式匹配"
          },
          {
            "line": 332,
            "content": "        if not line:"
          },
          {
            "line": 333,
            "content": "            return {\"node_type\": \"Empty\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 334,
            "content": "        elif line.startswith('#'):"
          },
          {
            "line": 335,
            "content": "            return {\"node_type\": \"Comment\", \"depth\": 0, \"children_types\": []}"
          }
        ]
      },
      {
        "line_number": 334,
        "content": "        elif line.startswith('#'):",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 332,
            "content": "        if not line:"
          },
          {
            "line": 333,
            "content": "            return {\"node_type\": \"Empty\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 334,
            "content": "        elif line.startswith('#'):"
          },
          {
            "line": 335,
            "content": "            return {\"node_type\": \"Comment\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 336,
            "content": "        elif line.startswith('def '):"
          }
        ]
      },
      {
        "line_number": 335,
        "content": "            return {\"node_type\": \"Comment\", \"depth\": 0, \"children_types\": []}",
        "ast_features": {
          "node_type": "Return",
          "depth": 3,
          "children_types": [
            "Dict"
          ]
        },
        "context_window": [
          {
            "line": 333,
            "content": "            return {\"node_type\": \"Empty\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 334,
            "content": "        elif line.startswith('#'):"
          },
          {
            "line": 335,
            "content": "            return {\"node_type\": \"Comment\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 336,
            "content": "        elif line.startswith('def '):"
          },
          {
            "line": 337,
            "content": "            return {\"node_type\": \"FunctionDef\", \"depth\": 1, \"children_types\": [\"Name\", \"arguments\"]}"
          }
        ]
      },
      {
        "line_number": 336,
        "content": "        elif line.startswith('def '):",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 334,
            "content": "        elif line.startswith('#'):"
          },
          {
            "line": 335,
            "content": "            return {\"node_type\": \"Comment\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 336,
            "content": "        elif line.startswith('def '):"
          },
          {
            "line": 337,
            "content": "            return {\"node_type\": \"FunctionDef\", \"depth\": 1, \"children_types\": [\"Name\", \"arguments\"]}"
          },
          {
            "line": 338,
            "content": "        elif line.startswith('class '):"
          }
        ]
      },
      {
        "line_number": 337,
        "content": "            return {\"node_type\": \"FunctionDef\", \"depth\": 1, \"children_types\": [\"Name\", \"arguments\"]}",
        "ast_features": {
          "node_type": "Return",
          "depth": 4,
          "children_types": [
            "Dict"
          ]
        },
        "context_window": [
          {
            "line": 335,
            "content": "            return {\"node_type\": \"Comment\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 336,
            "content": "        elif line.startswith('def '):"
          },
          {
            "line": 337,
            "content": "            return {\"node_type\": \"FunctionDef\", \"depth\": 1, \"children_types\": [\"Name\", \"arguments\"]}"
          },
          {
            "line": 338,
            "content": "        elif line.startswith('class '):"
          },
          {
            "line": 339,
            "content": "            return {\"node_type\": \"ClassDef\", \"depth\": 1, \"children_types\": [\"Name\"]}"
          }
        ]
      },
      {
        "line_number": 338,
        "content": "        elif line.startswith('class '):",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 336,
            "content": "        elif line.startswith('def '):"
          },
          {
            "line": 337,
            "content": "            return {\"node_type\": \"FunctionDef\", \"depth\": 1, \"children_types\": [\"Name\", \"arguments\"]}"
          },
          {
            "line": 338,
            "content": "        elif line.startswith('class '):"
          },
          {
            "line": 339,
            "content": "            return {\"node_type\": \"ClassDef\", \"depth\": 1, \"children_types\": [\"Name\"]}"
          },
          {
            "line": 340,
            "content": "        elif line.startswith('import ') or line.startswith('from '):"
          }
        ]
      },
      {
        "line_number": 339,
        "content": "            return {\"node_type\": \"ClassDef\", \"depth\": 1, \"children_types\": [\"Name\"]}",
        "ast_features": {
          "node_type": "Return",
          "depth": 4,
          "children_types": [
            "Dict"
          ]
        },
        "context_window": [
          {
            "line": 337,
            "content": "            return {\"node_type\": \"FunctionDef\", \"depth\": 1, \"children_types\": [\"Name\", \"arguments\"]}"
          },
          {
            "line": 338,
            "content": "        elif line.startswith('class '):"
          },
          {
            "line": 339,
            "content": "            return {\"node_type\": \"ClassDef\", \"depth\": 1, \"children_types\": [\"Name\"]}"
          },
          {
            "line": 340,
            "content": "        elif line.startswith('import ') or line.startswith('from '):"
          },
          {
            "line": 341,
            "content": "            return {\"node_type\": \"Import\", \"depth\": 0, \"children_types\": [\"alias\"]}"
          }
        ]
      },
      {
        "line_number": 340,
        "content": "        elif line.startswith('import ') or line.startswith('from '):",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 338,
            "content": "        elif line.startswith('class '):"
          },
          {
            "line": 339,
            "content": "            return {\"node_type\": \"ClassDef\", \"depth\": 1, \"children_types\": [\"Name\"]}"
          },
          {
            "line": 340,
            "content": "        elif line.startswith('import ') or line.startswith('from '):"
          },
          {
            "line": 341,
            "content": "            return {\"node_type\": \"Import\", \"depth\": 0, \"children_types\": [\"alias\"]}"
          },
          {
            "line": 342,
            "content": "        elif '=' in line:"
          }
        ]
      },
      {
        "line_number": 341,
        "content": "            return {\"node_type\": \"Import\", \"depth\": 0, \"children_types\": [\"alias\"]}",
        "ast_features": {
          "node_type": "Return",
          "depth": 4,
          "children_types": [
            "Dict"
          ]
        },
        "context_window": [
          {
            "line": 339,
            "content": "            return {\"node_type\": \"ClassDef\", \"depth\": 1, \"children_types\": [\"Name\"]}"
          },
          {
            "line": 340,
            "content": "        elif line.startswith('import ') or line.startswith('from '):"
          },
          {
            "line": 341,
            "content": "            return {\"node_type\": \"Import\", \"depth\": 0, \"children_types\": [\"alias\"]}"
          },
          {
            "line": 342,
            "content": "        elif '=' in line:"
          },
          {
            "line": 343,
            "content": "            return {\"node_type\": \"Assign\", \"depth\": 1, \"children_types\": [\"Name\", \"Constant\"]}"
          }
        ]
      },
      {
        "line_number": 342,
        "content": "        elif '=' in line:",
        "ast_features": {
          "node_type": "Assign",
          "depth": 1,
          "children_types": [
            "Name",
            "Constant"
          ]
        },
        "context_window": [
          {
            "line": 340,
            "content": "        elif line.startswith('import ') or line.startswith('from '):"
          },
          {
            "line": 341,
            "content": "            return {\"node_type\": \"Import\", \"depth\": 0, \"children_types\": [\"alias\"]}"
          },
          {
            "line": 342,
            "content": "        elif '=' in line:"
          },
          {
            "line": 343,
            "content": "            return {\"node_type\": \"Assign\", \"depth\": 1, \"children_types\": [\"Name\", \"Constant\"]}"
          },
          {
            "line": 344,
            "content": "        else:"
          }
        ]
      },
      {
        "line_number": 343,
        "content": "            return {\"node_type\": \"Assign\", \"depth\": 1, \"children_types\": [\"Name\", \"Constant\"]}",
        "ast_features": {
          "node_type": "Return",
          "depth": 4,
          "children_types": [
            "Dict"
          ]
        },
        "context_window": [
          {
            "line": 341,
            "content": "            return {\"node_type\": \"Import\", \"depth\": 0, \"children_types\": [\"alias\"]}"
          },
          {
            "line": 342,
            "content": "        elif '=' in line:"
          },
          {
            "line": 343,
            "content": "            return {\"node_type\": \"Assign\", \"depth\": 1, \"children_types\": [\"Name\", \"Constant\"]}"
          },
          {
            "line": 344,
            "content": "        else:"
          },
          {
            "line": 345,
            "content": "            return {\"node_type\": \"unknown\", \"depth\": 0, \"children_types\": []}"
          }
        ]
      },
      {
        "line_number": 344,
        "content": "        else:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 342,
            "content": "        elif '=' in line:"
          },
          {
            "line": 343,
            "content": "            return {\"node_type\": \"Assign\", \"depth\": 1, \"children_types\": [\"Name\", \"Constant\"]}"
          },
          {
            "line": 344,
            "content": "        else:"
          },
          {
            "line": 345,
            "content": "            return {\"node_type\": \"unknown\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 346,
            "content": ""
          }
        ]
      },
      {
        "line_number": 345,
        "content": "            return {\"node_type\": \"unknown\", \"depth\": 0, \"children_types\": []}",
        "ast_features": {
          "node_type": "Return",
          "depth": 3,
          "children_types": [
            "Dict"
          ]
        },
        "context_window": [
          {
            "line": 343,
            "content": "            return {\"node_type\": \"Assign\", \"depth\": 1, \"children_types\": [\"Name\", \"Constant\"]}"
          },
          {
            "line": 344,
            "content": "        else:"
          },
          {
            "line": 345,
            "content": "            return {\"node_type\": \"unknown\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 346,
            "content": ""
          },
          {
            "line": 347,
            "content": "    def _calculate_ast_depth(self, node) -> int:"
          }
        ]
      },
      {
        "line_number": 346,
        "content": "",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 344,
            "content": "        else:"
          },
          {
            "line": 345,
            "content": "            return {\"node_type\": \"unknown\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 346,
            "content": ""
          },
          {
            "line": 347,
            "content": "    def _calculate_ast_depth(self, node) -> int:"
          },
          {
            "line": 348,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 347,
        "content": "    def _calculate_ast_depth(self, node) -> int:",
        "ast_features": {
          "node_type": "FunctionDef",
          "depth": 1,
          "children_types": [
            "Name",
            "arguments"
          ]
        },
        "context_window": [
          {
            "line": 345,
            "content": "            return {\"node_type\": \"unknown\", \"depth\": 0, \"children_types\": []}"
          },
          {
            "line": 346,
            "content": ""
          },
          {
            "line": 347,
            "content": "    def _calculate_ast_depth(self, node) -> int:"
          },
          {
            "line": 348,
            "content": "        \"\"\""
          },
          {
            "line": 349,
            "content": "        计算AST节点深度"
          }
        ]
      },
      {
        "line_number": 348,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 346,
            "content": ""
          },
          {
            "line": 347,
            "content": "    def _calculate_ast_depth(self, node) -> int:"
          },
          {
            "line": 348,
            "content": "        \"\"\""
          },
          {
            "line": 349,
            "content": "        计算AST节点深度"
          },
          {
            "line": 350,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 349,
        "content": "        计算AST节点深度",
        "ast_features": {
          "node_type": "Name",
          "depth": 1,
          "children_types": [
            "Load"
          ]
        },
        "context_window": [
          {
            "line": 347,
            "content": "    def _calculate_ast_depth(self, node) -> int:"
          },
          {
            "line": 348,
            "content": "        \"\"\""
          },
          {
            "line": 349,
            "content": "        计算AST节点深度"
          },
          {
            "line": 350,
            "content": "        \"\"\""
          },
          {
            "line": 351,
            "content": "        if not hasattr(node, '_fields') or not node._fields:"
          }
        ]
      },
      {
        "line_number": 350,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 348,
            "content": "        \"\"\""
          },
          {
            "line": 349,
            "content": "        计算AST节点深度"
          },
          {
            "line": 350,
            "content": "        \"\"\""
          },
          {
            "line": 351,
            "content": "        if not hasattr(node, '_fields') or not node._fields:"
          },
          {
            "line": 352,
            "content": "            return 0"
          }
        ]
      },
      {
        "line_number": 351,
        "content": "        if not hasattr(node, '_fields') or not node._fields:",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 349,
            "content": "        计算AST节点深度"
          },
          {
            "line": 350,
            "content": "        \"\"\""
          },
          {
            "line": 351,
            "content": "        if not hasattr(node, '_fields') or not node._fields:"
          },
          {
            "line": 352,
            "content": "            return 0"
          },
          {
            "line": 353,
            "content": "        "
          }
        ]
      },
      {
        "line_number": 352,
        "content": "            return 0",
        "ast_features": {
          "node_type": "Return",
          "depth": 2,
          "children_types": [
            "Constant"
          ]
        },
        "context_window": [
          {
            "line": 350,
            "content": "        \"\"\""
          },
          {
            "line": 351,
            "content": "        if not hasattr(node, '_fields') or not node._fields:"
          },
          {
            "line": 352,
            "content": "            return 0"
          },
          {
            "line": 353,
            "content": "        "
          },
          {
            "line": 354,
            "content": "        max_depth = 0"
          }
        ]
      },
      {
        "line_number": 353,
        "content": "        ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 351,
            "content": "        if not hasattr(node, '_fields') or not node._fields:"
          },
          {
            "line": 352,
            "content": "            return 0"
          },
          {
            "line": 353,
            "content": "        "
          },
          {
            "line": 354,
            "content": "        max_depth = 0"
          },
          {
            "line": 355,
            "content": "        for child in ast.iter_child_nodes(node):"
          }
        ]
      },
      {
        "line_number": 354,
        "content": "        max_depth = 0",
        "ast_features": {
          "node_type": "Assign",
          "depth": 2,
          "children_types": [
            "Name",
            "Constant"
          ]
        },
        "context_window": [
          {
            "line": 352,
            "content": "            return 0"
          },
          {
            "line": 353,
            "content": "        "
          },
          {
            "line": 354,
            "content": "        max_depth = 0"
          },
          {
            "line": 355,
            "content": "        for child in ast.iter_child_nodes(node):"
          },
          {
            "line": 356,
            "content": "            child_depth = self._calculate_ast_depth(child)"
          }
        ]
      },
      {
        "line_number": 355,
        "content": "        for child in ast.iter_child_nodes(node):",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 353,
            "content": "        "
          },
          {
            "line": 354,
            "content": "        max_depth = 0"
          },
          {
            "line": 355,
            "content": "        for child in ast.iter_child_nodes(node):"
          },
          {
            "line": 356,
            "content": "            child_depth = self._calculate_ast_depth(child)"
          },
          {
            "line": 357,
            "content": "            max_depth = max(max_depth, child_depth)"
          }
        ]
      },
      {
        "line_number": 356,
        "content": "            child_depth = self._calculate_ast_depth(child)",
        "ast_features": {
          "node_type": "Assign",
          "depth": 4,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 354,
            "content": "        max_depth = 0"
          },
          {
            "line": 355,
            "content": "        for child in ast.iter_child_nodes(node):"
          },
          {
            "line": 356,
            "content": "            child_depth = self._calculate_ast_depth(child)"
          },
          {
            "line": 357,
            "content": "            max_depth = max(max_depth, child_depth)"
          },
          {
            "line": 358,
            "content": "        "
          }
        ]
      },
      {
        "line_number": 357,
        "content": "            max_depth = max(max_depth, child_depth)",
        "ast_features": {
          "node_type": "Assign",
          "depth": 3,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 355,
            "content": "        for child in ast.iter_child_nodes(node):"
          },
          {
            "line": 356,
            "content": "            child_depth = self._calculate_ast_depth(child)"
          },
          {
            "line": 357,
            "content": "            max_depth = max(max_depth, child_depth)"
          },
          {
            "line": 358,
            "content": "        "
          },
          {
            "line": 359,
            "content": "        return max_depth + 1"
          }
        ]
      },
      {
        "line_number": 358,
        "content": "        ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 356,
            "content": "            child_depth = self._calculate_ast_depth(child)"
          },
          {
            "line": 357,
            "content": "            max_depth = max(max_depth, child_depth)"
          },
          {
            "line": 358,
            "content": "        "
          },
          {
            "line": 359,
            "content": "        return max_depth + 1"
          },
          {
            "line": 360,
            "content": ""
          }
        ]
      },
      {
        "line_number": 359,
        "content": "        return max_depth + 1",
        "ast_features": {
          "node_type": "Return",
          "depth": 3,
          "children_types": [
            "BinOp"
          ]
        },
        "context_window": [
          {
            "line": 357,
            "content": "            max_depth = max(max_depth, child_depth)"
          },
          {
            "line": 358,
            "content": "        "
          },
          {
            "line": 359,
            "content": "        return max_depth + 1"
          },
          {
            "line": 360,
            "content": ""
          },
          {
            "line": 361,
            "content": "    def _build_context_window(self, lines: List[str], current_index: int) -> List[Dict[str, Any]]:"
          }
        ]
      },
      {
        "line_number": 360,
        "content": "",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 358,
            "content": "        "
          },
          {
            "line": 359,
            "content": "        return max_depth + 1"
          },
          {
            "line": 360,
            "content": ""
          },
          {
            "line": 361,
            "content": "    def _build_context_window(self, lines: List[str], current_index: int) -> List[Dict[str, Any]]:"
          },
          {
            "line": 362,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 361,
        "content": "    def _build_context_window(self, lines: List[str], current_index: int) -> List[Dict[str, Any]]:",
        "ast_features": {
          "node_type": "FunctionDef",
          "depth": 1,
          "children_types": [
            "Name",
            "arguments"
          ]
        },
        "context_window": [
          {
            "line": 359,
            "content": "        return max_depth + 1"
          },
          {
            "line": 360,
            "content": ""
          },
          {
            "line": 361,
            "content": "    def _build_context_window(self, lines: List[str], current_index: int) -> List[Dict[str, Any]]:"
          },
          {
            "line": 362,
            "content": "        \"\"\""
          },
          {
            "line": 363,
            "content": "        构建上下文窗口"
          }
        ]
      },
      {
        "line_number": 362,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 360,
            "content": ""
          },
          {
            "line": 361,
            "content": "    def _build_context_window(self, lines: List[str], current_index: int) -> List[Dict[str, Any]]:"
          },
          {
            "line": 362,
            "content": "        \"\"\""
          },
          {
            "line": 363,
            "content": "        构建上下文窗口"
          },
          {
            "line": 364,
            "content": "        :param lines: 所有行的列表"
          }
        ]
      },
      {
        "line_number": 363,
        "content": "        构建上下文窗口",
        "ast_features": {
          "node_type": "Name",
          "depth": 1,
          "children_types": [
            "Load"
          ]
        },
        "context_window": [
          {
            "line": 361,
            "content": "    def _build_context_window(self, lines: List[str], current_index: int) -> List[Dict[str, Any]]:"
          },
          {
            "line": 362,
            "content": "        \"\"\""
          },
          {
            "line": 363,
            "content": "        构建上下文窗口"
          },
          {
            "line": 364,
            "content": "        :param lines: 所有行的列表"
          },
          {
            "line": 365,
            "content": "        :param current_index: 当前行索引"
          }
        ]
      },
      {
        "line_number": 364,
        "content": "        :param lines: 所有行的列表",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 362,
            "content": "        \"\"\""
          },
          {
            "line": 363,
            "content": "        构建上下文窗口"
          },
          {
            "line": 364,
            "content": "        :param lines: 所有行的列表"
          },
          {
            "line": 365,
            "content": "        :param current_index: 当前行索引"
          },
          {
            "line": 366,
            "content": "        :return: 上下文窗口列表"
          }
        ]
      },
      {
        "line_number": 365,
        "content": "        :param current_index: 当前行索引",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 363,
            "content": "        构建上下文窗口"
          },
          {
            "line": 364,
            "content": "        :param lines: 所有行的列表"
          },
          {
            "line": 365,
            "content": "        :param current_index: 当前行索引"
          },
          {
            "line": 366,
            "content": "        :return: 上下文窗口列表"
          },
          {
            "line": 367,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 366,
        "content": "        :return: 上下文窗口列表",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 364,
            "content": "        :param lines: 所有行的列表"
          },
          {
            "line": 365,
            "content": "        :param current_index: 当前行索引"
          },
          {
            "line": 366,
            "content": "        :return: 上下文窗口列表"
          },
          {
            "line": 367,
            "content": "        \"\"\""
          },
          {
            "line": 368,
            "content": "        context_window = []"
          }
        ]
      },
      {
        "line_number": 367,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 365,
            "content": "        :param current_index: 当前行索引"
          },
          {
            "line": 366,
            "content": "        :return: 上下文窗口列表"
          },
          {
            "line": 367,
            "content": "        \"\"\""
          },
          {
            "line": 368,
            "content": "        context_window = []"
          },
          {
            "line": 369,
            "content": "        "
          }
        ]
      },
      {
        "line_number": 368,
        "content": "        context_window = []",
        "ast_features": {
          "node_type": "Assign",
          "depth": 2,
          "children_types": [
            "Name",
            "List"
          ]
        },
        "context_window": [
          {
            "line": 366,
            "content": "        :return: 上下文窗口列表"
          },
          {
            "line": 367,
            "content": "        \"\"\""
          },
          {
            "line": 368,
            "content": "        context_window = []"
          },
          {
            "line": 369,
            "content": "        "
          },
          {
            "line": 370,
            "content": "        start_index = max(0, current_index - self.context_window_size)"
          }
        ]
      },
      {
        "line_number": 369,
        "content": "        ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 367,
            "content": "        \"\"\""
          },
          {
            "line": 368,
            "content": "        context_window = []"
          },
          {
            "line": 369,
            "content": "        "
          },
          {
            "line": 370,
            "content": "        start_index = max(0, current_index - self.context_window_size)"
          },
          {
            "line": 371,
            "content": "        end_index = min(len(lines), current_index + self.context_window_size + 1)"
          }
        ]
      },
      {
        "line_number": 370,
        "content": "        start_index = max(0, current_index - self.context_window_size)",
        "ast_features": {
          "node_type": "Assign",
          "depth": 5,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 368,
            "content": "        context_window = []"
          },
          {
            "line": 369,
            "content": "        "
          },
          {
            "line": 370,
            "content": "        start_index = max(0, current_index - self.context_window_size)"
          },
          {
            "line": 371,
            "content": "        end_index = min(len(lines), current_index + self.context_window_size + 1)"
          },
          {
            "line": 372,
            "content": "        "
          }
        ]
      },
      {
        "line_number": 371,
        "content": "        end_index = min(len(lines), current_index + self.context_window_size + 1)",
        "ast_features": {
          "node_type": "Assign",
          "depth": 6,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 369,
            "content": "        "
          },
          {
            "line": 370,
            "content": "        start_index = max(0, current_index - self.context_window_size)"
          },
          {
            "line": 371,
            "content": "        end_index = min(len(lines), current_index + self.context_window_size + 1)"
          },
          {
            "line": 372,
            "content": "        "
          },
          {
            "line": 373,
            "content": "        for i in range(start_index, end_index):"
          }
        ]
      },
      {
        "line_number": 372,
        "content": "        ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 370,
            "content": "        start_index = max(0, current_index - self.context_window_size)"
          },
          {
            "line": 371,
            "content": "        end_index = min(len(lines), current_index + self.context_window_size + 1)"
          },
          {
            "line": 372,
            "content": "        "
          },
          {
            "line": 373,
            "content": "        for i in range(start_index, end_index):"
          },
          {
            "line": 374,
            "content": "            context_window.append({"
          }
        ]
      },
      {
        "line_number": 373,
        "content": "        for i in range(start_index, end_index):",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 371,
            "content": "        end_index = min(len(lines), current_index + self.context_window_size + 1)"
          },
          {
            "line": 372,
            "content": "        "
          },
          {
            "line": 373,
            "content": "        for i in range(start_index, end_index):"
          },
          {
            "line": 374,
            "content": "            context_window.append({"
          },
          {
            "line": 375,
            "content": "                \"line\": i + 1,"
          }
        ]
      },
      {
        "line_number": 374,
        "content": "            context_window.append({",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 372,
            "content": "        "
          },
          {
            "line": 373,
            "content": "        for i in range(start_index, end_index):"
          },
          {
            "line": 374,
            "content": "            context_window.append({"
          },
          {
            "line": 375,
            "content": "                \"line\": i + 1,"
          },
          {
            "line": 376,
            "content": "                \"content\": lines[i]"
          }
        ]
      },
      {
        "line_number": 375,
        "content": "                \"line\": i + 1,",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 373,
            "content": "        for i in range(start_index, end_index):"
          },
          {
            "line": 374,
            "content": "            context_window.append({"
          },
          {
            "line": 375,
            "content": "                \"line\": i + 1,"
          },
          {
            "line": 376,
            "content": "                \"content\": lines[i]"
          },
          {
            "line": 377,
            "content": "            })"
          }
        ]
      },
      {
        "line_number": 376,
        "content": "                \"content\": lines[i]",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 374,
            "content": "            context_window.append({"
          },
          {
            "line": 375,
            "content": "                \"line\": i + 1,"
          },
          {
            "line": 376,
            "content": "                \"content\": lines[i]"
          },
          {
            "line": 377,
            "content": "            })"
          },
          {
            "line": 378,
            "content": "        "
          }
        ]
      },
      {
        "line_number": 377,
        "content": "            })",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 375,
            "content": "                \"line\": i + 1,"
          },
          {
            "line": 376,
            "content": "                \"content\": lines[i]"
          },
          {
            "line": 377,
            "content": "            })"
          },
          {
            "line": 378,
            "content": "        "
          },
          {
            "line": 379,
            "content": "        return context_window"
          }
        ]
      },
      {
        "line_number": 378,
        "content": "        ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 376,
            "content": "                \"content\": lines[i]"
          },
          {
            "line": 377,
            "content": "            })"
          },
          {
            "line": 378,
            "content": "        "
          },
          {
            "line": 379,
            "content": "        return context_window"
          },
          {
            "line": 380,
            "content": ""
          }
        ]
      },
      {
        "line_number": 379,
        "content": "        return context_window",
        "ast_features": {
          "node_type": "Return",
          "depth": 2,
          "children_types": [
            "Name"
          ]
        },
        "context_window": [
          {
            "line": 377,
            "content": "            })"
          },
          {
            "line": 378,
            "content": "        "
          },
          {
            "line": 379,
            "content": "        return context_window"
          },
          {
            "line": 380,
            "content": ""
          },
          {
            "line": 381,
            "content": "    def _create_fallback_result(self, input_data: Dict[str, str]) -> Dict[str, Any]:"
          }
        ]
      },
      {
        "line_number": 380,
        "content": "",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 378,
            "content": "        "
          },
          {
            "line": 379,
            "content": "        return context_window"
          },
          {
            "line": 380,
            "content": ""
          },
          {
            "line": 381,
            "content": "    def _create_fallback_result(self, input_data: Dict[str, str]) -> Dict[str, Any]:"
          },
          {
            "line": 382,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 381,
        "content": "    def _create_fallback_result(self, input_data: Dict[str, str]) -> Dict[str, Any]:",
        "ast_features": {
          "node_type": "FunctionDef",
          "depth": 1,
          "children_types": [
            "Name",
            "arguments"
          ]
        },
        "context_window": [
          {
            "line": 379,
            "content": "        return context_window"
          },
          {
            "line": 380,
            "content": ""
          },
          {
            "line": 381,
            "content": "    def _create_fallback_result(self, input_data: Dict[str, str]) -> Dict[str, Any]:"
          },
          {
            "line": 382,
            "content": "        \"\"\""
          },
          {
            "line": 383,
            "content": "        创建降级解析结果"
          }
        ]
      },
      {
        "line_number": 382,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 380,
            "content": ""
          },
          {
            "line": 381,
            "content": "    def _create_fallback_result(self, input_data: Dict[str, str]) -> Dict[str, Any]:"
          },
          {
            "line": 382,
            "content": "        \"\"\""
          },
          {
            "line": 383,
            "content": "        创建降级解析结果"
          },
          {
            "line": 384,
            "content": "        \"\"\""
          }
        ]
      },
      {
        "line_number": 383,
        "content": "        创建降级解析结果",
        "ast_features": {
          "node_type": "Name",
          "depth": 1,
          "children_types": [
            "Load"
          ]
        },
        "context_window": [
          {
            "line": 381,
            "content": "    def _create_fallback_result(self, input_data: Dict[str, str]) -> Dict[str, Any]:"
          },
          {
            "line": 382,
            "content": "        \"\"\""
          },
          {
            "line": 383,
            "content": "        创建降级解析结果"
          },
          {
            "line": 384,
            "content": "        \"\"\""
          },
          {
            "line": 385,
            "content": "        content = input_data.get(\"content\", \"\")"
          }
        ]
      },
      {
        "line_number": 384,
        "content": "        \"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 382,
            "content": "        \"\"\""
          },
          {
            "line": 383,
            "content": "        创建降级解析结果"
          },
          {
            "line": 384,
            "content": "        \"\"\""
          },
          {
            "line": 385,
            "content": "        content = input_data.get(\"content\", \"\")"
          },
          {
            "line": 386,
            "content": "        file_path = input_data.get(\"file_path\", \"\")"
          }
        ]
      },
      {
        "line_number": 385,
        "content": "        content = input_data.get(\"content\", \"\")",
        "ast_features": {
          "node_type": "Assign",
          "depth": 4,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 383,
            "content": "        创建降级解析结果"
          },
          {
            "line": 384,
            "content": "        \"\"\""
          },
          {
            "line": 385,
            "content": "        content = input_data.get(\"content\", \"\")"
          },
          {
            "line": 386,
            "content": "        file_path = input_data.get(\"file_path\", \"\")"
          },
          {
            "line": 387,
            "content": "        lines = content.split('\\n')"
          }
        ]
      },
      {
        "line_number": 386,
        "content": "        file_path = input_data.get(\"file_path\", \"\")",
        "ast_features": {
          "node_type": "Assign",
          "depth": 4,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 384,
            "content": "        \"\"\""
          },
          {
            "line": 385,
            "content": "        content = input_data.get(\"content\", \"\")"
          },
          {
            "line": 386,
            "content": "        file_path = input_data.get(\"file_path\", \"\")"
          },
          {
            "line": 387,
            "content": "        lines = content.split('\\n')"
          },
          {
            "line": 388,
            "content": "        "
          }
        ]
      },
      {
        "line_number": 387,
        "content": "        lines = content.split('\\n')",
        "ast_features": {
          "node_type": "Assign",
          "depth": 4,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 385,
            "content": "        content = input_data.get(\"content\", \"\")"
          },
          {
            "line": 386,
            "content": "        file_path = input_data.get(\"file_path\", \"\")"
          },
          {
            "line": 387,
            "content": "        lines = content.split('\\n')"
          },
          {
            "line": 388,
            "content": "        "
          },
          {
            "line": 389,
            "content": "        lines_data = []"
          }
        ]
      },
      {
        "line_number": 388,
        "content": "        ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 386,
            "content": "        file_path = input_data.get(\"file_path\", \"\")"
          },
          {
            "line": 387,
            "content": "        lines = content.split('\\n')"
          },
          {
            "line": 388,
            "content": "        "
          },
          {
            "line": 389,
            "content": "        lines_data = []"
          },
          {
            "line": 390,
            "content": "        for i, line in enumerate(lines):"
          }
        ]
      },
      {
        "line_number": 389,
        "content": "        lines_data = []",
        "ast_features": {
          "node_type": "Assign",
          "depth": 2,
          "children_types": [
            "Name",
            "List"
          ]
        },
        "context_window": [
          {
            "line": 387,
            "content": "        lines = content.split('\\n')"
          },
          {
            "line": 388,
            "content": "        "
          },
          {
            "line": 389,
            "content": "        lines_data = []"
          },
          {
            "line": 390,
            "content": "        for i, line in enumerate(lines):"
          },
          {
            "line": 391,
            "content": "            lines_data.append({"
          }
        ]
      },
      {
        "line_number": 390,
        "content": "        for i, line in enumerate(lines):",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 388,
            "content": "        "
          },
          {
            "line": 389,
            "content": "        lines_data = []"
          },
          {
            "line": 390,
            "content": "        for i, line in enumerate(lines):"
          },
          {
            "line": 391,
            "content": "            lines_data.append({"
          },
          {
            "line": 392,
            "content": "                \"line_number\": i + 1,"
          }
        ]
      },
      {
        "line_number": 391,
        "content": "            lines_data.append({",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 389,
            "content": "        lines_data = []"
          },
          {
            "line": 390,
            "content": "        for i, line in enumerate(lines):"
          },
          {
            "line": 391,
            "content": "            lines_data.append({"
          },
          {
            "line": 392,
            "content": "                \"line_number\": i + 1,"
          },
          {
            "line": 393,
            "content": "                \"content\": line,"
          }
        ]
      },
      {
        "line_number": 392,
        "content": "                \"line_number\": i + 1,",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 390,
            "content": "        for i, line in enumerate(lines):"
          },
          {
            "line": 391,
            "content": "            lines_data.append({"
          },
          {
            "line": 392,
            "content": "                \"line_number\": i + 1,"
          },
          {
            "line": 393,
            "content": "                \"content\": line,"
          },
          {
            "line": 394,
            "content": "                \"ast_features\": {"
          }
        ]
      },
      {
        "line_number": 393,
        "content": "                \"content\": line,",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 391,
            "content": "            lines_data.append({"
          },
          {
            "line": 392,
            "content": "                \"line_number\": i + 1,"
          },
          {
            "line": 393,
            "content": "                \"content\": line,"
          },
          {
            "line": 394,
            "content": "                \"ast_features\": {"
          },
          {
            "line": 395,
            "content": "                    \"node_type\": \"unknown\","
          }
        ]
      },
      {
        "line_number": 394,
        "content": "                \"ast_features\": {",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 392,
            "content": "                \"line_number\": i + 1,"
          },
          {
            "line": 393,
            "content": "                \"content\": line,"
          },
          {
            "line": 394,
            "content": "                \"ast_features\": {"
          },
          {
            "line": 395,
            "content": "                    \"node_type\": \"unknown\","
          },
          {
            "line": 396,
            "content": "                    \"depth\": 0,"
          }
        ]
      },
      {
        "line_number": 395,
        "content": "                    \"node_type\": \"unknown\",",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 393,
            "content": "                \"content\": line,"
          },
          {
            "line": 394,
            "content": "                \"ast_features\": {"
          },
          {
            "line": 395,
            "content": "                    \"node_type\": \"unknown\","
          },
          {
            "line": 396,
            "content": "                    \"depth\": 0,"
          },
          {
            "line": 397,
            "content": "                    \"children_types\": []"
          }
        ]
      },
      {
        "line_number": 396,
        "content": "                    \"depth\": 0,",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 394,
            "content": "                \"ast_features\": {"
          },
          {
            "line": 395,
            "content": "                    \"node_type\": \"unknown\","
          },
          {
            "line": 396,
            "content": "                    \"depth\": 0,"
          },
          {
            "line": 397,
            "content": "                    \"children_types\": []"
          },
          {
            "line": 398,
            "content": "                },"
          }
        ]
      },
      {
        "line_number": 397,
        "content": "                    \"children_types\": []",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 395,
            "content": "                    \"node_type\": \"unknown\","
          },
          {
            "line": 396,
            "content": "                    \"depth\": 0,"
          },
          {
            "line": 397,
            "content": "                    \"children_types\": []"
          },
          {
            "line": 398,
            "content": "                },"
          },
          {
            "line": 399,
            "content": "                \"context_window\": [{\"line\": i + 1, \"content\": line}]"
          }
        ]
      },
      {
        "line_number": 398,
        "content": "                },",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 396,
            "content": "                    \"depth\": 0,"
          },
          {
            "line": 397,
            "content": "                    \"children_types\": []"
          },
          {
            "line": 398,
            "content": "                },"
          },
          {
            "line": 399,
            "content": "                \"context_window\": [{\"line\": i + 1, \"content\": line}]"
          },
          {
            "line": 400,
            "content": "            })"
          }
        ]
      },
      {
        "line_number": 399,
        "content": "                \"context_window\": [{\"line\": i + 1, \"content\": line}]",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 397,
            "content": "                    \"children_types\": []"
          },
          {
            "line": 398,
            "content": "                },"
          },
          {
            "line": 399,
            "content": "                \"context_window\": [{\"line\": i + 1, \"content\": line}]"
          },
          {
            "line": 400,
            "content": "            })"
          },
          {
            "line": 401,
            "content": "        "
          }
        ]
      },
      {
        "line_number": 400,
        "content": "            })",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 398,
            "content": "                },"
          },
          {
            "line": 399,
            "content": "                \"context_window\": [{\"line\": i + 1, \"content\": line}]"
          },
          {
            "line": 400,
            "content": "            })"
          },
          {
            "line": 401,
            "content": "        "
          },
          {
            "line": 402,
            "content": "        return {"
          }
        ]
      },
      {
        "line_number": 401,
        "content": "        ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 399,
            "content": "                \"context_window\": [{\"line\": i + 1, \"content\": line}]"
          },
          {
            "line": 400,
            "content": "            })"
          },
          {
            "line": 401,
            "content": "        "
          },
          {
            "line": 402,
            "content": "        return {"
          },
          {
            "line": 403,
            "content": "            \"language\": \"unknown\","
          }
        ]
      },
      {
        "line_number": 402,
        "content": "        return {",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 400,
            "content": "            })"
          },
          {
            "line": 401,
            "content": "        "
          },
          {
            "line": 402,
            "content": "        return {"
          },
          {
            "line": 403,
            "content": "            \"language\": \"unknown\","
          },
          {
            "line": 404,
            "content": "            \"file_context\": {"
          }
        ]
      },
      {
        "line_number": 403,
        "content": "            \"language\": \"unknown\",",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 401,
            "content": "        "
          },
          {
            "line": 402,
            "content": "        return {"
          },
          {
            "line": 403,
            "content": "            \"language\": \"unknown\","
          },
          {
            "line": 404,
            "content": "            \"file_context\": {"
          },
          {
            "line": 405,
            "content": "                \"file_name\": os.path.basename(file_path) if file_path else \"unknown\","
          }
        ]
      },
      {
        "line_number": 404,
        "content": "            \"file_context\": {",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 402,
            "content": "        return {"
          },
          {
            "line": 403,
            "content": "            \"language\": \"unknown\","
          },
          {
            "line": 404,
            "content": "            \"file_context\": {"
          },
          {
            "line": 405,
            "content": "                \"file_name\": os.path.basename(file_path) if file_path else \"unknown\","
          },
          {
            "line": 406,
            "content": "                \"imports\": [],"
          }
        ]
      },
      {
        "line_number": 405,
        "content": "                \"file_name\": os.path.basename(file_path) if file_path else \"unknown\",",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 403,
            "content": "            \"language\": \"unknown\","
          },
          {
            "line": 404,
            "content": "            \"file_context\": {"
          },
          {
            "line": 405,
            "content": "                \"file_name\": os.path.basename(file_path) if file_path else \"unknown\","
          },
          {
            "line": 406,
            "content": "                \"imports\": [],"
          },
          {
            "line": 407,
            "content": "                \"class_name\": None,"
          }
        ]
      },
      {
        "line_number": 406,
        "content": "                \"imports\": [],",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 404,
            "content": "            \"file_context\": {"
          },
          {
            "line": 405,
            "content": "                \"file_name\": os.path.basename(file_path) if file_path else \"unknown\","
          },
          {
            "line": 406,
            "content": "                \"imports\": [],"
          },
          {
            "line": 407,
            "content": "                \"class_name\": None,"
          },
          {
            "line": 408,
            "content": "                \"function_name\": None"
          }
        ]
      },
      {
        "line_number": 407,
        "content": "                \"class_name\": None,",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 405,
            "content": "                \"file_name\": os.path.basename(file_path) if file_path else \"unknown\","
          },
          {
            "line": 406,
            "content": "                \"imports\": [],"
          },
          {
            "line": 407,
            "content": "                \"class_name\": None,"
          },
          {
            "line": 408,
            "content": "                \"function_name\": None"
          },
          {
            "line": 409,
            "content": "            },"
          }
        ]
      },
      {
        "line_number": 408,
        "content": "                \"function_name\": None",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 406,
            "content": "                \"imports\": [],"
          },
          {
            "line": 407,
            "content": "                \"class_name\": None,"
          },
          {
            "line": 408,
            "content": "                \"function_name\": None"
          },
          {
            "line": 409,
            "content": "            },"
          },
          {
            "line": 410,
            "content": "            \"lines\": lines_data"
          }
        ]
      },
      {
        "line_number": 409,
        "content": "            },",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 407,
            "content": "                \"class_name\": None,"
          },
          {
            "line": 408,
            "content": "                \"function_name\": None"
          },
          {
            "line": 409,
            "content": "            },"
          },
          {
            "line": 410,
            "content": "            \"lines\": lines_data"
          },
          {
            "line": 411,
            "content": "        }"
          }
        ]
      },
      {
        "line_number": 410,
        "content": "            \"lines\": lines_data",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 408,
            "content": "                \"function_name\": None"
          },
          {
            "line": 409,
            "content": "            },"
          },
          {
            "line": 410,
            "content": "            \"lines\": lines_data"
          },
          {
            "line": 411,
            "content": "        }"
          },
          {
            "line": 412,
            "content": ""
          }
        ]
      },
      {
        "line_number": 411,
        "content": "        }",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 409,
            "content": "            },"
          },
          {
            "line": 410,
            "content": "            \"lines\": lines_data"
          },
          {
            "line": 411,
            "content": "        }"
          },
          {
            "line": 412,
            "content": ""
          },
          {
            "line": 413,
            "content": ""
          }
        ]
      },
      {
        "line_number": 412,
        "content": "",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 410,
            "content": "            \"lines\": lines_data"
          },
          {
            "line": 411,
            "content": "        }"
          },
          {
            "line": 412,
            "content": ""
          },
          {
            "line": 413,
            "content": ""
          },
          {
            "line": 414,
            "content": "# 使用示例"
          }
        ]
      },
      {
        "line_number": 413,
        "content": "",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 411,
            "content": "        }"
          },
          {
            "line": 412,
            "content": ""
          },
          {
            "line": 413,
            "content": ""
          },
          {
            "line": 414,
            "content": "# 使用示例"
          },
          {
            "line": 415,
            "content": "if __name__ == \"__main__\":"
          }
        ]
      },
      {
        "line_number": 414,
        "content": "# 使用示例",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 412,
            "content": ""
          },
          {
            "line": 413,
            "content": ""
          },
          {
            "line": 414,
            "content": "# 使用示例"
          },
          {
            "line": 415,
            "content": "if __name__ == \"__main__\":"
          },
          {
            "line": 416,
            "content": "    parser = CodeFileParser()"
          }
        ]
      },
      {
        "line_number": 415,
        "content": "if __name__ == \"__main__\":",
        "ast_features": {
          "node_type": "Assign",
          "depth": 1,
          "children_types": [
            "Name",
            "Constant"
          ]
        },
        "context_window": [
          {
            "line": 413,
            "content": ""
          },
          {
            "line": 414,
            "content": "# 使用示例"
          },
          {
            "line": 415,
            "content": "if __name__ == \"__main__\":"
          },
          {
            "line": 416,
            "content": "    parser = CodeFileParser()"
          },
          {
            "line": 417,
            "content": "    "
          }
        ]
      },
      {
        "line_number": 416,
        "content": "    parser = CodeFileParser()",
        "ast_features": {
          "node_type": "Assign",
          "depth": 3,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 414,
            "content": "# 使用示例"
          },
          {
            "line": 415,
            "content": "if __name__ == \"__main__\":"
          },
          {
            "line": 416,
            "content": "    parser = CodeFileParser()"
          },
          {
            "line": 417,
            "content": "    "
          },
          {
            "line": 418,
            "content": "    # 测试Python代码"
          }
        ]
      },
      {
        "line_number": 417,
        "content": "    ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 415,
            "content": "if __name__ == \"__main__\":"
          },
          {
            "line": 416,
            "content": "    parser = CodeFileParser()"
          },
          {
            "line": 417,
            "content": "    "
          },
          {
            "line": 418,
            "content": "    # 测试Python代码"
          },
          {
            "line": 419,
            "content": "    test_input = {"
          }
        ]
      },
      {
        "line_number": 418,
        "content": "    # 测试Python代码",
        "ast_features": {
          "node_type": "Comment",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 416,
            "content": "    parser = CodeFileParser()"
          },
          {
            "line": 417,
            "content": "    "
          },
          {
            "line": 418,
            "content": "    # 测试Python代码"
          },
          {
            "line": 419,
            "content": "    test_input = {"
          },
          {
            "line": 420,
            "content": "        \"file_path\": \"/projects/math_utils.py\","
          }
        ]
      },
      {
        "line_number": 419,
        "content": "    test_input = {",
        "ast_features": {
          "node_type": "Assign",
          "depth": 1,
          "children_types": [
            "Name",
            "Constant"
          ]
        },
        "context_window": [
          {
            "line": 417,
            "content": "    "
          },
          {
            "line": 418,
            "content": "    # 测试Python代码"
          },
          {
            "line": 419,
            "content": "    test_input = {"
          },
          {
            "line": 420,
            "content": "        \"file_path\": \"/projects/math_utils.py\","
          },
          {
            "line": 421,
            "content": "        \"content\": \"\"\"import math"
          }
        ]
      },
      {
        "line_number": 420,
        "content": "        \"file_path\": \"/projects/math_utils.py\",",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 418,
            "content": "    # 测试Python代码"
          },
          {
            "line": 419,
            "content": "    test_input = {"
          },
          {
            "line": 420,
            "content": "        \"file_path\": \"/projects/math_utils.py\","
          },
          {
            "line": 421,
            "content": "        \"content\": \"\"\"import math"
          },
          {
            "line": 422,
            "content": ""
          }
        ]
      },
      {
        "line_number": 421,
        "content": "        \"content\": \"\"\"import math",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 419,
            "content": "    test_input = {"
          },
          {
            "line": 420,
            "content": "        \"file_path\": \"/projects/math_utils.py\","
          },
          {
            "line": 421,
            "content": "        \"content\": \"\"\"import math"
          },
          {
            "line": 422,
            "content": ""
          },
          {
            "line": 423,
            "content": "def factorial(n):"
          }
        ]
      },
      {
        "line_number": 422,
        "content": "",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 420,
            "content": "        \"file_path\": \"/projects/math_utils.py\","
          },
          {
            "line": 421,
            "content": "        \"content\": \"\"\"import math"
          },
          {
            "line": 422,
            "content": ""
          },
          {
            "line": 423,
            "content": "def factorial(n):"
          },
          {
            "line": 424,
            "content": "    '''计算阶乘'''"
          }
        ]
      },
      {
        "line_number": 423,
        "content": "def factorial(n):",
        "ast_features": {
          "node_type": "FunctionDef",
          "depth": 1,
          "children_types": [
            "Name",
            "arguments"
          ]
        },
        "context_window": [
          {
            "line": 421,
            "content": "        \"content\": \"\"\"import math"
          },
          {
            "line": 422,
            "content": ""
          },
          {
            "line": 423,
            "content": "def factorial(n):"
          },
          {
            "line": 424,
            "content": "    '''计算阶乘'''"
          },
          {
            "line": 425,
            "content": "    return 1 if n <= 1 else n * factorial(n-1)"
          }
        ]
      },
      {
        "line_number": 424,
        "content": "    '''计算阶乘'''",
        "ast_features": {
          "node_type": "Constant",
          "depth": 1,
          "children_types": []
        },
        "context_window": [
          {
            "line": 422,
            "content": ""
          },
          {
            "line": 423,
            "content": "def factorial(n):"
          },
          {
            "line": 424,
            "content": "    '''计算阶乘'''"
          },
          {
            "line": 425,
            "content": "    return 1 if n <= 1 else n * factorial(n-1)"
          },
          {
            "line": 426,
            "content": ""
          }
        ]
      },
      {
        "line_number": 425,
        "content": "    return 1 if n <= 1 else n * factorial(n-1)",
        "ast_features": {
          "node_type": "Return",
          "depth": 6,
          "children_types": [
            "IfExp"
          ]
        },
        "context_window": [
          {
            "line": 423,
            "content": "def factorial(n):"
          },
          {
            "line": 424,
            "content": "    '''计算阶乘'''"
          },
          {
            "line": 425,
            "content": "    return 1 if n <= 1 else n * factorial(n-1)"
          },
          {
            "line": 426,
            "content": ""
          },
          {
            "line": 427,
            "content": "class MathUtils:"
          }
        ]
      },
      {
        "line_number": 426,
        "content": "",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 424,
            "content": "    '''计算阶乘'''"
          },
          {
            "line": 425,
            "content": "    return 1 if n <= 1 else n * factorial(n-1)"
          },
          {
            "line": 426,
            "content": ""
          },
          {
            "line": 427,
            "content": "class MathUtils:"
          },
          {
            "line": 428,
            "content": "    def __init__(self):"
          }
        ]
      },
      {
        "line_number": 427,
        "content": "class MathUtils:",
        "ast_features": {
          "node_type": "ClassDef",
          "depth": 1,
          "children_types": [
            "Name"
          ]
        },
        "context_window": [
          {
            "line": 425,
            "content": "    return 1 if n <= 1 else n * factorial(n-1)"
          },
          {
            "line": 426,
            "content": ""
          },
          {
            "line": 427,
            "content": "class MathUtils:"
          },
          {
            "line": 428,
            "content": "    def __init__(self):"
          },
          {
            "line": 429,
            "content": "        self.pi = math.pi"
          }
        ]
      },
      {
        "line_number": 428,
        "content": "    def __init__(self):",
        "ast_features": {
          "node_type": "FunctionDef",
          "depth": 1,
          "children_types": [
            "Name",
            "arguments"
          ]
        },
        "context_window": [
          {
            "line": 426,
            "content": ""
          },
          {
            "line": 427,
            "content": "class MathUtils:"
          },
          {
            "line": 428,
            "content": "    def __init__(self):"
          },
          {
            "line": 429,
            "content": "        self.pi = math.pi"
          },
          {
            "line": 430,
            "content": "    "
          }
        ]
      },
      {
        "line_number": 429,
        "content": "        self.pi = math.pi",
        "ast_features": {
          "node_type": "Assign",
          "depth": 3,
          "children_types": [
            "Attribute",
            "Attribute"
          ]
        },
        "context_window": [
          {
            "line": 427,
            "content": "class MathUtils:"
          },
          {
            "line": 428,
            "content": "    def __init__(self):"
          },
          {
            "line": 429,
            "content": "        self.pi = math.pi"
          },
          {
            "line": 430,
            "content": "    "
          },
          {
            "line": 431,
            "content": "    def circle_area(self, radius):"
          }
        ]
      },
      {
        "line_number": 430,
        "content": "    ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 428,
            "content": "    def __init__(self):"
          },
          {
            "line": 429,
            "content": "        self.pi = math.pi"
          },
          {
            "line": 430,
            "content": "    "
          },
          {
            "line": 431,
            "content": "    def circle_area(self, radius):"
          },
          {
            "line": 432,
            "content": "        return self.pi * radius ** 2"
          }
        ]
      },
      {
        "line_number": 431,
        "content": "    def circle_area(self, radius):",
        "ast_features": {
          "node_type": "FunctionDef",
          "depth": 1,
          "children_types": [
            "Name",
            "arguments"
          ]
        },
        "context_window": [
          {
            "line": 429,
            "content": "        self.pi = math.pi"
          },
          {
            "line": 430,
            "content": "    "
          },
          {
            "line": 431,
            "content": "    def circle_area(self, radius):"
          },
          {
            "line": 432,
            "content": "        return self.pi * radius ** 2"
          },
          {
            "line": 433,
            "content": "\"\"\""
          }
        ]
      },
      {
        "line_number": 432,
        "content": "        return self.pi * radius ** 2",
        "ast_features": {
          "node_type": "Return",
          "depth": 4,
          "children_types": [
            "BinOp"
          ]
        },
        "context_window": [
          {
            "line": 430,
            "content": "    "
          },
          {
            "line": 431,
            "content": "    def circle_area(self, radius):"
          },
          {
            "line": 432,
            "content": "        return self.pi * radius ** 2"
          },
          {
            "line": 433,
            "content": "\"\"\""
          },
          {
            "line": 434,
            "content": "    }"
          }
        ]
      },
      {
        "line_number": 433,
        "content": "\"\"\"",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 431,
            "content": "    def circle_area(self, radius):"
          },
          {
            "line": 432,
            "content": "        return self.pi * radius ** 2"
          },
          {
            "line": 433,
            "content": "\"\"\""
          },
          {
            "line": 434,
            "content": "    }"
          },
          {
            "line": 435,
            "content": "    "
          }
        ]
      },
      {
        "line_number": 434,
        "content": "    }",
        "ast_features": {
          "node_type": "unknown",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 432,
            "content": "        return self.pi * radius ** 2"
          },
          {
            "line": 433,
            "content": "\"\"\""
          },
          {
            "line": 434,
            "content": "    }"
          },
          {
            "line": 435,
            "content": "    "
          },
          {
            "line": 436,
            "content": "    result = parser.parse(test_input)"
          }
        ]
      },
      {
        "line_number": 435,
        "content": "    ",
        "ast_features": {
          "node_type": "Empty",
          "depth": 0,
          "children_types": []
        },
        "context_window": [
          {
            "line": 433,
            "content": "\"\"\""
          },
          {
            "line": 434,
            "content": "    }"
          },
          {
            "line": 435,
            "content": "    "
          },
          {
            "line": 436,
            "content": "    result = parser.parse(test_input)"
          },
          {
            "line": 437,
            "content": "    print(json.dumps(result, indent=2, ensure_ascii=False)) "
          }
        ]
      },
      {
        "line_number": 436,
        "content": "    result = parser.parse(test_input)",
        "ast_features": {
          "node_type": "Assign",
          "depth": 4,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 434,
            "content": "    }"
          },
          {
            "line": 435,
            "content": "    "
          },
          {
            "line": 436,
            "content": "    result = parser.parse(test_input)"
          },
          {
            "line": 437,
            "content": "    print(json.dumps(result, indent=2, ensure_ascii=False)) "
          }
        ]
      },
      {
        "line_number": 437,
        "content": "    print(json.dumps(result, indent=2, ensure_ascii=False)) ",
        "ast_features": {
          "node_type": "Call",
          "depth": 4,
          "children_types": [
            "Name",
            "Call"
          ]
        },
        "context_window": [
          {
            "line": 435,
            "content": "    "
          },
          {
            "line": 436,
            "content": "    result = parser.parse(test_input)"
          },
          {
            "line": 437,
            "content": "    print(json.dumps(result, indent=2, ensure_ascii=False)) "
          }
        ]
      }
    ],
    "meta": {
      "file_size": 15283,
      "absolute_path": "/Users/yangwei/Desktop/detect/parsers/fileParser/code_file_parser.py"
    }
  }
}这个是我实际推理时候的输入。我现在构建文件到输入格式的解析器完成了，然后我需要对接上codebert然后再接一个下游的分类器，请给我一个可执行的详细设计。