# 代码特征提取系统

端到端的代码特征提取管道，将源代码文件转换为CodeBERT兼容的输入格式，用于代码来源检测模型。

## 🎯 功能特性

### 文件级特征提取
- **语言识别**: 支持Python、Java、JavaScript、C++，返回标准语言标识符
- **导入分析**: 提取和标准化依赖库列表，构建CodeBERT库上下文提示  
- **文件元数据**: 提取行数、缩进类型、编码等元数据用于特征归一化
- **文件编码**: 使用CodeBERT生成768维文件级嵌入表示

### 行级特征提取
- **文本特征**: 长度、操作符密度、信息熵、注释标记 (4维)
- **AST特征**: 深度、节点类型、子节点数、控制流标记 (4维)  
- **风格特征**: 缩进差异、命名规范、括号风格 (3维)
- **上下文窗口**: 前后N行的token编码，支持位置编码

### 特征融合与输入构造
- **特征融合**: 将11维行级特征与768维文件嵌入对齐
- **上下文注入**: 将上下文向量集成到BERT注意力机制
- **输入构造**: 生成CodeBERT友好的最终输入格式

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 基本使用
```python
from src.pipeline import FeatureExtractionPipeline

# 初始化管道
pipeline = FeatureExtractionPipeline()

# 处理单个文件
result = pipeline.process("your_code.py")

# 获取模型输入
model_input = result["model_input"]
print(f"文件嵌入: {model_input['file_embedding'].shape}")
print(f"行级特征: {model_input['line_features'].shape}")
print(f"上下文向量: {model_input['context_vectors'].shape}")
```

### 命令行演示
```bash
# 创建测试文件并运行演示
python demo_pipeline.py --create-test --output results.json

# 处理指定文件
python demo_pipeline.py --files file1.py file2.java --output results.json

# 自定义设置
python demo_pipeline.py --files *.py --window-size 3 --no-gpu
```

## 📁 项目结构

```
src/
├── file_features/          # 文件级特征提取
│   ├── language.py         # 语言检测
│   ├── imports.py          # 导入分析  
│   ├── metadata.py         # 元数据提取
│   └── encoder.py          # CodeBERT编码器
├── line_features/          # 行级特征提取
│   ├── text.py            # 文本特征
│   ├── ast.py             # AST特征
│   ├── style.py           # 风格特征
│   └── context.py         # 上下文编码
├── fusion/                 # 特征融合
│   ├── feature_fuser.py    # 特征融合器
│   └── input_builder.py    # 输入构造器
├── utils/                  # 工具模块
│   ├── ast_parser.py       # AST解析器
│   └── memory.py           # 内存管理
└── pipeline.py             # 主管道
```

## 🔧 详细配置

### 管道参数
```python
pipeline = FeatureExtractionPipeline(
    codebert_model="microsoft/codebert-base",  # CodeBERT模型
    context_window_size=2,                     # 上下文窗口大小
    use_gpu=True                               # 是否使用GPU
)
```

### 输出格式
```python
{
    "model_input": {
        "file_embedding": tensor,     # [1, 768] 文件级表示
        "line_features": tensor,      # [n_lines, 11] 行级特征矩阵
        "context_vectors": tensor,    # [n_lines, window_size, 128] 上下文编码
        "attention_mask": tensor,     # [n_lines, window_size] 注意力掩码
        "position_embeddings": tensor # [n_lines, 64] 位置编码
    },
    "metadata": {
        "language": "python",
        "line_count": 100,
        "imports": ["torch", "numpy"],
        "file_size": 2048
    }
}
```

## 💡 性能优化

### 内存管理
```python
# 检查内存使用
memory_info = pipeline.memory_manager.get_memory_usage()
print(f"内存使用: {memory_info['rss_mb']:.1f}MB")

# 处理大文件
with pipeline.memory_manager.memory_monitor("large_file_processing"):
    result = pipeline.process("large_file.py")
```

### 批处理
```python
# 批量处理多个文件
file_paths = ["file1.py", "file2.java", "file3.js"]
results = pipeline.process_batch(file_paths)
```

### AST缓存
```python
# 获取缓存统计
cache_stats = pipeline.ast_parser.get_cache_stats()
print(f"缓存命中率: {cache_stats['hit_rate']:.2%}")
```

## 🎨 特征详解

### 文本特征 (4维)
- `length_norm`: 行长度归一化 (0-1)
- `operator_density`: 操作符密度
- `entropy`: 字符级信息熵
- `is_comment`: 是否为注释行

### AST特征 (4维)  
- `depth_norm`: AST深度归一化
- `node_type_id`: 节点类型ID
- `children_count`: 子节点数量
- `is_control_flow`: 是否为控制流节点

### 风格特征 (3维)
- `indent_diff`: 缩进一致性
- `naming_score`: 命名规范评分
- `brace_style`: 括号风格评分

## 🔍 使用案例

### 代码来源检测
```python
# 提取两个文件的特征进行比较
result1 = pipeline.process("author1_code.py") 
result2 = pipeline.process("author2_code.py")

# 比较文件级嵌入相似度
similarity = torch.cosine_similarity(
    result1["model_input"]["file_embedding"],
    result2["model_input"]["file_embedding"]
)
```

### 代码风格分析
```python
# 分析项目的代码风格一致性
project_files = ["module1.py", "module2.py", "module3.py"]
results = pipeline.process_batch(project_files)

# 提取风格特征进行聚类分析
style_features = []
for result in results:
    line_features = result["model_input"]["line_features"]
    style_cols = line_features[:, 8:11]  # 风格特征列
    style_features.append(style_cols.mean(dim=0))
```

## ⚡ 性能基准

| 文件大小 | 处理时间 | 内存使用 |
|---------|---------|---------|
| <100行  | 50ms    | <500MB  |
| 1000行  | 200ms   | <1GB    |
| 10000行 | 1.5s    | <2GB    |

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [CodeBERT](https://github.com/microsoft/CodeBERT) - 预训练代码表示模型
- [Transformers](https://github.com/huggingface/transformers) - Transformer模型库
- [PyTorch](https://pytorch.org/) - 深度学习框架
