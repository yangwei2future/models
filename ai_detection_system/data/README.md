# 🗂️ AI代码检测训练数据集

这个目录包含用于训练AI代码检测模型的数据集，专门用于区分人类编写的代码和AI生成的代码。

## 📁 数据集结构

```
data/
├── README.md                    # 本文件
├── dataset_labels.json          # 数据集标注文件
├── dataset_loader.py           # 数据集加载器
├── human_code_samples.py       # 人类代码样本
├── ai_code_samples.py          # AI生成代码样本
└── mixed_code_samples.py       # 混合代码样本
```

## 📊 数据集概览

- **总样本数**: ~450行代码
- **人类代码**: ~180行 (40%)
- **AI代码**: ~270行 (60%)
- **文件数**: 3个Python文件
- **标注质量**: 手工标注 + 启发式规则

## 📄 文件说明

### 1. human_code_samples.py
**特点**: 人类真实编程风格
- 简洁直接的函数定义
- 最少必要的注释
- 基础错误处理
- 实用的变量命名
- 较少的类型注解
- 实际工作代码模式

**示例**:
```python
def add(a, b):
    return a + b

for num in numbers:
    if num % 2 == 0:
        print(num)
```

### 2. ai_code_samples.py
**特点**: AI生成代码特征
- 详细的文档字符串
- 全面的类型注解
- 复杂的错误处理和日志
- 抽象基类和设计模式
- 详细的参数文档
- 复杂的类层次结构

**示例**:
```python
class AdvancedDataProcessor(DataProcessorInterface):
    """
    Advanced data processor with comprehensive functionality.
    
    This class implements sophisticated data processing algorithms
    with support for asynchronous operations, error handling,
    and detailed logging capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data processor with configuration parameters.
        
        Args:
            config: Configuration dictionary containing processing parameters
        """
```

### 3. mixed_code_samples.py
**特点**: 混合编程风格
- 包含人类和AI风格的代码段
- 用于测试模型的泛化能力
- 真实项目中的代码混合情况

## 🏷️ 标注系统

### dataset_labels.json
包含每行代码的详细标注：
```json
{
  "line_number": {
    "content": "代码内容",
    "is_ai": true/false,
    "confidence": 0.0-1.0
  }
}
```

### 置信度评分
- **0.9-1.0**: 非常确信 - 明确的AI或人类模式
- **0.8-0.9**: 确信 - 强指标存在
- **0.7-0.8**: 中等置信度 - 一些指标
- **0.6-0.7**: 低置信度 - 模糊模式
- **0.5-0.6**: 很低置信度 - 不清楚

## 🔍 AI检测特征

### 人类代码特征
- ✅ 简单直接的函数定义
- ✅ 最少必要的注释
- ✅ 基础错误处理
- ✅ 直观的变量名
- ✅ 较少的类型注解
- ✅ 实用的工作代码模式

### AI代码特征
- 🤖 详细的文档字符串和描述
- 🤖 广泛的类型注解
- 🤖 复杂的错误处理和日志
- 🤖 抽象基类和设计模式
- 🤖 详细的参数文档
- 🤖 复杂的类层次结构
- 🤖 全面的配置系统

## 🚀 使用方法

### 1. 加载数据集
```python
from dataset_loader import CodeDatasetLoader

# 初始化加载器
loader = CodeDatasetLoader()

# 加载所有样本
all_samples = loader.load_all_samples()

# 获取平衡数据集
human_samples, ai_samples = loader.get_balanced_dataset()

# 获取训练和测试数据
train_samples, test_samples = loader.get_training_data(test_split=0.2)
```

### 2. 数据集统计
```python
# 获取统计信息
stats = loader.get_dataset_statistics()
print(f"Total samples: {stats['total_samples']}")
print(f"Human: {stats['human_samples']}, AI: {stats['ai_samples']}")
```

### 3. 导出训练数据
```python
# 导出为JSON格式
export_data = loader.export_training_data("training_data.json")
```

### 4. 运行演示
```bash
python dataset_loader.py
```

## 📈 数据质量

### 标注质量保证
- **手工标注**: 关键行进行人工标注
- **启发式规则**: 基于AI代码特征的自动标注
- **交叉验证**: 多种方法验证标注准确性
- **置信度评分**: 每个标注都有置信度分数

### 数据平衡
- 支持自动数据平衡
- 可配置每类最大样本数
- 训练/测试集自动分割
- 随机打乱确保公平性

## 🔧 扩展数据集

### 添加新样本
1. 在相应的.py文件中添加代码
2. 在dataset_labels.json中添加标注
3. 运行dataset_loader.py验证

### 添加新文件
1. 创建新的.py文件
2. 在dataset_loader.py中添加文件名
3. 在dataset_labels.json中添加文件信息

## 📊 数据集统计示例

运行`python dataset_loader.py`可以看到：

```
🗂️  AI Code Detection Dataset Loader Demo
==================================================
📊 Dataset Statistics:
  Total samples: 312
  Human samples: 125 (40.1%)
  AI samples: 187 (59.9%)
  Average confidence: 0.823

📁 File Statistics:
  human_code_samples.py: 89 total (89 human, 0 AI)
  ai_code_samples.py: 156 total (0 human, 156 AI)
  mixed_code_samples.py: 67 total (36 human, 31 AI)

⚖️  Balanced Dataset: 50 human, 50 AI
🚂 Training Data: 80 train, 20 test
💾 Exported training data with 80 training samples
```

## 🎯 训练建议

### 数据预处理
- 使用平衡数据集避免类别偏差
- 适当的训练/测试分割比例
- 考虑置信度权重

### 模型训练
- 使用CodeBERT作为基础模型
- 结合手工特征和深度特征
- 注意过拟合问题

### 评估指标
- 准确率、精确率、召回率、F1分数
- 混淆矩阵分析
- 不同置信度阈值的性能

---

**注意**: 这个数据集是为演示和研究目的创建的。在实际应用中，建议使用更大规模和更多样化的数据集。 