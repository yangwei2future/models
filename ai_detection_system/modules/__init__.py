"""
AI代码检测系统 - 模块化架构

按照架构流程图组织的模块：
1. file_parser - 文件解析器
2. feature_extraction - 特征提取（文件级和行级）
3. feature_fusion - 特征融合
4. inter_line_modeling - 行间关系建模
5. codebert_encoding - CodeBERT编码
6. classification - 行分类器
7. threshold_filter - 阈值过滤
8. result_aggregation - 结果聚合
9. output_system - 输出系统
"""

# 文件解析器
try:
    from .file_parser import FileParser
except ImportError:
    FileParser = None

# 特征提取
try:
    from .feature_extraction import LineFeatureExtractor, FileFeatureExtractor
except ImportError:
    LineFeatureExtractor = None
    FileFeatureExtractor = None

# 特征融合
try:
    from .feature_fusion import FeatureFusion
except ImportError:
    FeatureFusion = None

# 行间关系建模
try:
    from .inter_line_modeling import InterLineRelationship
except ImportError:
    InterLineRelationship = None

# CodeBERT编码
try:
    from .codebert_encoding import CodeBERTEncoder
except ImportError:
    CodeBERTEncoder = None

# 分类器
try:
    from .classification import LineClassifier
except ImportError:
    LineClassifier = None

# 阈值过滤
try:
    from .threshold_filter import ThresholdFilter
except ImportError:
    ThresholdFilter = None

# 结果聚合
try:
    from .result_aggregation import ResultAggregator
except ImportError:
    ResultAggregator = None

# 输出系统
try:
    from .output_system import OutputSystem
except ImportError:
    OutputSystem = None

# 导出所有可用的类
__all__ = [name for name in [
    'FileParser',
    'LineFeatureExtractor',
    'FileFeatureExtractor', 
    'FeatureFusion',
    'InterLineRelationship',
    'CodeBERTEncoder',
    'LineClassifier',
    'ThresholdFilter',
    'ResultAggregator',
    'OutputSystem'
] if globals()[name] is not None] 