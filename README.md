# 🤖 AI代码检测系统

## 📍 项目已重新组织

AI代码检测系统已经重新组织到 `ai_detection_system/` 目录中。

## 🚀 快速开始

```bash
cd ai_detection_system
pip install -r requirements.txt
python core/train_ai_detector.py
python core/line_ai_detector.py --model models/ai_detector.pt --input your_file.py
```

## 📁 新的目录结构

```
ai_detection_system/
├── core/                    # 核心模块
│   ├── ai_code_detector.py     # 通用AI检测器
│   ├── line_ai_detector.py     # 主要工具 - 输出您要求的格式
│   └── train_ai_detector.py    # 模型训练脚本
├── tools/                   # 辅助工具
├── examples/                # 演示脚本
├── docs/                    # 文档
├── models/                  # 模型文件
├── README.md               # 详细说明
└── requirements.txt        # 依赖列表
```

## 🎯 主要功能

- **逐行AI检测**: 输出每行代码的AI概率
- **精确JSON格式**: 完全符合您的要求
- **多语言支持**: Python, Java, JavaScript, C/C++, Go等
- **批量处理**: 支持文件、目录、递归处理

## 📖 详细文档

请查看 `ai_detection_system/README.md` 和 `ai_detection_system/docs/` 目录获取完整的使用指南。

---

**核心命令**: `python ai_detection_system/core/line_ai_detector.py --model ai_detection_system/models/ai_detector.pt --input your_file.py --output results.json` 