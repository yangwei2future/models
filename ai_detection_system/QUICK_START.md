# 🚀 AI代码检测系统 - 快速开始

## ⚡ 5分钟快速部署

### 1️⃣ 安装依赖
```bash
pip install -r requirements.txt
```

### 2️⃣ 启动API服务
```bash
# 方式1: 使用启动脚本 (推荐)
./start_api.sh

# 方式2: 直接运行
python api_server.py

# 方式3: 后台运行
./start_api.sh -d
```

### 3️⃣ 验证服务
```bash
# 健康检查
curl http://localhost:8000/v1/health

# 查看API文档
# 浏览器访问: http://localhost:8000/v1/docs
```

### 4️⃣ 测试API
```bash
# 运行客户端测试
python client_example.py
```

## 🔧 快速配置

### 更改端口
```bash
./start_api.sh -p 8080
```

### 开发模式 (热重载)
```bash
./start_api.sh -r
```

### 生产模式 (多进程)
```bash
./start_api.sh -w 4
```

### 停止服务
```bash
./stop_api.sh
```

## 📡 API使用示例

### 代码片段检测
```bash
curl -X POST "http://localhost:8000/v1/detect/code" \
  -H "Content-Type: application/json" \
  -d '{
    "code_snippets": [
      {
        "content": "def hello():\n    print(\"Hello World\")",
        "filename": "test.py"
      }
    ],
    "threshold": 0.5
  }'
```

### 文件上传检测
```bash
curl -X POST "http://localhost:8000/v1/detect/file" \
  -F "file=@your_code_file.py" \
  -F "threshold=0.5"
```

## 🐍 Python客户端
```python
import requests

# 代码检测
response = requests.post(
    "http://localhost:8000/v1/detect/code",
    json={
        "code_snippets": [
            {
                "content": "def calculate_metrics(data):\n    return sum(data)/len(data)",
                "filename": "example.py"
            }
        ]
    }
)

result = response.json()
print(f"AI概率: {result['results'][0]['summary']['ai_percentage']}%")
```

## 🔗 重要链接

- 📖 **API文档**: http://localhost:8000/v1/docs
- 🔍 **健康检查**: http://localhost:8000/v1/health
- 📚 **详细文档**: [API_DEPLOYMENT_GUIDE.md](API_DEPLOYMENT_GUIDE.md)

## ❓ 常见问题

**Q: 端口被占用怎么办？**
```bash
./start_api.sh -p 8080  # 使用其他端口
```

**Q: 如何查看日志？**
```bash
./start_api.sh -d  # 后台运行会生成日志文件
tail -f api_*.log  # 查看日志
```

**Q: 如何停止服务？**
```bash
./stop_api.sh  # 使用停止脚本
```

**Q: 依赖安装失败？**
```bash
# 升级pip
pip install --upgrade pip

# 重新安装
pip install -r requirements.txt
```

---

🎉 **恭喜！您的AI代码检测系统已经成功部署！**

开始体验智能代码检测功能吧！ 🚀 