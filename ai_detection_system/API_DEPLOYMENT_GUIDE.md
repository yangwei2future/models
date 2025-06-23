# 🚀 AI代码检测系统 - API部署指南

## 📋 目录
- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [启动服务](#启动服务)
- [API接口文档](#api接口文档)
- [客户端测试](#客户端测试)
- [生产部署](#生产部署)
- [故障排除](#故障排除)

## 🔧 环境要求

### 系统要求
- **操作系统**: Linux, macOS, Windows
- **Python版本**: Python 3.8+
- **内存**: 最少4GB RAM (推荐8GB+)
- **存储**: 最少2GB可用空间
- **GPU**: 可选，支持CUDA加速

### 依赖库
```bash
torch>=1.9.0
numpy>=1.20.0
transformers>=4.20.0
tokenizers>=0.12.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
pydantic>=2.0.0
requests>=2.28.0
```

## 📦 安装步骤

### 1. 克隆项目
```bash
git clone <your-repo-url>
cd detect/ai_detection_system
```

### 2. 创建虚拟环境
```bash
# 使用venv
python -m venv ai_detection_env
source ai_detection_env/bin/activate  # Linux/macOS
# ai_detection_env\Scripts\activate  # Windows

# 或使用conda
conda create -n ai_detection python=3.9
conda activate ai_detection
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 验证安装
```bash
python -c "import torch; import transformers; import fastapi; print('✅ 所有依赖安装成功!')"
```

## 🚀 启动服务

### 开发模式启动
```bash
# 基本启动
python api_server.py

# 自定义配置启动
python api_server.py --host 0.0.0.0 --port 8000 --reload

# 后台启动
nohup python api_server.py > api.log 2>&1 &
```

### 生产模式启动
```bash
# 使用uvicorn直接启动
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4

# 使用gunicorn (需要安装gunicorn)
pip install gunicorn
gunicorn api_server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 启动验证
```bash
# 检查服务状态
curl http://localhost:8000/v1/health

# 查看API文档
# 浏览器访问: http://localhost:8000/v1/docs
```

## 📖 API接口文档

### 基础信息
- **基础URL**: `http://localhost:8000/v1`
- **文档地址**: `http://localhost:8000/v1/docs`
- **API版本**: v1

### 主要接口

#### 1. 健康检查
```http
GET /v1/health
```

**响应示例**:
```json
{
  "status": "healthy",
  "version": "v1", 
  "model_loaded": true,
  "timestamp": "2024-01-01T12:00:00"
}
```

#### 2. 代码片段检测
```http
POST /v1/detect/code
Content-Type: application/json
```

**请求体**:
```json
{
  "code_snippets": [
    {
      "content": "def hello():\n    print('Hello World')",
      "filename": "test.py",
      "language": "python"
    }
  ],
  "threshold": 0.5,
  "output_format": "json"
}
```

**响应示例**:
```json
{
  "success": true,
  "results": [...],
  "statistics": {
    "total_snippets": 1,
    "successful_detections": 1,
    "total_code_lines": 2,
    "total_ai_lines": 1,
    "overall_ai_percentage": 50.0
  },
  "processing_time": 1.23,
  "timestamp": "2024-01-01T12:00:00"
}
```

#### 3. 文件上传检测
```http
POST /v1/detect/file
Content-Type: multipart/form-data
```

**参数**:
- `file`: 上传的代码文件
- `threshold`: 检测阈值 (0.0-1.0)
- `output_format`: 输出格式 (json/csv/html)

#### 4. 批量文件检测
```http
POST /v1/detect/batch
Content-Type: multipart/form-data
```

**参数**:
- `files`: 多个代码文件
- `threshold`: 检测阈值
- `output_format`: 输出格式

#### 5. 获取架构信息
```http
GET /v1/info/architecture
```

#### 6. 更新检测阈值
```http
POST /v1/config/threshold
Content-Type: application/x-www-form-urlencoded
```

**参数**:
- `threshold`: 新的阈值 (0.0-1.0)

## 🧪 客户端测试

### 使用提供的测试脚本
```bash
# 确保API服务器正在运行
python api_server.py

# 在另一个终端运行测试
python client_example.py
```

### 使用curl测试

#### 健康检查
```bash
curl -X GET "http://localhost:8000/v1/health"
```

#### 代码片段检测
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

#### 文件上传检测
```bash
curl -X POST "http://localhost:8000/v1/detect/file" \
  -F "file=@example_test.py" \
  -F "threshold=0.5" \
  -F "output_format=json"
```

### 使用Python requests
```python
import requests

# 代码片段检测
response = requests.post(
    "http://localhost:8000/v1/detect/code",
    json={
        "code_snippets": [
            {
                "content": "def hello():\n    print('Hello World')",
                "filename": "test.py"
            }
        ],
        "threshold": 0.5
    }
)

result = response.json()
print(result)
```

## 🏭 生产部署

### Docker部署

#### 1. 创建Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 2. 构建和运行
```bash
# 构建镜像
docker build -t ai-detection-api .

# 运行容器
docker run -p 8000:8000 ai-detection-api

# 后台运行
docker run -d -p 8000:8000 --name ai-detection ai-detection-api
```

### 使用Nginx反向代理

#### nginx.conf
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 使用systemd服务

#### 创建服务文件
```bash
sudo nano /etc/systemd/system/ai-detection.service
```

```ini
[Unit]
Description=AI Code Detection API
After=network.target

[Service]
Type=exec
User=your-user
Group=your-group
WorkingDirectory=/path/to/ai_detection_system
Environment=PATH=/path/to/venv/bin
ExecStart=/path/to/venv/bin/python api_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

#### 启动服务
```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-detection
sudo systemctl start ai-detection
sudo systemctl status ai-detection
```

## 🔧 配置选项

### 环境变量
```bash
# API配置
export API_HOST=0.0.0.0
export API_PORT=8000
export API_WORKERS=4

# 模型配置
export CODEBERT_MODEL=microsoft/codebert-base
export DEFAULT_THRESHOLD=0.5
export OUTPUT_DIR=./output

# 日志配置
export LOG_LEVEL=INFO
export LOG_FILE=api.log
```

### 性能优化
```python
# api_server.py 中的配置
detector = ModularAIDetector(
    codebert_model="microsoft/codebert-base",
    threshold=0.5,
    output_dir="./output"
)

# 可以调整的参数:
# - batch_size: 批处理大小
# - max_length: 最大序列长度
# - num_workers: 工作进程数
```

## 🚨 故障排除

### 常见问题

#### 1. 模型下载失败
```bash
# 手动下载模型
python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('microsoft/codebert-base'); AutoTokenizer.from_pretrained('microsoft/codebert-base')"
```

#### 2. 内存不足
```bash
# 减少批处理大小
# 在代码中调整 batch_size 参数
```

#### 3. 端口被占用
```bash
# 查看端口占用
lsof -i :8000

# 杀死占用进程
kill -9 <PID>

# 使用其他端口
python api_server.py --port 8001
```

#### 4. CORS错误
```python
# 在api_server.py中已配置CORS
# 如果仍有问题，检查前端请求头
```

### 日志查看
```bash
# 查看实时日志
tail -f api.log

# 查看错误日志
grep -i error api.log

# 查看系统服务日志
sudo journalctl -u ai-detection -f
```

### 性能监控
```bash
# 监控CPU和内存使用
htop

# 监控API响应时间
curl -w "@curl-format.txt" -s -o /dev/null http://localhost:8000/v1/health
```

## 📞 技术支持

如果遇到问题，请提供以下信息：
1. 错误信息和日志
2. 系统环境 (OS, Python版本)
3. 请求示例和响应
4. 配置信息

---

## 🎉 部署完成！

现在您的AI代码检测系统已经成功部署！

- 📖 **API文档**: http://localhost:8000/v1/docs
- 🔍 **健康检查**: http://localhost:8000/v1/health
- 🧪 **测试客户端**: `python client_example.py`

享受使用AI代码检测系统！ 🚀 