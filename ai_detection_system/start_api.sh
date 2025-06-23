#!/bin/bash

# AI代码检测系统 - 启动脚本
# 使用方法: ./start_api.sh [选项]

set -e

# 默认配置
HOST="0.0.0.0"
PORT="8000"
WORKERS="1"
RELOAD="false"
LOG_LEVEL="info"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 帮助信息
show_help() {
    echo -e "${BLUE}🚀 AI代码检测系统 - 启动脚本${NC}"
    echo ""
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --host HOST        服务器主机地址 (默认: 0.0.0.0)"
    echo "  -p, --port PORT        服务器端口 (默认: 8000)"
    echo "  -w, --workers WORKERS  工作进程数 (默认: 1)"
    echo "  -r, --reload          启用热重载 (开发模式)"
    echo "  -d, --daemon          后台运行"
    echo "  -l, --log-level LEVEL  日志级别 (默认: info)"
    echo "  --help                显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                    # 使用默认配置启动"
    echo "  $0 -p 8080 -r         # 端口8080，启用热重载"
    echo "  $0 -w 4 -d            # 4个工作进程，后台运行"
    echo ""
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -r|--reload)
            RELOAD="true"
            shift
            ;;
        -d|--daemon)
            DAEMON="true"
            shift
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}❌ 未知选项: $1${NC}"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查Python环境
check_python() {
    echo -e "${BLUE}🔍 检查Python环境...${NC}"
    
    if ! command -v python &> /dev/null; then
        echo -e "${RED}❌ 未找到Python！请安装Python 3.8+${NC}"
        exit 1
    fi
    
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo -e "${GREEN}✅ Python版本: $PYTHON_VERSION${NC}"
    
    if [[ $(echo "$PYTHON_VERSION < 3.8" | bc -l) -eq 1 ]]; then
        echo -e "${RED}❌ Python版本过低！需要Python 3.8+${NC}"
        exit 1
    fi
}

# 检查依赖
check_dependencies() {
    echo -e "${BLUE}🔍 检查依赖包...${NC}"
    
    REQUIRED_PACKAGES=("torch" "transformers" "fastapi" "uvicorn")
    
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if python -c "import $package" &> /dev/null; then
            echo -e "${GREEN}✅ $package${NC}"
        else
            echo -e "${RED}❌ 缺少依赖: $package${NC}"
            echo -e "${YELLOW}💡 请运行: pip install -r requirements.txt${NC}"
            exit 1
        fi
    done
}

# 检查端口是否被占用
check_port() {
    echo -e "${BLUE}🔍 检查端口 $PORT...${NC}"
    
    if command -v lsof &> /dev/null; then
        if lsof -i :$PORT &> /dev/null; then
            echo -e "${YELLOW}⚠️  端口 $PORT 已被占用${NC}"
            echo -e "${YELLOW}💡 请使用其他端口或停止占用进程${NC}"
            exit 1
        fi
    fi
    
    echo -e "${GREEN}✅ 端口 $PORT 可用${NC}"
}

# 启动服务
start_server() {
    echo -e "${BLUE}🚀 启动AI代码检测API服务器...${NC}"
    echo -e "${GREEN}📍 服务地址: http://$HOST:$PORT${NC}"
    echo -e "${GREEN}📖 API文档: http://$HOST:$PORT/v1/docs${NC}"
    echo -e "${GREEN}🔧 配置信息:${NC}"
    echo -e "   - 主机: $HOST"
    echo -e "   - 端口: $PORT"
    echo -e "   - 工作进程: $WORKERS"
    echo -e "   - 热重载: $RELOAD"
    echo -e "   - 日志级别: $LOG_LEVEL"
    
    if [[ "$DAEMON" == "true" ]]; then
        echo -e "   - 运行模式: 后台"
    else
        echo -e "   - 运行模式: 前台"
    fi
    
    echo ""
    echo -e "${YELLOW}💡 按 Ctrl+C 停止服务${NC}"
    echo ""
    
    # 构建启动命令
    if [[ "$RELOAD" == "true" ]]; then
        # 开发模式 - 使用热重载
        CMD="python api_server.py --host $HOST --port $PORT --reload"
    else
        # 生产模式 - 使用uvicorn
        CMD="uvicorn api_server:app --host $HOST --port $PORT --workers $WORKERS --log-level $LOG_LEVEL"
    fi
    
    # 执行启动命令
    if [[ "$DAEMON" == "true" ]]; then
        # 后台运行
        LOG_FILE="api_$(date +%Y%m%d_%H%M%S).log"
        nohup $CMD > $LOG_FILE 2>&1 &
        PID=$!
        echo -e "${GREEN}✅ 服务已在后台启动 (PID: $PID)${NC}"
        echo -e "${GREEN}📝 日志文件: $LOG_FILE${NC}"
        echo -e "${GREEN}🛑 停止服务: kill $PID${NC}"
        
        # 保存PID到文件
        echo $PID > api_server.pid
        echo -e "${GREEN}💾 PID已保存到 api_server.pid${NC}"
    else
        # 前台运行
        exec $CMD
    fi
}

# 主函数
main() {
    echo -e "${BLUE}🤖 AI代码检测系统启动器${NC}"
    echo "================================="
    
    check_python
    check_dependencies
    check_port
    start_server
}

# 信号处理
cleanup() {
    echo -e "\n${YELLOW}🛑 正在停止服务...${NC}"
    if [[ -f "api_server.pid" ]]; then
        PID=$(cat api_server.pid)
        kill $PID 2>/dev/null || true
        rm -f api_server.pid
        echo -e "${GREEN}✅ 服务已停止${NC}"
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

# 运行主函数
main 