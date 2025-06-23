#!/bin/bash

# AI代码检测系统 - 停止脚本
# 使用方法: ./stop_api.sh

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🛑 AI代码检测系统 - 停止服务${NC}"
echo "================================="

# 检查PID文件
if [[ -f "api_server.pid" ]]; then
    PID=$(cat api_server.pid)
    echo -e "${BLUE}🔍 找到PID文件，进程ID: $PID${NC}"
    
    # 检查进程是否存在
    if kill -0 $PID 2>/dev/null; then
        echo -e "${YELLOW}⏳ 正在停止服务...${NC}"
        kill $PID
        
        # 等待进程结束
        sleep 2
        
        # 检查是否成功停止
        if kill -0 $PID 2>/dev/null; then
            echo -e "${YELLOW}⚠️  进程未响应，强制终止...${NC}"
            kill -9 $PID
        fi
        
        echo -e "${GREEN}✅ 服务已停止${NC}"
    else
        echo -e "${YELLOW}⚠️  进程已不存在${NC}"
    fi
    
    # 删除PID文件
    rm -f api_server.pid
    echo -e "${GREEN}🗑️  已清理PID文件${NC}"
else
    echo -e "${YELLOW}⚠️  未找到PID文件，尝试按端口查找进程...${NC}"
    
    # 尝试根据端口查找进程
    if command -v lsof &> /dev/null; then
        PIDS=$(lsof -ti :8000 2>/dev/null || true)
        if [[ -n "$PIDS" ]]; then
            echo -e "${BLUE}🔍 找到占用端口8000的进程: $PIDS${NC}"
            for pid in $PIDS; do
                echo -e "${YELLOW}⏳ 停止进程 $pid...${NC}"
                kill $pid 2>/dev/null || true
            done
            echo -e "${GREEN}✅ 相关进程已停止${NC}"
        else
            echo -e "${GREEN}ℹ️  未找到运行中的服务${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  lsof命令不可用，无法自动查找进程${NC}"
        echo -e "${YELLOW}💡 请手动查找并停止API服务进程${NC}"
    fi
fi

echo -e "${GREEN}🎉 停止操作完成！${NC}" 