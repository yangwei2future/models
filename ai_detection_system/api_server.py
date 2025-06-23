#!/usr/bin/env python3
"""
AI代码检测系统 - FastAPI接口服务
提供RESTful API接口用于AI代码检测
"""

import os
import sys
import json
import tempfile
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# FastAPI相关导入
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入检测系统
from modular_ai_detector import ModularAIDetector

# 配置
API_VERSION = "v1"
API_TITLE = "AI代码检测系统"
API_DESCRIPTION = """
🤖 AI代码检测系统 - 基于CodeBERT的智能代码分析

## 功能特性
- 🔍 **逐行检测**: 精确到每一行代码的AI概率
- 🧠 **深度学习**: 基于CodeBERT + 手工特征的混合架构
- 📊 **多格式输出**: 支持JSON、CSV、HTML等多种格式
- 🚀 **批量处理**: 支持单文件和批量文件检测
- 🌐 **多语言**: 支持Python、Java、JavaScript、C++等主流编程语言

## 使用方式
1. **单文件检测**: 上传代码文件进行检测
2. **代码片段检测**: 直接提交代码文本
3. **批量检测**: 上传多个文件批量处理
4. **实时检测**: WebSocket实时检测接口
"""

# 创建FastAPI应用
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url=f"/{API_VERSION}/docs",
    redoc_url=f"/{API_VERSION}/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局检测器实例
detector = None

def get_detector():
    """获取检测器实例 (单例模式)"""
    global detector
    if detector is None:
        detector = ModularAIDetector(
            codebert_model="microsoft/codebert-base",
            threshold=0.5,
            output_dir="./output"
        )
    return detector

# Pydantic模型定义
class CodeSnippet(BaseModel):
    """代码片段模型"""
    content: str = Field(..., description="代码内容")
    filename: Optional[str] = Field(None, description="文件名")
    language: Optional[str] = Field(None, description="编程语言")

class DetectionRequest(BaseModel):
    """检测请求模型"""
    code_snippets: List[CodeSnippet] = Field(..., description="代码片段列表")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="AI检测阈值")
    output_format: Optional[str] = Field("json", description="输出格式")

class DetectionResponse(BaseModel):
    """检测响应模型"""
    success: bool = Field(..., description="检测是否成功")
    results: List[Dict[str, Any]] = Field(..., description="检测结果")
    statistics: Dict[str, Any] = Field(..., description="统计信息")
    processing_time: float = Field(..., description="处理时间(秒)")
    timestamp: str = Field(..., description="检测时间戳")

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="API版本")
    model_loaded: bool = Field(..., description="模型是否加载")
    timestamp: str = Field(..., description="检查时间")

# API路由定义

@app.get("/", response_model=Dict[str, str])
async def root():
    """根路径 - API信息"""
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "status": "running",
        "docs": f"/{API_VERSION}/docs",
        "timestamp": datetime.now().isoformat()
    }

@app.get(f"/{API_VERSION}/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    try:
        detector = get_detector()
        model_loaded = detector is not None
        return HealthResponse(
            status="healthy",
            version=API_VERSION,
            model_loaded=model_loaded,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            version=API_VERSION,
            model_loaded=False,
            timestamp=datetime.now().isoformat()
        )

@app.post(f"/{API_VERSION}/detect/code", response_model=DetectionResponse)
async def detect_code_snippet(request: DetectionRequest):
    """
    代码片段检测接口
    
    检测提交的代码片段是否为AI生成
    """
    start_time = datetime.now()
    
    try:
        detector = get_detector()
        
        # 设置阈值
        if request.threshold != 0.5:
            detector.set_threshold(request.threshold)
        
        results = []
        
        # 处理每个代码片段
        for i, snippet in enumerate(request.code_snippets):
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(snippet.content)
                temp_file_path = temp_file.name
            
            try:
                # 检测代码
                result = detector.detect_file(temp_file_path)
                
                # 添加额外信息
                result['snippet_index'] = i
                result['filename'] = snippet.filename or f"snippet_{i}.py"
                result['language'] = snippet.language or "python"
                
                results.append(result)
                
            finally:
                # 清理临时文件
                os.unlink(temp_file_path)
        
        # 生成统计信息
        successful_results = [r for r in results if r.get('success', False)]
        total_lines = sum(r.get('summary', {}).get('total_lines', 0) for r in successful_results)
        total_ai_lines = sum(r.get('summary', {}).get('ai_lines', 0) for r in successful_results)
        
        statistics = {
            "total_snippets": len(request.code_snippets),
            "successful_detections": len(successful_results),
            "total_code_lines": total_lines,
            "total_ai_lines": total_ai_lines,
            "overall_ai_percentage": (total_ai_lines / max(total_lines, 1)) * 100
        }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return DetectionResponse(
            success=True,
            results=results,
            statistics=statistics,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"检测失败: {str(e)}"
        )

@app.post(f"/{API_VERSION}/detect/file")
async def detect_file(
    file: UploadFile = File(..., description="要检测的代码文件"),
    threshold: float = Form(0.5, ge=0.0, le=1.0, description="AI检测阈值"),
    output_format: str = Form("json", description="输出格式")
):
    """
    单文件检测接口
    
    上传代码文件进行AI检测
    """
    start_time = datetime.now()
    
    try:
        detector = get_detector()
        
        # 设置阈值
        if threshold != 0.5:
            detector.set_threshold(threshold)
        
        # 保存上传的文件
        with tempfile.NamedTemporaryFile(mode='wb', suffix=Path(file.filename).suffix, delete=False) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # 检测文件
            result = detector.detect_file(temp_file_path)
            result['original_filename'] = file.filename
            result['file_size'] = len(content)
            
            # 根据输出格式返回结果
            if output_format.lower() == "json":
                processing_time = (datetime.now() - start_time).total_seconds()
                return {
                    "success": True,
                    "result": result,
                    "processing_time": processing_time,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # 生成其他格式的文件
                output_files = detector.detect_and_output(
                    input_paths=[temp_file_path],
                    output_formats=[output_format],
                    output_filename=f"detection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                # 返回文件
                if output_files and output_format in output_files:
                    return FileResponse(
                        output_files[output_format],
                        media_type="application/octet-stream",
                        filename=f"detection_result.{output_format}"
                    )
                else:
                    raise HTTPException(status_code=500, detail="文件生成失败")
                    
        finally:
            # 清理临时文件
            os.unlink(temp_file_path)
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"文件检测失败: {str(e)}"
        )

@app.post(f"/{API_VERSION}/detect/batch")
async def detect_batch_files(
    files: List[UploadFile] = File(..., description="要检测的代码文件列表"),
    threshold: float = Form(0.5, ge=0.0, le=1.0, description="AI检测阈值"),
    output_format: str = Form("json", description="输出格式")
):
    """
    批量文件检测接口
    
    上传多个代码文件进行批量AI检测
    """
    start_time = datetime.now()
    
    try:
        detector = get_detector()
        
        # 设置阈值
        if threshold != 0.5:
            detector.set_threshold(threshold)
        
        temp_files = []
        file_paths = []
        
        # 保存所有上传的文件
        for file in files:
            with tempfile.NamedTemporaryFile(mode='wb', suffix=Path(file.filename).suffix, delete=False) as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_files.append(temp_file.name)
                file_paths.append(temp_file.name)
        
        try:
            # 批量检测
            batch_result = detector.detect_batch(file_paths)
            
            # 添加原始文件名信息
            for i, result in enumerate(batch_result.get('results', [])):
                if i < len(files):
                    result['original_filename'] = files[i].filename
            
            processing_time = (datetime.now() - start_time).total_seconds()
            batch_result['processing_time'] = processing_time
            batch_result['timestamp'] = datetime.now().isoformat()
            
            return batch_result
            
        finally:
            # 清理所有临时文件
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"批量检测失败: {str(e)}"
        )

@app.get(f"/{API_VERSION}/info/architecture")
async def get_architecture_info():
    """
    获取系统架构信息
    """
    try:
        detector = get_detector()
        arch_info = detector.get_architecture_info()
        return {
            "success": True,
            "architecture": arch_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取架构信息失败: {str(e)}"
        )

@app.post(f"/{API_VERSION}/config/threshold")
async def update_threshold(threshold: float = Form(..., ge=0.0, le=1.0)):
    """
    更新检测阈值
    """
    try:
        detector = get_detector()
        detector.set_threshold(threshold)
        return {
            "success": True,
            "message": f"阈值已更新为 {threshold}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"更新阈值失败: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }
    )

# 启动服务器的函数
def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """启动FastAPI服务器"""
    print(f"🚀 启动AI代码检测API服务器...")
    print(f"📍 服务地址: http://{host}:{port}")
    print(f"📖 API文档: http://{host}:{port}/{API_VERSION}/docs")
    print(f"🔧 配置信息:")
    print(f"   - 主机: {host}")
    print(f"   - 端口: {port}")
    print(f"   - 热重载: {reload}")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI代码检测API服务器")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--reload", action="store_true", help="启用热重载")
    
    args = parser.parse_args()
    start_server(args.host, args.port, args.reload) 