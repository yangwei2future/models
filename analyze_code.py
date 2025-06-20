#!/usr/bin/env python3
"""
代码分析命令行工具
使用 CodeFileParser 分析文件或文件夹中的代码文件

使用方法:
  python analyze_code.py <file_or_folder_path> [options]

示例:
  python analyze_code.py test_sample.py                # 分析单个文件
  python analyze_code.py ./parsers                     # 分析文件夹
  python analyze_code.py test_sample.py --pretty       # 美化输出
  python analyze_code.py test_sample.py --output result.json  # 输出到文件
"""

import os
import sys
import json
import argparse

# 添加parsers目录到Python路径
parsers_path = os.path.join(os.path.dirname(__file__), 'parsers', 'fileParser')
if parsers_path not in sys.path:
    sys.path.insert(0, parsers_path)

from code_file_parser import CodeFileParser


def is_code_file(file_path):
    """检查是否为支持的代码文件"""
    supported_extensions = {'.py', '.java', '.js', '.jsx', '.ts', '.tsx'}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in supported_extensions


def read_file_safely(file_path):
    """安全读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # 尝试其他编码
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()
        except Exception:
            print(f"警告: 无法读取文件 {file_path}，跳过处理")
            return None
    except Exception as e:
        print(f"错误: 读取文件 {file_path} 失败: {e}")
        return None


def analyze_single_file(file_path, parser):
    """分析单个文件"""
    print(f"正在分析文件: {file_path}")
    
    content = read_file_safely(file_path)
    if content is None:
        return None
    
    input_data = {
        "file_path": file_path,
        "content": content
    }
    
    try:
        result = parser.parse(input_data)
        # 添加元信息
        result["meta"] = {
            "file_size": len(content),
            "absolute_path": os.path.abspath(file_path),
        }
        return result
    except Exception as e:
        print(f"错误: 分析文件 {file_path} 失败: {e}")
        return None


def find_code_files(folder_path):
    """在文件夹中找到所有代码文件"""
    code_files = []
    
    for root, dirs, files in os.walk(folder_path):
        # 忽略常见的非代码目录
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in 
                  {'node_modules', '__pycache__', 'venv', 'env', 'build', 'dist', 'target'}]
        
        for file in files:
            if is_code_file(file):
                file_path = os.path.join(root, file)
                code_files.append(file_path)
    
    return code_files


def analyze_folder(folder_path, parser, max_files=None):
    """分析文件夹中的代码文件"""
    print(f"正在分析文件夹: {folder_path}")
    
    code_files = find_code_files(folder_path)
    
    if not code_files:
        return {
            "summary": {
                "total_files": 0,
                "analyzed_files": 0,
                "failed_files": 0,
                "folder_path": os.path.abspath(folder_path)
            },
            "files": []
        }
    
    if max_files and len(code_files) > max_files:
        print(f"找到 {len(code_files)} 个文件，限制分析前 {max_files} 个")
        code_files = code_files[:max_files]
    
    print(f"找到 {len(code_files)} 个代码文件，开始分析...")
    
    results = []
    failed_count = 0
    
    for i, file_path in enumerate(code_files, 1):
        print(f"进度: {i}/{len(code_files)} - {os.path.basename(file_path)}")
        
        result = analyze_single_file(file_path, parser)
        if result:
            results.append(result)
        else:
            failed_count += 1
    
    # 统计信息
    languages_found = list(set(r.get("language", "unknown") for r in results))
    
    return {
        "summary": {
            "total_files": len(code_files),
            "analyzed_files": len(results),
            "failed_files": failed_count,
            "folder_path": os.path.abspath(folder_path),
            "languages_found": languages_found
        },
        "files": results
    }


def main():
    parser = argparse.ArgumentParser(
        description="代码文件分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s test.py                          # 分析单个文件
  %(prog)s ./src                            # 分析整个文件夹
  %(prog)s ./src --max-files 10             # 限制分析文件数量
  %(prog)s test.py --output result.json     # 保存结果到文件
  %(prog)s ./src --pretty                   # 美化JSON输出
  %(prog)s ./src --window-size 3            # 设置上下文窗口大小
        """
    )
    
    parser.add_argument('target', help='要分析的文件或文件夹路径')
    parser.add_argument('-o', '--output', help='输出文件路径 (默认输出到控制台)')
    parser.add_argument('--pretty', action='store_true', help='美化JSON输出')
    parser.add_argument('--window-size', type=int, default=2, help='上下文窗口大小 (默认: 2)')
    parser.add_argument('--max-files', type=int, help='最大分析文件数量 (仅对文件夹有效)')
    
    args = parser.parse_args()
    
    # 检查目标路径是否存在
    if not os.path.exists(args.target):
        print(f"错误: 路径不存在: {args.target}")
        sys.exit(1)
    
    try:
        # 创建解析器
        code_parser = CodeFileParser(context_window_size=args.window_size)
        
        # 分析目标
        if os.path.isfile(args.target):
            # 分析单个文件
            result = analyze_single_file(args.target, code_parser)
            if result:
                output_data = {"type": "single_file", "result": result}
            else:
                output_data = {"type": "single_file", "result": None, "error": "文件分析失败"}
        
        elif os.path.isdir(args.target):
            # 分析文件夹
            result = analyze_folder(args.target, code_parser, args.max_files)
            output_data = {"type": "folder", "result": result}
        
        else:
            print(f"错误: 不支持的路径类型: {args.target}")
            sys.exit(1)
        
        # 格式化输出
        if args.pretty:
            json_output = json.dumps(output_data, indent=2, ensure_ascii=False)
        else:
            json_output = json.dumps(output_data, ensure_ascii=False)
        
        # 输出结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(json_output)
            print(f"\n分析完成！结果已保存到: {args.output}")
        else:
            print("\n" + "="*50)
            print("分析结果:")
            print("="*50)
            print(json_output)
    
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 