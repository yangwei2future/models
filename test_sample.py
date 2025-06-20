#!/usr/bin/env python3
"""
测试样例文件
包含各种Python代码结构，用于测试CodeFileParser的解析功能
"""

import os
import sys
import math
from collections import defaultdict
from typing import List, Dict, Any, Optional

# 全局变量
GLOBAL_CONSTANT = 42
debug_mode = True

class Calculator:
    """
    简单计算器类
    支持基本的数学运算
    """
    
    def __init__(self, precision: int = 2):
        """初始化计算器"""
        self.precision = precision
        self.history = []
        self.pi = math.pi
    
    def add(self, a: float, b: float) -> float:
        """加法运算"""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return round(result, self.precision)
    
    def subtract(self, a: float, b: float) -> float:
        """减法运算"""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return round(result, self.precision)
    
    def multiply(self, a: float, b: float) -> float:
        """乘法运算"""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return round(result, self.precision)
    
    def divide(self, a: float, b: float) -> float:
        """除法运算"""
        if b == 0:
            raise ValueError("除数不能为零")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return round(result, self.precision)
    
    def circle_area(self, radius: float) -> float:
        """计算圆的面积"""
        if radius < 0:
            raise ValueError("半径不能为负数")
        area = self.pi * radius ** 2
        return round(area, self.precision)
    
    def get_history(self) -> List[str]:
        """获取计算历史"""
        return self.history.copy()
    
    def clear_history(self):
        """清空计算历史"""
        self.history.clear()

def factorial(n: int) -> int:
    """
    计算阶乘
    使用递归实现
    """
    if n < 0:
        raise ValueError("阶乘不能计算负数")
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n: int) -> int:
    """
    计算斐波那契数列
    使用递归实现
    """
    if n < 0:
        raise ValueError("斐波那契数列不能计算负数")
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def is_prime(num: int) -> bool:
    """检查是否为质数"""
    if num < 2:
        return False
    
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    
    return True

def find_primes(limit: int) -> List[int]:
    """找出指定范围内的所有质数"""
    primes = []
    for num in range(2, limit + 1):
        if is_prime(num):
            primes.append(num)
    return primes

def process_data(data: List[int]) -> Dict[str, Any]:
    """
    处理数据列表，返回统计信息
    """
    if not data:
        return {"error": "数据为空"}
    
    stats = {
        "count": len(data),
        "sum": sum(data),
        "average": sum(data) / len(data),
        "min": min(data),
        "max": max(data),
        "even_count": sum(1 for x in data if x % 2 == 0),
        "odd_count": sum(1 for x in data if x % 2 == 1)
    }
    
    return stats

# 装饰器函数
def timer(func):
    """计时装饰器"""
    import time
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 执行时间: {end_time - start_time:.4f}秒")
        return result
    
    return wrapper

@timer
def complex_calculation(n: int) -> int:
    """复杂计算示例"""
    result = 0
    for i in range(n):
        result += i ** 2
    return result

def main():
    """主函数"""
    print("=== 计算器测试 ===")
    
    # 创建计算器实例
    calc = Calculator(precision=3)
    
    # 测试基本运算
    print(f"加法: {calc.add(10, 5)}")
    print(f"减法: {calc.subtract(10, 5)}")
    print(f"乘法: {calc.multiply(10, 5)}")
    print(f"除法: {calc.divide(10, 5)}")
    print(f"圆面积: {calc.circle_area(5)}")
    
    print("\n=== 数学函数测试 ===")
    print(f"阶乘 5!: {factorial(5)}")
    print(f"斐波那契数列第10项: {fibonacci(10)}")
    
    print("\n=== 质数测试 ===")
    primes = find_primes(20)
    print(f"20以内的质数: {primes}")
    
    print("\n=== 数据处理测试 ===")
    test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    stats = process_data(test_data)
    print(f"数据统计: {stats}")
    
    print("\n=== 性能测试 ===")
    result = complex_calculation(1000)
    print(f"复杂计算结果: {result}")
    
    print("\n=== 计算历史 ===")
    history = calc.get_history()
    for record in history:
        print(f"  {record}")

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            print("运行测试模式")
            main()
        elif sys.argv[1] == "--help":
            print(__doc__)
        else:
            print("未知参数，使用 --help 查看帮助")
    else:
        main() 