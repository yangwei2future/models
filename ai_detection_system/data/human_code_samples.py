# Human Code Samples Dataset
# 人类编写的代码样本 - 真实的编程风格

# 基础函数定义
def add(a, b):
    return a + b

def subtract(x, y):
    return x - y

# 简单的类定义
class Calculator:
    def __init__(self):
        self.result = 0
    
    def multiply(self, a, b):
        return a * b

# 循环和条件
numbers = [1, 2, 3, 4, 5]
for num in numbers:
    if num % 2 == 0:
        print(num)

# 字典操作
config = {'host': 'localhost', 'port': 8080}
host = config.get('host')

# 文件操作
def read_file(filename):
    try:
        with open(filename, 'r') as f:
            return f.read()
    except:
        return None

# 列表推导
squares = [x**2 for x in range(10)]

# 简单算法
def find_max(arr):
    max_val = arr[0]
    for val in arr:
        if val > max_val:
            max_val = val
    return max_val

# 字符串处理
def clean_text(text):
    return text.strip().lower()

# 数据处理
data = [{'name': 'Alice', 'age': 25}, {'name': 'Bob', 'age': 30}]
names = [person['name'] for person in data]

# 错误处理
def safe_divide(a, b):
    if b == 0:
        return 0
    return a / b

# 递归函数
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# 生成器
def count_up_to(max):
    count = 1
    while count <= max:
        yield count
        count += 1

# 装饰器
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Time: {end - start}")
        return result
    return wrapper

# 类继承
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

# 模块导入和使用
import os
import sys
import time
from datetime import datetime

current_time = datetime.now()
file_exists = os.path.exists('test.txt')

# 异常处理
def parse_int(value):
    try:
        return int(value)
    except ValueError:
        return None

# 上下文管理器
def process_file(filename):
    with open(filename, 'w') as f:
        f.write("Hello World")

# 简单的API调用模拟
def get_user_data(user_id):
    users = {1: 'Alice', 2: 'Bob'}
    return users.get(user_id)

# 数据验证
def validate_email(email):
    return '@' in email and '.' in email

# 排序算法
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# 缓存实现
cache = {}
def fibonacci_cached(n):
    if n in cache:
        return cache[n]
    if n <= 1:
        return n
    cache[n] = fibonacci_cached(n-1) + fibonacci_cached(n-2)
    return cache[n]

# 配置管理
DEFAULT_CONFIG = {
    'debug': False,
    'timeout': 30
}

def load_config():
    return DEFAULT_CONFIG.copy()

# 工具函数
def flatten_list(nested_list):
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result

# 简单的状态机
class StateMachine:
    def __init__(self):
        self.state = 'idle'
    
    def start(self):
        self.state = 'running'
    
    def stop(self):
        self.state = 'idle'

# 数据结构
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        return self.items.pop()
    
    def is_empty(self):
        return len(self.items) == 0

# 主函数
if __name__ == '__main__':
    calc = Calculator()
    result = calc.multiply(5, 3)
    print(f"Result: {result}")
    
    # 测试其他功能
    test_data = [3, 1, 4, 1, 5, 9]
    max_value = find_max(test_data)
    sorted_data = bubble_sort(test_data.copy())
    
    print(f"Max: {max_value}")
    print(f"Sorted: {sorted_data}") 