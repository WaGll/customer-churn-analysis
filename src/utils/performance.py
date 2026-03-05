"""
性能监控工具模块 - 提供函数执行时间、内存和CPU监控
"""

import time
import functools  # 核心模块：用于保留被装饰函数的元数据
import logging
import psutil     # 用于获取内存使用情况
import os

logger = logging.getLogger(__name__)

def monitor_performance(func):
    """
    高级性能监控装饰器
    功能：
    1. 修正了参数透传问题，支持 *args 和 **kwargs
    2. 记录函数运行前后的内存差异 (MB)
    3. 记录函数精确执行时间 (秒)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 1. 记录初始状态
        start_time = time.perf_counter()
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)
        
        # 2. 执行原函数 (透传所有参数，包括 progress_callback)
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logger.error(f"❌ 函数 [{func.__name__}] 执行期间发生异常: {str(e)}")
            raise e
            
        # 3. 记录结束状态
        end_time = time.perf_counter()
        end_mem = process.memory_info().rss / (1024 * 1024)
        
        # 4. 计算指标
        duration = end_time - start_time
        mem_diff = end_mem - start_mem
        
        # 5. 输出格式化日志
        logger.info(f"📊 性能分析 [{func.__name__}]:")
        logger.info(f"   - 耗时: {duration:.4f}s")
        logger.info(f"   - 内存变化: {mem_diff:+.2f} MB (当前占用: {end_mem:.2f} MB)")
        
        return result
        
    return wrapper

def monitor_performance_simple(func):
    """
    简化版性能监控装饰器
    仅记录函数执行时长，适用于轻量级任务
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"⏱️  {func.__name__} 执行耗时: {end - start:.2f}s")
        return result
    return wrapper