"""
数据加载模块 - 支持高效数据加载和缓存
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from functools import lru_cache
import psutil
import time
from config.settings import config
from utils.performance import monitor_performance

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """高效数据加载器 - 支持缓存和内存监控"""

    def __init__(self):
        self.cache_info = {}
        self.memory_monitor = MemoryMonitor()

    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """
        加载数据文件，支持缓存机制

        Args:
            file_path: 数据文件路径，默认使用配置中的路径

        Returns:
            DataFrame: 加载的数据
        """
        if file_path is None:
            file_path = config.data.raw_data_path

        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        # 使用缓存装饰器
        try:
            df = self._cached_load_data(file_path)

            # 记录内存使用
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            logger.info(f"数据加载完成 - 内存使用: {memory_usage:.2f} MB")

            return df

        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise

    @lru_cache(maxsize=config.algorithm.cache_size)
    def _cached_load_data(self, file_path: str) -> pd.DataFrame:
        """
        缓存版本的数据加载函数

        Args:
            file_path: 数据文件路径

        Returns:
            DataFrame: 加载的数据
        """
        logger.info(f"正在加载数据: {file_path}")

        start_time = time.time()

        # 根据文件扩展名选择加载方式
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("不支持的文件格式")

        # 基本信息
        load_time = time.time() - start_time
        shape = df.shape
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

        logger.info(f"数据加载完成 - 形状: {shape}, 耗时: {load_time:.2f}s, 内存: {memory_mb:.2f}MB")

        return df

    def generate_data_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        生成数据质量报告

        Args:
            df: 数据DataFrame

        Returns:
            Dict: 数据质量报告
        """
        report = {
            '基本信息': {
                '行数': df.shape[0],
                '列数': df.shape[1],
                '内存使用(MB)': df.memory_usage(deep=True).sum() / 1024 / 1024
            },
            '缺失值统计': df.isnull().sum().to_dict(),
            '重复值统计': df.duplicated().sum(),
            '数据类型统计': df.dtypes.value_counts().to_dict(),
            '数值列统计': {},
            '分类列统计': {}
        }

        # 数值列统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            report['数值列统计'] = df[numeric_cols].describe().to_dict()

        # 分类列统计
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            for col in cat_cols:
                report['分类列统计'][col] = df[col].value_counts().to_dict()

        return report

    def optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        优化DataFrame内存使用

        Args:
            df: 原始DataFrame

        Returns:
            DataFrame: 优化后的DataFrame
        """
        logger.info("开始优化内存使用...")

        start_memory = df.memory_usage(deep=True).sum() / 1024 / 1024

        # 优化数值类型
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')

        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')

        # 优化分类类型
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # 如果唯一值比例小于50%
                df[col] = df[col].astype('category')

        end_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        reduction = (start_memory - end_memory) / start_memory * 100

        logger.info(f"内存优化完成 - 减少 {reduction:.2f}% ({start_memory:.2f}MB -> {end_memory:.2f}MB)")

        return df

    def batch_process_data(self, df: pd.DataFrame, batch_size: int = 1000) -> list:
        """
        批量处理数据以减少内存压力

        Args:
            df: 要处理的DataFrame
            batch_size: 每批处理的大小

        Returns:
            list: 处理后的批次结果
        """
        logger.info(f"开始批量处理，批次大小: {batch_size}")

        batches = []
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            batches.append(batch)

            # 监控内存使用
            if i % (batch_size * 10) == 0:
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                logger.info(f"处理进度: {i}/{len(df)}, 内存使用: {memory_mb:.2f}MB")

        return batches


class MemoryMonitor:
    """内存监控器"""

    def __init__(self):
        self.thresholds = {
            'warning': config.performance.max_memory_mb * 0.8,
            'critical': config.performance.max_memory_mb
        }

    def check_memory(self) -> Dict[str, float]:
        """
        检查当前内存使用情况

        Returns:
            Dict: 内存使用信息
        """
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # 物理内存使用
            'vms_mb': memory_info.vms / 1024 / 1024,  # 虚拟内存使用
            'percent': process.memory_percent()
        }

    def check_threshold(self) -> Optional[str]:
        """
        检查内存是否超过阈值

        Returns:
            Optional[str]: 警告级别，None表示正常
        """
        memory_info = self.check_memory()

        if memory_info['rss_mb'] > self.thresholds['critical']:
            return 'critical'
        elif memory_info['rss_mb'] > self.thresholds['warning']:
            return 'warning'

        return None


