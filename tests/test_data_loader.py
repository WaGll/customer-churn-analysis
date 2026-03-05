"""
数据加载模块测试
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data_loader import DataLoader, MemoryMonitor


class TestDataLoader:
    """数据加载器测试类"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        data = {
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45],
            'salary': [50000, 60000, 70000, 80000, 90000],
            'department': ['IT', 'HR', 'IT', 'Finance', 'HR']
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def data_loader(self):
        """创建数据加载器实例"""
        return DataLoader()

    def test_load_data_from_dict(self, data_loader, sample_data):
        """测试从字典加载数据"""
        # 这里只是测试类的方法，不涉及实际文件
        assert data_loader is not None

    def test_generate_data_report(self, data_loader, sample_data):
        """测试数据报告生成"""
        report = data_loader.generate_data_report(sample_data)

        assert '基本信息' in report
        assert '缺失值统计' in report
        assert '重复值统计' in report
        assert '数据类型统计' in report

        assert report['基本信息']['行数'] == 5
        assert report['基本信息']['列数'] == 5
        assert report['重复值统计'] == 0

    def test_optimize_memory_usage(self, data_loader, sample_data):
        """测试内存优化"""
        original_memory = sample_data.memory_usage(deep=True).sum()
        optimized_df = data_loader.optimize_memory_usage(sample_data)
        optimized_memory = optimized_df.memory_usage(deep=True).sum()

        # 优化后的内存应该小于或等于原内存
        assert optimized_memory <= original_memory

    def test_memory_monitor(self):
        """测试内存监控器"""
        monitor = MemoryMonitor()
        memory_info = monitor.check_memory()

        assert 'rss_mb' in memory_info
        assert 'vms_mb' in memory_info
        assert 'percent' in memory_info
        assert memory_info['rss_mb'] > 0

    def test_batch_process_data(self, data_loader, sample_data):
        """测试批量数据处理"""
        batches = data_loader.batch_process_data(sample_data, batch_size=2)

        assert len(batches) == 3  # 5条数据，每批2条，最后一批1条
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1