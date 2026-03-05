"""
配置文件 - 客户流失分析系统
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class DataConfig:
    """数据配置"""
    raw_data_path: str = "data/customer_churn_data.xlsx"
    processed_data_path: str = "data/processed/"
    cache_dir: str = "cache/"

    # 数据划分比例
    train_ratio: float = 0.7
    test_ratio: float = 0.2
    val_ratio: float = 0.1

    # 随机种子
    random_seed: int = 42

@dataclass
class AlgorithmConfig:
    """算法配置"""
    # 关联规则参数
    min_support: float = 0.02
    min_confidence: float = 0.3
    max_length: int = 3

    # 聚类参数
    kmeans_max_clusters: int = 10
    kprototypes_n_clusters: int = 3

    # 并行计算
    n_jobs: int = -1  # -1表示使用所有CPU核心

    # 缓存设置
    cache_size: int = 128  # 缓存最大条目数

@dataclass
class VisualizationConfig:
    """可视化配置"""
    # 图表样式
    style: str = "seaborn-v0_8"
    figure_size: tuple = (12, 8)
    dpi: int = 300

    # 颜色方案
    color_palette: str = "Set2"

    # 字体设置
    font_size: int = 12
    label_size: int = 10

@dataclass
class PerformanceConfig:
    """性能监控配置"""
    # 内存监控
    max_memory_mb: int = 1024  # 最大内存使用限制(MB)

    # 日志配置
    log_level: str = "INFO"
    log_file: str = "logs/performance.log"

    # 性能阈值
    warning_time: float = 1.0  # 警告阈值(秒)
    critical_time: float = 5.0  # 严重阈值(秒)

class Config:
    """全局配置类"""

    def __init__(self):
        self.data = DataConfig()
        self.algorithm = AlgorithmConfig()
        self.visualization = VisualizationConfig()
        self.performance = PerformanceConfig()

        # 确保目录存在
        os.makedirs(self.data.processed_data_path, exist_ok=True)
        os.makedirs(self.data.cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.performance.log_file), exist_ok=True)

    def update_from_env(self):
        """从环境变量更新配置"""
        # 可以从环境变量读取配置
        pass

# 全局配置实例
config = Config()