"""
特征工程模块 - 高效的特征处理和编码
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.cluster import MeanShift
import logging
from functools import lru_cache
import time
from config.settings import config
from utils.performance import monitor_performance, monitor_performance_simple

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """特征工程处理器 - 支持多种编码方式和并行计算"""

    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.boundaries = {}

    @monitor_performance
    def preprocess_data(self, df: pd.DataFrame, progress_callback=None) -> Dict[str, pd.DataFrame]:
        """
        数据预处理 - 创建三种编码方式的数据集

        Args:
            df: 原始数据DataFrame
            progress_callback: 进度回调函数，用于更新进度条

        Returns:
            Dict: 包含三种编码方式的DataFrame字典
        """
        logger.info("开始数据预处理...")

        # 1. 数据清洗
        cleaned_df = self._clean_data(df)

        # 2. 创建不同编码的数据集
        datasets = {
            'one_hot': self._create_one_hot_encoded(cleaned_df, progress_callback),
            'mixed': self._create_mixed_encoded(cleaned_df, progress_callback),
            'standardized': self._create_standardized(cleaned_df, progress_callback)
        }

        logger.info("数据预处理完成")
        if progress_callback:
            progress_callback()
        return datasets

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗

        Args:
            df: 原始数据

        Returns:
            DataFrame: 清洗后的数据
        """
        # 删除不必要的列（示例）
        drop_cols = ['顾客ID'] if '顾客ID' in df.columns else []
        df_cleaned = df.drop(columns=drop_cols, errors='ignore')

        # 处理缺失值
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().sum() > 0:
                if df_cleaned[col].dtype in ['object', 'category']:
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
                else:
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

        return df_cleaned

    @monitor_performance
    def _create_one_hot_encoded(self, df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
        """
        创建全热编码的数据集

        Args:
            df: 原始数据
            progress_callback: 进度回调函数

        Returns:
            DataFrame: 全热编码后的数据
        """
        logger.info("创建全热编码数据集...")

        # 识别分类列
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 分离数据
        cat_data = df[categorical_cols].copy()
        num_data = df[numeric_cols].copy()

        # 并行处理分类变量编码
        encoded_cat = self._parallel_one_hot_encode(cat_data, progress_callback)

        # 合并数据
        one_hot_df = pd.concat([encoded_cat, num_data], axis=1)

        return one_hot_df

    def _parallel_one_hot_encode(self, cat_data: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
        """
        并行热编码分类变量

        Args:
            cat_data: 分类数据
            progress_callback: 进度回调函数

        Returns:
            DataFrame: 编码后的数据
        """
        encoded_cols = []
        for i, col in enumerate(cat_data.columns):
            # 获取该列的编码器
            encoder = self.encoders.get(col)
            if encoder is None:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore',drop='first')
                encoder.fit(cat_data[[col]])
                self.encoders[col] = encoder

            # 转换数据
            encoded = encoder.transform(cat_data[[col]])
            feature_names = encoder.get_feature_names_out([col])
            encoded_cols.append(pd.DataFrame(encoded, columns=feature_names, index=cat_data.index))

            # 更新进度
            if progress_callback is not None:
                progress_callback()

        return pd.concat(encoded_cols, axis=1)

    @monitor_performance
    def _create_mixed_encoded(self, df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
        """
        创建混合编码的数据集（分类独热编码 + 数值标准化）

        Args:
            df: 原始数据
            progress_callback: 进度回调函数

        Returns:
            DataFrame: 混合编码后的数据
        """
        logger.info("创建混合编码数据集...")

        # 识别列类型
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 处理分类变量
        cat_encoded = self._parallel_one_hot_encode(df[categorical_cols], progress_callback)

        # 处理数值变量
        scaler = self.scalers.get('mixed')
        if scaler is None:
            scaler = MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self.scalers['mixed'] = scaler
        else:
            df[numeric_cols] = scaler.transform(df[numeric_cols])

        # 合并数据
        mixed_df = pd.concat([cat_encoded, df[numeric_cols]], axis=1)

        return mixed_df

    @monitor_performance
    def _create_standardized(self, df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
        """
        创建标准化后的数据集（分类变量保留，数值变量Z-score标准化）

        Args:
            df: 原始数据
            progress_callback: 进度回调函数

        Returns:
            DataFrame: 标准化后的数据
        """
        logger.info("创建标准化数据集...")

        # 识别列类型
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 复制数据
        std_df = df.copy()

        # 标准化数值变量
        scaler = self.scalers.get('standard')
        if scaler is None:
            scaler = StandardScaler()
            std_df[numeric_cols] = scaler.fit_transform(std_df[numeric_cols])
            self.scalers['standard'] = scaler
        else:
            std_df[numeric_cols] = scaler.transform(std_df[numeric_cols])

        return std_df

    @lru_cache(maxsize=config.algorithm.cache_size)
    def adaptive_binning(self, series: pd.Series, n_bins: int = 10) -> pd.Series:
        """
        使用MeanShift进行自适应分箱（带缓存）

        Args:
            series: 要分箱的数据
            n_bins: 分箱数量

        Returns:
            Series: 分箱后的数据
        """
        # 过滤掉NaN值
        clean_series = series.dropna()

        if len(clean_series) == 0:
            return series

        # 使用MeanShift进行聚类
        ms = MeanShift(bandwidth=None, bin_seeding=True, n_jobs=config.algorithm.n_jobs)
        clusters = ms.fit_predict(clean_series.values.reshape(-1, 1))

        # 获取聚类中心
        centers = ms.cluster_centers_.flatten()
        centers.sort()

        # 创建分箱边界
        boundaries = []
        for i in range(len(centers) - 1):
            boundaries.append((centers[i] + centers[i + 1]) / 2)

        # 边界补全
        boundaries = [-np.inf] + boundaries + [np.inf]

        # 分箱
        binned = pd.cut(clean_series, bins=boundaries, labels=False, include_lowest=True)

        # 重新填充NaN
        result = series.copy()
        result[clean_series.index] = binned

        return result

    @monitor_performance
    def batch_feature_processing(self, df: pd.DataFrame, feature_cols: List[str],
                               batch_size: int = 1000) -> pd.DataFrame:
        """
        批量处理特征工程

        Args:
            df: 数据DataFrame
            feature_cols: 要处理的特征列
            batch_size: 批次大小

        Returns:
            DataFrame: 处理后的数据
        """
        logger.info(f"开始批量特征处理，批次大小: {batch_size}")

        processed_features = []
        for i in range(0, len(feature_cols), batch_size):
            batch_features = feature_cols[i:i + batch_size]

            # 对每个批次进行特征处理
            for col in batch_features:
                if df[col].dtype in ['object', 'category']:
                    # 分类变量处理
                    processed_col = pd.get_dummies(df[col], prefix=col)
                else:
                    # 数值变量处理
                    processed_col = self.adaptive_binning(df[col])

                processed_features.append(processed_col)

            # 监控内存
            if i % (batch_size * 2) == 0:
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                logger.info(f"处理进度: {i}/{len(feature_cols)}, 内存使用: {memory_mb:.2f}MB")

        # 合并所有处理后的特征
        if processed_features:
            result_df = pd.concat(processed_features, axis=1)
        else:
            result_df = pd.DataFrame(index=df.index)

        return result_df

    def get_feature_importance(self, df: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """
        快速特征重要性评估

        Args:
            df: 特征数据
            target: 目标变量

        Returns:
            Dict: 特征重要性分数
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import mutual_info_classif

        # 使用随机森林计算特征重要性
        rf = RandomForestClassifier(n_estimators=50, n_jobs=config.algorithm.n_jobs, random_state=config.data.random_seed)
        rf.fit(df, target)
        rf_importance = dict(zip(df.columns, rf.feature_importances_))

        # 使用互信息计算
        mi_scores = mutual_info_classif(df, target, n_jobs=config.algorithm.n_jobs)
        mi_importance = dict(zip(df.columns, mi_scores))

        # 综合评分
        combined_importance = {}
        for feature in df.columns:
            combined_importance[feature] = (rf_importance[feature] + mi_importance.get(feature, 0)) / 2

        # 排序
        sorted_importance = dict(sorted(combined_importance.items(), key=lambda x: x[1], reverse=True))

        return sorted_importance




