"""
辅助函数模块

包含了各种通用的辅助函数。
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
import json
import pickle
import os
from datetime import datetime


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2):
    """
    保存数据为JSON文件

    Args:
        data: 要保存的数据
        filepath: 文件路径
        indent: 缩进空格数
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    从JSON文件加载数据

    Args:
        filepath: 文件路径

    Returns:
        加载的数据
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: str):
    """
    保存数据为pickle文件

    Args:
        data: 要保存的数据
        filepath: 文件路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: str) -> Any:
    """
    从pickle文件加载数据

    Args:
        filepath: 文件路径

    Returns:
        加载的数据
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def create_timestamp_filename(prefix: str, suffix: str = '.json') -> str:
    """
    创建带时间戳的文件名

    Args:
        prefix: 文件名前缀
        suffix: 文件后缀

    Returns:
        完整的文件名
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}{suffix}"


def get_memory_usage(obj: Any) -> float:
    """
    获取对象的内存使用量（MB）

    Args:
        obj: 要检查的对象

    Returns:
        内存使用量（MB）
    """
    import sys
    return sys.getsizeof(obj) / 1024 / 1024


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全除法，避免除零错误

    Args:
        numerator: 分子
        denominator: 分母
        default: 除零时的默认值

    Returns:
        除法结果
    """
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default


def format_number(num: Union[int, float], precision: int = 2) -> str:
    """
    格式化数字

    Args:
        num: 要格式化的数字
        precision: 小数位数

    Returns:
        格式化后的字符串
    """
    if isinstance(num, int):
        return f"{num:,}"
    else:
        return f"{num:,.{precision}f}"


def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    验证DataFrame是否包含必需的列

    Args:
        df: DataFrame
        required_columns: 必需的列名列表

    Returns:
        是否包含所有必需的列
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"缺少必需的列: {missing_columns}")
    return True


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    获取DataFrame中的数值列

    Args:
        df: DataFrame

    Returns:
        数值列名列表
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    获取DataFrame中的分类列

    Args:
        df: DataFrame

    Returns:
        分类列名列表
    """
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def calculate_percentiles(series: pd.Series, percentiles: List[float] = None) -> Dict[str, float]:
    """
    计算序列的分位数

    Args:
        series: 数据序列
        percentiles: 要计算的分位数列表

    Returns:
        分位数字典
    """
    if percentiles is None:
        percentiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

    result = {}
    for p in percentiles:
        result[f"p{int(p*100)}"] = series.quantile(p)

    return result


def detect_outliers_iqr(series: pd.Series, threshold: float = 1.5) -> pd.Series:
    """
    使用IQR方法检测异常值

    Args:
        series: 数据序列
        threshold: IQR倍数阈值

    Returns:
        异常值掩码
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    return (series < lower_bound) | (series > upper_bound)


def flatten_nested_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    展平嵌套字典

    Args:
        d: 嵌套字典
        parent_key: 父键
        sep: 分隔符

    Returns:
        展平后的字典
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)