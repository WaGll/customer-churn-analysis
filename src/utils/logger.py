"""
日志工具模块

提供统一的日志配置和管理功能。
"""

import logging
import logging.config
import os
from pathlib import Path

from config.settings import config


def setup_logging(log_level=None, log_file=None):
    """
    设置日志配置

    Args:
        log_level: 日志级别
        log_file: 日志文件路径
    """
    if log_level is None:
        log_level = config.performance.log_level

    # 创建日志目录
    log_dir = Path(log_file).parent if log_file else Path("logs")
    log_dir.mkdir(exist_ok=True)

    # 日志配置
    log_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
            }
        },
        'handlers': {
            'console': {
                'level': log_level,
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'level': log_level,
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': log_file or 'logs/app.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'formatter': 'detailed'
            },
            'performance': {
                'level': log_level,
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': log_file or 'logs/performance.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'formatter': 'detailed'
            }
        },
        'loggers': {
            '': {
                'handlers': ['console', 'file'],
                'level': log_level,
                'propagate': False
            },
            'performance': {
                'handlers': ['performance'],
                'level': log_level,
                'propagate': False
            }
        }
    }

    # 配置日志
    logging.config.dictConfig(log_config)

    return logging.getLogger(__name__)