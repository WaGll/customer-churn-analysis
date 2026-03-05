#!/usr/bin/env python3
"""
性能测试脚本
用于测试优化后的代码性能
"""

import os
import sys
import time
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入模块
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.association_rules import AssociationRuleMiner
from src.clustering import ClusterAnalyzer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceTester:
    """性能测试器"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.results = {}

    def measure_performance(self, func, *args, **kwargs):
        """
        测量函数性能

        Args:
            func: 要测试的函数
            *args, **kwargs: 函数参数

        Returns:
            tuple: (结果, 执行时间, 内存使用)
        """
        # 监控初始状态
        process = psutil.Process()
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB

        try:
            # 执行函数
            result = func(*args, **kwargs)

            # 计算资源使用
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024
            execution_time = end_time - start_time
            memory_increase = end_memory - start_memory

            return result, execution_time, memory_increase

        except Exception as e:
            logger.error(f"函数执行失败: {str(e)}")
            raise

    def test_data_loading(self):
        """测试数据加载性能"""
        logger.info("测试数据加载性能...")

        data_loader = DataLoader()

        # 测试多次取平均
        times = []
        memories = []

        for i in range(3):
            _, t, m = self.measure_performance(data_loader.load_data, self.data_path)
            times.append(t)
            memories.append(m)

        self.results['data_loading'] = {
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'avg_memory': np.mean(memories),
            'max_memory': np.max(memories)
        }

        logger.info(f"数据加载平均耗时: {np.mean(times):.2f}秒")
        logger.info(f"平均内存使用: {np.mean(memories):.2f}MB")

    def test_feature_engineering(self):
        """测试特征工程性能"""
        logger.info("测试特征工程性能...")

        # 加载数据
        data_loader = DataLoader()
        df = data_loader.load_data(self.data_path)

        feature_engineer = FeatureEngineer()

        # 测试三种编码方式
        encodings = ['one_hot', 'mixed', 'standardized']

        for encoding in encodings:
            logger.info(f"测试 {encoding} 编码...")
            _, t, m = self.measure_performance(feature_engineer._create_one_hot_encoded if encoding == 'one_hot' else
                                             feature_engineer._create_mixed_encoded if encoding == 'mixed' else
                                             feature_engineer._create_standardized, df)

            self.results[f'feature_engineering_{encoding}'] = {
                'time': t,
                'memory': m
            }

    def test_association_rules(self):
        """测试关联规则挖掘性能"""
        logger.info("测试关联规则挖掘性能...")

        # 准备数据
        data_loader = DataLoader()
        df = data_loader.load_data(self.data_path)

        feature_engineer = FeatureEngineer()
        datasets = feature_engineer.preprocess_data(df)
        mixed_df = datasets['mixed']

        rule_miner = AssociationRuleMiner()

        # 测试两种算法
        algorithms = ['apriori', 'fp_growth']

        for algorithm in algorithms:
            logger.info(f"测试 {algorithm} 算法...")
            _, t, m = self.measure_performance(rule_miner.mine_association_rules, mixed_df, algorithm)

            self.results[f'association_rules_{algorithm}'] = {
                'time': t,
                'memory': m
            }

    def test_clustering(self):
        """测试聚类分析性能"""
        logger.info("测试聚类分析性能...")

        # 准备数据
        data_loader = DataLoader()
        df = data_loader.load_data(self.data_path)

        feature_engineer = FeatureEngineer()
        datasets = feature_engineer.preprocess_data(df)
        standardized_df = datasets['standardized']

        cluster_analyzer = ClusterAnalyzer()

        # 测试K-Means
        logger.info("测试K-Means聚类...")
        _, t, m = self.measure_performance(cluster_analyzer.kmeans_clustering, standardized_df)

        self.results['clustering_kmeans'] = {
            'time': t,
            'memory': m
        }

        # 测试K-Prototypes
        logger.info("测试K-Prototypes聚类...")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        _, t, m = self.measure_performance(cluster_analyzer.kprototypes_clustering, df, categorical_cols)

        self.results['clustering_kprototypes'] = {
            'time': t,
            'memory': m
        }

    def run_all_tests(self):
        """运行所有性能测试"""
        logger.info("开始性能测试...")
        start_time = time.time()

        # 检查数据文件
        if not os.path.exists(self.data_path):
            logger.error(f"数据文件不存在: {self.data_path}")
            return

        # 运行各项测试
        self.test_data_loading()
        self.test_feature_engineering()
        self.test_association_rules()
        self.test_clustering()

        total_time = time.time() - start_time
        logger.info(f"\n性能测试完成，总耗时: {total_time:.2f}秒")

    def generate_report(self, output_file=None):
        """生成性能测试报告"""
        report = {
            '测试时间': time.strftime('%Y-%m-%d %H:%M:%S'),
            '测试环境': {
                'CPU核心数': psutil.cpu_count(),
                '总内存(GB)': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                'Python版本': sys.version
            },
            '性能结果': self.results,
            '总结': self._analyze_results()
        }

        if output_file:
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"性能报告已保存: {output_file}")

        return report

    def _analyze_results(self):
        """分析性能测试结果"""
        summary = {
            '最快操作': [],
            '最慢操作': [],
            '内存使用大户': []
        }

        for test_name, result in self.results.items():
            if isinstance(result, dict):
                # 找出最快和最慢的操作
                if 'time' in result:
                    summary['最快操作'].append((test_name, result['time']))
                    summary['最慢操作'].append((test_name, result['time']))

                # 找出内存使用大户
                if 'memory' in result:
                    summary['内存使用大户'].append((test_name, result['memory']))

        # 排序
        summary['最快操作'].sort(key=lambda x: x[1])
        summary['最慢操作'].sort(key=lambda x: x[1], reverse=True)
        summary['内存使用大户'].sort(key=lambda x: x[1], reverse=True)

        # 只保留前3个
        summary['最快操作'] = summary['最快操作'][:3]
        summary['最慢操作'] = summary['最慢操作'][:3]
        summary['内存使用大户'] = summary['内存使用大户'][:3]

        return summary

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='性能测试工具')
    parser.add_argument('--data-path', '-d', default='data/customer_churn_data.xlsx',
                       help='数据文件路径')
    parser.add_argument('--output', '-o', default='performance_report.json',
                       help='输出报告文件')

    args = parser.parse_args()

    # 创建测试器
    tester = PerformanceTester(args.data_path)

    # 运行测试
    tester.run_all_tests()

    # 生成报告
    report = tester.generate_report(args.output)

    # 打印摘要
    print("\n" + "="*50)
    print("性能测试摘要")
    print("="*50)
    print(f"测试完成时间: {report['测试时间']}")
    print("\n最快操作:")
    for name, time in report['总结']['最快操作']:
        print(f"  - {name}: {time:.2f}秒")
    print("\n最慢操作:")
    for name, time in report['总结']['最慢操作']:
        print(f"  - {name}: {time:.2f}秒")
    print("\n内存使用大户:")
    for name, memory in report['总结']['内存使用大户']:
        print(f"  - {name}: {memory:.2f}MB")

if __name__ == '__main__':
    main()