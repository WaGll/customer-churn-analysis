#!/usr/bin/env python3
"""
客户流失分析主程序
高性能版本，支持Docker部署
"""

import os
import sys
import logging
import argparse
import traceback
from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入模块
from src.data_loader import DataLoader, monitor_performance
from src.feature_engineering import FeatureEngineer
from src.association_rules import AssociationRuleMiner
from src.clustering import ClusterAnalyzer
from src.visualization import DataVisualizer

# 配置日志
def setup_logging(log_level='INFO', log_file=None):
    """设置日志配置"""
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)

    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))

    # 配置根日志器
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=[console_handler] if not log_file else [console_handler, file_handler]
    )

    return logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='客户流失分析系统')

    parser.add_argument('--data-path', '-d', default='data/customer_churn_data.xlsx',
                       help='数据文件路径')
    parser.add_argument('--output-dir', '-o', default='output',
                       help='输出目录')
    parser.add_argument('--log-level', '-l', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    parser.add_argument('--log-file',
                       help='日志文件路径')
    parser.add_argument('--no-visualization', action='store_true',
                       help='跳过可视化生成')
    parser.add_argument('--algorithm', '-a', default='fp_growth',
                       choices=['apriori', 'fp_growth'],
                       help='关联规则算法')
    parser.add_argument('--max-clusters', '-k', type=int, default=5,
                       help='最大聚类数')

    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 设置日志
    logger = setup_logging(args.log_level, args.log_file)
    logger.info("=" * 60)
    logger.info("客户流失分析系统启动")
    logger.info("=" * 60)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"数据路径: {args.data_path}")
    logger.info(f"输出目录: {args.output_dir}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    vis_subdir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_subdir, exist_ok=True)

    try:
        # 使用 tqdm 进度条
        with tqdm(total=5, desc="整体分析进度", unit="模块") as pbar:
            # 1. 加载数据
            logger.info("\n[步骤 1/5] 加载数据...")
            start_time = time.time()

            data_loader = DataLoader()
            df = data_loader.load_data(args.data_path)

            # 生成数据质量报告
            data_report = data_loader.generate_data_report(df)
            data_loader.optimize_memory_usage(df)

            # 保存数据报告
            import json
            import pandas as pd

            # 处理数据类型问题 - 更安全的转换
            def safe_serialize(obj):
                if isinstance(obj, dict):
                    return {str(k): safe_serialize(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [safe_serialize(item) for item in obj]
                elif isinstance(obj, pd.Series):
                    return safe_serialize(obj.to_dict())
                elif isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                else:
                    return str(obj)
            
            results_dir = os.path.join(args.output_dir, 'results')
            os.makedirs(results_dir, exist_ok=True)

            report_path = os.path.join(results_dir, 'data_report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(safe_serialize(data_report), f, ensure_ascii=False, indent=2)

            logger.info(f"数据加载完成，耗时: {time.time() - start_time:.2f}秒")
            pbar.update(1)

            # 2. 特征工程
            logger.info("\n[步骤 2/5] 特征工程...")
            start_time = time.time()

            feature_engineer = FeatureEngineer()

            # 创建子进度条 - 处理每一列
            columns = df.columns.tolist()
            with tqdm(total=len(columns), desc="特征工程", unit="列", leave=False) as sub_pbar:
                datasets = feature_engineer.preprocess_data(df, progress_callback=lambda: sub_pbar.update(1))

            logger.info(f"特征工程完成，生成了 {len(datasets)} 种编码方式的数据集")
            logger.info(f"耗时: {time.time() - start_time:.2f}秒")

            # 保存处理后的数据
            for name, dataset in datasets.items():
                output_file = os.path.join(results_dir, f'data_{name}.csv')
                dataset.to_csv(output_file, index=False)
                logger.info(f"保存 {name} 数据集到: {output_file}")

            pbar.update(1)

            # 3. 关联规则挖掘
            logger.info("\n[步骤 3/5] 关联规则挖掘...")
            start_time = time.time()

            rule_miner = AssociationRuleMiner()

            # 使用混合编码的数据
            mixed_df = datasets['mixed']

            # 转换为布尔类型用于关联规则
            # 只选择出现频率较高的特征
            threshold = mixed_df.mean().quantile(0.5)  # 取中位数作为阈值
            bool_df = mixed_df > threshold

            # 参数优化 - 使用更保守的参数范围
            logger.info("正在优化参数...")
            with tqdm(total=3, desc="参数优化", unit="次", leave=False) as sub_pbar:
                opt_result = rule_miner.optimize_parameters(
                    bool_df,
                    algorithm=args.algorithm,
                    param_space={
                        'min_support': [0.1, 0.15, 0.2],  # 提高最小支持度
                        'min_confidence': [0.5, 0.6, 0.7]  # 提高最小置信度
                    },
                    progress_callback=lambda: sub_pbar.update(1)
                )

            # 使用最优参数挖掘规则
            rule_result = rule_miner.mine_association_rules(
                bool_df,
                algorithm=args.algorithm,
                min_support=opt_result['optimal_params']['min_support'],
                min_confidence=opt_result['optimal_params']['min_confidence']
            )

            # 找到流失相关规则
            loss_rules = rule_miner.find_loss_related_rules(rule_result['rules'])

            # 生成规则质量报告
            rule_report = rule_miner.rule_quality_report(rule_result['rules'])

            logger.info(f"关联规则挖掘完成，共生成 {rule_result['rule_count']} 条规则")
            logger.info(f"其中流失相关规则 {len(loss_rules)} 条")
            logger.info(f"耗时: {time.time() - start_time:.2f}秒")

            # 保存关联规则结果
            rule_result['loss_rules'] = loss_rules
            rule_result['report'] = rule_report

            

            rules_path = os.path.join(results_dir, 'association_rules.json')

            with open(rules_path, 'w', encoding='utf-8') as f:
                json.dump(rule_result, f, ensure_ascii=False, indent=2, default=str)

            pbar.update(1)

            # 4. 聚类分析
            logger.info("\n[步骤 4/5] 聚类分析...")
            start_time = time.time()

            cluster_analyzer = ClusterAnalyzer()

            # K-Means聚类 - 使用One-Hot编码后的数据（所有特征都是数值）
            kmeans_result = cluster_analyzer.kmeans_clustering(
                datasets['one_hot'],
                max_clusters=args.max_clusters
            )

            # K-Prototypes聚类 - 使用原始数据
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                kproto_result = cluster_analyzer.kprototypes_clustering(
                    datasets['standardized'],
                    categorical_cols=categorical_cols,
                    max_clusters=args.max_clusters
                )
            else:
                logger.warning("没有分类列，跳过K-Prototypes聚类")
                kproto_result = {'n_clusters': 0, 'labels': []}

            # 聚类稳定性测试 - 使用One-Hot编码数据
            with tqdm(total=5, desc="稳定性测试", unit="次", leave=False) as sub_pbar:
                stability_kmeans = cluster_analyzer.stability_test(
                    datasets['one_hot'],
                    'kmeans',
                    n_iterations=5,
                    progress_callback=lambda: sub_pbar.update(1)
                )

            # 分析聚类特征
            cluster_characteristics = cluster_analyzer.analyze_cluster_characteristics(
                df,
                kmeans_result['labels']
            )

            logger.info(f"聚类分析完成 - K-Means: {kmeans_result['n_clusters']} 个聚类")
            if kproto_result['n_clusters'] > 0:
                logger.info(f"K-Prototypes: {kproto_result['n_clusters']} 个聚类")
            logger.info(f"稳定性: {stability_kmeans['stability_level']}")
            logger.info(f"耗时: {time.time() - start_time:.2f}秒")

            # 保存聚类结果
            cluster_results = {
                'kmeans': kmeans_result,
                'kprototypes': kproto_result,
                'stability': stability_kmeans,
                'characteristics': cluster_characteristics
            }

            cluster_path = os.path.join(results_dir, 'clustering_results.json')
            with open(cluster_path, 'w', encoding='utf-8') as f:
                json.dump(cluster_results, f, ensure_ascii=False, indent=2, default=str)

            pbar.update(1)
            
            
            # 5. 可视化
            if not args.no_visualization:
                logger.info("\n[步骤 5/5] 生成可视化...")
                start_time = time.time()

                visualizer = DataVisualizer()

                # 创建交互式仪表板
                dashboard_path = visualizer.create_interactive_dashboard(
                    df,
                    kmeans_result,
                    rule_result
                )

                # 创建聚类分析图表
                cluster_plots = visualizer.create_cluster_profile_plots(
                    df,
                    kmeans_result
                )

                # 创建关联规则图表
                rule_plots = visualizer.create_association_rules_plots(
                    rule_result['rules']
                )

                # 创建数据分布图表
                dist_plots = visualizer.create_data_distribution_plots(df)

                # 导出所有图表
                all_plots = cluster_plots + rule_plots + dist_plots
                if dashboard_path:
                    all_plots.append(dashboard_path)

                visualizer.export_all_plots_to_html(all_plots,
                    os.path.join(vis_subdir, 'visualization_report.html'))

                logger.info(f"可视化生成完成，共生成 {len(all_plots)} 个图表")
                logger.info(f"耗时: {time.time() - start_time:.2f}秒")

            pbar.update(1)

        # 生成总结报告
        logger.info("\n" + "=" * 60)
        logger.info("分析完成！")
        logger.info("=" * 60)
        logger.info(f"总耗时: {time.time() - start_time:.2f}秒")
        logger.info(f"输出目录: {args.output_dir}")

        if not args.no_visualization:
            logger.info("访问 http://localhost 查看可视化报告")

        # 输出关键发现
        logger.info("\n关键发现:")
        if loss_rules:
            logger.info(f"1. 发现 {len(loss_rules)} 条与客户流失强相关的关联规则")
            for rule in loss_rules[:3]:
                logger.info(f"   - {rule['rule']}: 提升度={rule['lift']:.2f}")

        if 'kmeans' in cluster_results:
            logger.info(f"2. 客户分为 {cluster_results['kmeans']['n_clusters']} 个群体")
            logger.info(f"3. 聚类稳定性: {stability_kmeans['stability_level']}")

    except Exception as e:
        logger.error(f"分析过程中发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()