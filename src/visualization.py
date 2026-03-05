"""
可视化模块 - 高效的数据可视化工具
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
from config.settings import config
import plotly.io as pio
from utils.performance import monitor_performance

logging.getLogger('matplotlib.category').setLevel(logging.WARNING)

# 设置Plotly默认主题
pio.templates.default = "plotly_white"


# 1. 先应用基础样式 (这会重置所有 rcParams)
plt.style.use(config.visualization.style)
sns.set_palette(config.visualization.color_palette)

# 2. 立即重新注入中文字体配置 (覆盖样式的默认设置)
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

class DataVisualizer:
    """数据可视化器 - 支持静态和交互式可视化"""

    def __init__(self, output_dir: str = "output/visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.figure_counter = 0

    @monitor_performance
    def create_interactive_dashboard(self,
                                   data: pd.DataFrame,
                                   cluster_results: Dict,
                                   association_rules: Dict,
                                   output_file: str = None) -> str:
        """
        创建交互式分析仪表板

        Args:
            data: 原始数据
            cluster_results: 聚类分析结果
            association_rules: 关联规则结果
            output_file: 输出文件名

        Returns:
            str: HTML文件路径
        """
        logger.info("创建交互式仪表板...")

        if output_file is None:
            output_file = os.path.join(self.output_dir, "interactive_dashboard.html")

        # 创建子图
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '客户分布热力图',
                '流失率vs聚类分布',
                '特征重要性',
                '关联规则网络图',
                '聚类特征雷达图',
                '客户价值分布'
            ),
            specs=[
                [{"type": "heatmap"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatterpolar"}, {"type": "scatterpolar"}]
            ]
        )

        # 1. 客户分布热力图
        if '用户流失标签' in data.columns:
            pivot_table = pd.pivot_table(
                data,
                values='用户流失标签',
                index='年龄分组',
                columns='城市等级',
                aggfunc='mean'
            )
            fig.add_trace(
                go.Heatmap(
                    z=pivot_table.values,
                    x=pivot_table.columns,
                    y=pivot_table.index,
                    colorscale='RdYlBu_r',
                    showscale=True,
                    name='流失率'
                ),
                row=1, col=1
            )

        # 2. 流失率vs聚类分布
        if 'labels' in cluster_results:
            cluster_data = data.copy()
            cluster_data['cluster'] = cluster_results['labels']
            cluster_stats = cluster_data.groupby('cluster')['用户流失标签'].agg(['mean', 'count']).reset_index()
            fig.add_trace(
                go.Scatter(
                    x=cluster_stats['cluster'],
                    y=cluster_stats['mean'] * 100,
                    mode='markers+lines',
                    marker_size=10,
                    name='流失率(%)'
                ),
                row=1, col=2
            )

        # 3. 特征重要性
        if 'cluster_centers' in cluster_results:
            # 使用聚类中心计算特征重要性
            importance = np.std(cluster_results['cluster_centers'], axis=0)
            feature_names = data.select_dtypes(include=[np.number]).columns[:10]  # 取前10个特征
            importance = importance[:len(feature_names)]

            fig.add_trace(
                go.Bar(
                    x=feature_names,
                    y=importance,
                    name='特征重要性'
                ),
                row=2, col=1
            )

        # 4. 关联规则网络图
        if 'rules' in association_rules and not association_rules['rules'].empty:
            # 取前20条规则
            top_rules = association_rules['rules'].head(20)
            fig.add_trace(
                go.Scatter(
                    x=top_rules['support'],
                    y=top_rules['confidence'],
                    mode='markers',
                    marker=dict(
                        size=top_rules['lift'],
                        color=top_rules['lift'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="提升度")
                    ),
                    text=top_rules.apply(lambda x: f"{list(x['antecedents'])} → {list(x['consequents'])}", axis=1),
                    name='关联规则'
                ),
                row=2, col=2
            )

        # 5. 聚类特征雷达图
        if 'cluster_centers' in cluster_results:
            # 准备雷达图数据
            n_clusters = cluster_results['n_clusters']
            features = ['使用平台时间_月', '上月订单数量_单', '订单数量较去年增加_单', '用户关注的主播数量']

            # 标准化特征
            centers = cluster_results['cluster_centers']

            min_vals = centers.min(axis=0)
            max_vals = centers.max(axis=0)
            range_vals = max_vals - min_vals
            range_vals_safe = np.where(range_vals == 0, 1.0, range_vals)  # 分母为0时设为1

    # 归一化
            centers_normalized = (centers - min_vals) / range_vals_safe

    # 全相同特征强制设为 0（雷达图更合理）
            centers_normalized = np.where(range_vals == 0, 0.0, centers_normalized)

            # 添加第一个聚类
            fig.add_trace(
                go.Scatterpolar(
                    r=centers_normalized[0],
                    theta=features,
                    fill='toself',
                    name=f'Cluster 1'
                ),
                row=3, col=2
            )

            # 添加其他聚类
            for i in range(1, min(n_clusters, 4)):  # 最多显示4个聚类
                fig.add_trace(
                    go.Scatterpolar(
                        r=centers_normalized[i],
                        theta=features,
                        fill='toself',
                        name=f'Cluster {i+1}'
                    ),
                    row=3, col=2
                )

        # 6. 客户价值分布
        if '上月订单数量_单' in data.columns and '上月平均折扣金额' in data.columns:
            fig.add_trace(
                go.Histogram2d(
                    x=data['上月订单数量_单'],
                    y=data['上月平均折扣金额'],
                    colorscale='Blues',
                    name='客户分布'
                ),
                row=3, col=1
            )

        # 更新布局
        fig.update_layout(
            title_text="客户流失分析交互式仪表板",
            showlegend=True,
            height=1200,
            width=1600
        )

        # 保存HTML文件
        fig.write_html(output_file)
        logger.info(f"交互式仪表板已保存: {output_file}")

        return output_file

    @monitor_performance
    def create_cluster_profile_plots(self,
                                   data: pd.DataFrame,
                                   cluster_results: Dict,
                                   output_dir: str = None) -> List[str]:
        """
        创建聚类特征分析图表

        Args:
            data: 原始数据
            cluster_results: 聚类结果
            output_dir: 输出目录

        Returns:
            List[str]: 生成的图表文件路径列表
        """
        logger.info("创建聚类特征分析图表...")

        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "cluster_profiles")

        os.makedirs(output_dir, exist_ok=True)
        generated_files = []

        if 'labels' not in cluster_results:
            logger.warning("聚类结果中没有labels字段")
            return generated_files

        labels = cluster_results['labels']
        n_clusters = len(np.unique(labels))

        # 添加聚类标签到数据
        cluster_data = data.copy()
        cluster_data['cluster'] = labels

        # 1. 聚类大小分布
        plt.figure(figsize=(10, 6))
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        plt.pie(cluster_sizes, labels=[f'Cluster {i}' for i in cluster_sizes.index],
                autopct='%1.1f%%', startangle=90)
        plt.title('聚类大小分布')
        pie_file = os.path.join(output_dir, "cluster_distribution.png")
        plt.savefig(pie_file, dpi=config.visualization.dpi, bbox_inches='tight')
        plt.close()
        generated_files.append(pie_file)

        # 2. 数值特征箱线图
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            n_cols = min(4, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

            for i, col in enumerate(numeric_cols[:n_cols * n_rows]):
                sns.boxplot(data=cluster_data, x='cluster', y=col, ax=axes[i])
                axes[i].set_title(f'{col} by Cluster')
                axes[i].set_xlabel('Cluster')

            plt.tight_layout()
            boxplot_file = os.path.join(output_dir, "numeric_features_boxplots.png")
            plt.savefig(boxplot_file, dpi=config.visualization.dpi, bbox_inches='tight')
            plt.close()
            generated_files.append(boxplot_file)

        # 3. 分类特征条形图
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols[:5]:  # 限制最多5个分类变量
                plt.figure(figsize=(12, 6))
                cross_tab = pd.crosstab(cluster_data['cluster'], cluster_data[col])
                cross_tab.div(cross_tab.sum(1), axis=0).plot(kind='bar', stacked=True, ax=plt.gca())
                plt.title(f'{col} Distribution by Cluster')
                plt.xlabel('Cluster')
                plt.ylabel('Proportion')
                plt.xticks(rotation=45)
                plt.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')

                cat_plot_file = os.path.join(output_dir, f"categorical_{col}_by_cluster.png")
                plt.savefig(cat_plot_file, dpi=config.visualization.dpi, bbox_inches='tight')
                plt.close()
                generated_files.append(cat_plot_file)

        # 4. 聚类中心热力图
        if 'cluster_centers' in cluster_results:
            plt.figure(figsize=(12, 8))
            centers = cluster_results['cluster_centers']

            # 确保特征数量匹配
            num_features = min(centers.shape[1], len(numeric_cols))
            centers_df = pd.DataFrame(centers[:, :num_features], columns=numeric_cols[:num_features])

            sns.heatmap(centers_df, annot=True, fmt='.2f', cmap='coolwarm')
            plt.title('Cluster Centers Heatmap')
            plt.xlabel('Features')
            plt.ylabel('Cluster')

            centers_heatmap_file = os.path.join(output_dir, "cluster_centers_heatmap.png")
            plt.savefig(centers_heatmap_file, dpi=config.visualization.dpi, bbox_inches='tight')
            plt.close()
            generated_files.append(centers_heatmap_file)

        logger.info(f"生成 {len(generated_files)} 个聚类分析图表")

        return generated_files

    @monitor_performance
    def create_association_rules_plots(self,
                                     rules_df: pd.DataFrame,
                                     output_dir: str = None) -> List[str]:
        """
        创建关联规则可视化

        Args:
            rules_df: 关联规则DataFrame
            output_dir: 输出目录

        Returns:
            List[str]: 生成的图表文件路径列表
        """
        logger.info("创建关联规则可视化...")

        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "association_rules")

        os.makedirs(output_dir, exist_ok=True)
        generated_files = []

        if rules_df.empty:
            logger.warning("没有关联规则可可视化")
            return generated_files

        # 1. 散点图：支持度 vs 置信度
        plt.figure(figsize=(10, 8))
        plt.scatter(rules_df['support'], rules_df['confidence'],
                   s=rules_df['lift']*50, alpha=0.5, c=rules_df['lift'], cmap='viridis')
        plt.colorbar(label='Lift')
        plt.xlabel('Support')
        plt.ylabel('Confidence')
        plt.title('Association Rules: Support vs Confidence')
        plt.grid(True, alpha=0.3)

        scatter_file = os.path.join(output_dir, "support_vs_confidence.png")
        plt.savefig(scatter_file, dpi=config.visualization.dpi, bbox_inches='tight')
        plt.close()
        generated_files.append(scatter_file)

        # 2. 网络图（简化版）
        top_rules = rules_df.head(20)  # 取前20条规则
        plt.figure(figsize=(12, 10))

        # 创建节点位置
        nodes = set()
        for _, rule in top_rules.iterrows():
            nodes.update(list(rule['antecedents']))
            nodes.update(list(rule['consequents']))

        # 简化的网络可视化
        pos = {node: (i, 0) for i, node in enumerate(nodes)}

        # 绘制边
        for _, rule in top_rules.iterrows():
            for antecedent in rule['antecedents']:
                for consequent in rule['consequents']:
                    plt.plot([pos[antecedent][0], pos[consequent][0]],
                            [pos[antecedent][1], pos[consequent][1]],
                            'b-', alpha=0.6, linewidth=rule['lift'])

        # 绘制节点
        for node, (x, y) in pos.items():
            plt.scatter(x, y, s=100, c='red', zorder=5)
            plt.text(x, y, str(node), fontsize=8, ha='center', va='center')

        plt.title('Association Rules Network (Top 20)')
        plt.axis('off')
        plt.tight_layout()

        network_file = os.path.join(output_dir, "rules_network.png")
        plt.savefig(network_file, dpi=config.visualization.dpi, bbox_inches='tight')
        plt.close()
        generated_files.append(network_file)

        # 3. 按前项分组的提升度条形图
        plt.figure(figsize=(12, 8))
        top_rules_sorted = top_rules.sort_values('lift', ascending=False)

        # 创建标签
        labels = [f"{', '.join(list(a))}" for a in top_rules_sorted['antecedents']]

        plt.bar(range(len(labels)), top_rules_sorted['lift'])
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.ylabel('Lift')
        plt.title('Top Rules by Lift')
        plt.grid(True, alpha=0.3)

        lift_bar_file = os.path.join(output_dir, "rules_lift.png")
        plt.savefig(lift_bar_file, dpi=config.visualization.dpi, bbox_inches='tight')
        plt.close()
        generated_files.append(lift_bar_file)

        logger.info(f"生成 {len(generated_files)} 个关联规则图表")

        return generated_files

    @monitor_performance
    def create_data_distribution_plots(self,
                                     data: pd.DataFrame,
                                     output_dir: str = None) -> List[str]:
        """
        创建数据分布可视化

        Args:
            data: 数据集
            output_dir: 输出目录

        Returns:
            List[str]: 生成的图表文件路径列表
        """
        logger.info("创建数据分布可视化...")

        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "data_distribution")

        os.makedirs(output_dir, exist_ok=True)
        generated_files = []

        # 1. 目标变量分布
        if '用户流失标签' in data.columns:
            plt.figure(figsize=(8, 6))
            data['用户流失标签'].value_counts().plot.pie(autopct='%1.1f%%',
                                                     labels=['未流失', '已流失'],
                                                     colors=['lightblue', 'lightcoral'])
            plt.title('客户流失分布')
            plt.ylabel('')

            target_dist_file = os.path.join(output_dir, "target_distribution.png")
            plt.savefig(target_dist_file, dpi=config.visualization.dpi, bbox_inches='tight')
            plt.close()
            generated_files.append(target_dist_file)

        # 2. 数值特征分布
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

            for i, col in enumerate(numeric_cols):
                sns.histplot(data[col], kde=True, ax=axes[i])
                axes[i].set_title(f'{col} Distribution')

            plt.tight_layout()
            numeric_dist_file = os.path.join(output_dir, "numeric_distribution.png")
            plt.savefig(numeric_dist_file, dpi=config.visualization.dpi, bbox_inches='tight')
            plt.close()
            generated_files.append(numeric_dist_file)

        # 3. 分类变量分布
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols[:5]:  # 限制最多5个分类变量
                plt.figure(figsize=(10, 6))
                data[col].value_counts().plot.bar()
                plt.title(f'{col} Distribution')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45)

                cat_dist_file = os.path.join(output_dir, f"categorical_{col}_distribution.png")
                plt.savefig(cat_dist_file, dpi=config.visualization.dpi, bbox_inches='tight')
                plt.close()
                generated_files.append(cat_dist_file)

        # 4. 相关性热力图
        numeric_data = data.select_dtypes(include=[np.number]).dropna()
        if len(numeric_data.columns) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = numeric_data.corr()
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, square=True)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()

            corr_heatmap_file = os.path.join(output_dir, "correlation_heatmap.png")
            plt.savefig(corr_heatmap_file, dpi=config.visualization.dpi, bbox_inches='tight')
            plt.close()
            generated_files.append(corr_heatmap_file)

        logger.info(f"生成 {len(generated_files)} 个数据分布图表")

        return generated_files

    @monitor_performance
    def export_all_plots_to_html(self, plot_files: List[str], output_file: str = None) -> str:
        """
        将所有图表导出到单个HTML文件

        Args:
            plot_files: 图像文件路径列表
            output_file: 输出HTML文件名

        Returns:
            str: HTML文件路径
        """
        if output_file is None:
            output_file = os.path.join(self.output_dir, "all_visualizations.html")

        logger.info(f"导出 {len(plot_files)} 个图表到HTML...")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>客户流失分析可视化报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .plot-container { margin: 20px 0; text-align: center; }
        .plot-container img { max-width: 100%; height: auto; border: 1px solid #ddd; }
        h1 { color: #333; }
        h2 { color: #666; }
    </style>
</head>
<body>
    <h1>客户流失分析可视化报告</h1>
            """)

            # 按类别分组
            plot_categories = {
                '数据分布': [],
                '聚类分析': [],
                '关联规则': [],
                '交互式仪表板': []
            }

            for file_path in plot_files:
                if 'data_distribution' in file_path:
                    plot_categories['数据分布'].append(file_path)
                elif 'cluster' in file_path:
                    plot_categories['聚类分析'].append(file_path)
                elif 'association' in file_path or 'rules' in file_path:
                    plot_categories['关联规则'].append(file_path)
                elif 'dashboard' in file_path:
                    plot_categories['交互式仪表板'].append(file_path)

            # 写入每个类别的图表
            for category, files in plot_categories.items():
                if files:
                    f.write(f"\n<h2>{category}</h2>\n")
                    for file_path in files:
                        rel_path = os.path.relpath(file_path)
                        f.write(f"""
<div class="plot-container">
    <img src="{rel_path}" alt="{os.path.basename(file_path)}">
</div>
                        """)

            f.write("""
</body>
</html>
            """)

        logger.info(f"可视化报告已保存: {output_file}")
        return output_file


