"""
聚类分析模块 - 优化的聚类算法实现
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
from config.settings import config
import psutil
from utils.performance import monitor_performance

logger = logging.getLogger(__name__)

class ClusterAnalyzer:
    """聚类分析器 - 支持多种聚类算法和评估方法"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.best_k = None

    @monitor_performance
    def kmeans_clustering(self,
                         data: pd.DataFrame,
                         max_clusters: int = None,
                         random_state: int = None) -> Dict[str, Any]:
        """
        K-Means聚类分析

        Args:
            data: 数据集
            max_clusters: 最大聚类数
            random_state: 随机种子

        Returns:
            Dict: 聚类结果
        """
        if max_clusters is None:
            max_clusters = config.algorithm.kmeans_max_clusters
        if random_state is None:
            random_state = config.data.random_seed

        logger.info("开始K-Means聚类分析...")

        # 标准化数据
        scaler = self.scalers.get('kmeans')
        if scaler is None:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            self.scalers['kmeans'] = scaler
        else:
            data_scaled = scaler.transform(data)

        # 确定最佳聚类数
        best_k = self._find_best_k(data_scaled, max_clusters)

        # 执行聚类
        kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=20)
        labels = kmeans.fit_predict(data_scaled)

        # 评估聚类结果
        metrics = self._evaluate_clustering(data_scaled, labels)

        # 存储模型
        self.models['kmeans'] = kmeans

        result = {
            'algorithm': 'kmeans',
            'n_clusters': best_k,
            'labels': labels,
            'cluster_centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'metrics': metrics,
            'data_scaled': data_scaled
        }

        logger.info(f"K-Means聚类完成 - 最佳K值: {best_k}")

        return result

    @monitor_performance
    def kprototypes_clustering(self,
                             data: pd.DataFrame,
                             categorical_cols: List[str],
                             max_clusters: int = None,
                             random_state: int = None) -> Dict[str, Any]:
        """
        K-Prototypes聚类分析（混合数据类型）

        Args:
            data: 数据集
            categorical_cols: 分类列名列表
            max_clusters: 最大聚类数
            random_state: 随机种子

        Returns:
            Dict: 聚类结果
        """
        if max_clusters is None:
            max_clusters = config.algorithm.kprototypes_n_clusters
        if random_state is None:
            random_state = config.data.random_seed

        logger.info("开始K-Prototypes聚类分析...")

        #检查分类列
        missing = [c for c in categorical_cols if c not in data.columns]
        if missing:
            logger.error(f"分类列不存在: {missing}")
            return {'error': f"无效分类列: {missing}"}

        # 分离数值型和分类型数据序列号
        numeric_cols = [col for col in data.columns if col not in categorical_cols]
        #data_numeric = data[numeric_cols].values
        #data_categorical = data[categorical_cols].values
        cat_indices = [data.columns.get_loc(col) for col in categorical_cols]

    # 3. 转成 numpy array
        X = data.values.astype(object)   # 重要：用 object dtype 避免数值列被强制转float导致类别列出问题

    # 4. 确定最佳

        # 确定最佳聚类数
        best_k = self._find_best_k_prototype(X, cat_indices,max_k=max_clusters)

        # 执行聚类
        try:
            kproto = KPrototypes(n_clusters=best_k, init='Cao', random_state=random_state, n_jobs=config.algorithm.n_jobs)

            labels = kproto.fit_predict(X, categorical=cat_indices)
        except Exception as e:
            logger.error(f"K-Prototypes聚类失败: {str(e)}")
            logger.warning("跳过K-Prototypes聚类，返回空结果")
            return {
                'algorithm': 'kprototypes',
                'n_clusters': 0,
                'labels': [],
                'cluster_centers': None,
                'cost': None,
                'metrics': {},
                'categorical_cols': categorical_cols,
                'numeric_cols': numeric_cols,
                'error': str(e)
            }

        # 评估聚类结果
        metrics = self._evaluate_clustering_mixed(data, labels)

        # 存储模型
        self.models['kprototypes'] = kproto

        result = {
            'algorithm': 'kprototypes',
            'n_clusters': best_k,
            'labels': labels,
            'cluster_centers': kproto.cluster_centroids_,
            'cost': kproto.cost_,
            'metrics': metrics,
            'categorical_cols': categorical_cols,
            'numeric_cols': numeric_cols
        }

        logger.info(f"K-Prototypes聚类完成 - 最佳K值: {best_k}")

        return result

    def _find_best_k(self, data: np.ndarray, max_k: int) -> int:
        """
        k-means使用肘部法和轮廓系数确定最佳K值

        Args:
            data: 标准化后的数据
            max_k: 最大K值

        Returns:
            int: 最佳K值
        """
        wcss = []
        silhouette_scores = []
        ch_scores = []

        for k in range(2, min(max_k + 1, len(data) - 1)):
            try:
                kmeans = KMeans(n_clusters=k, random_state=config.data.random_seed, n_init=10)
                labels = kmeans.fit_predict(data)

                wcss.append(kmeans.inertia_)

                if len(np.unique(labels)) > 1:
                    silhouette_scores.append(silhouette_score(data, labels))
                    ch_scores.append(calinski_harabasz_score(data, labels))
                else:
                    silhouette_scores.append(0)
                    ch_scores.append(0)

            except Exception as e:
                logger.warning(f"K={k} 聚类失败: {str(e)}")
                wcss.append(np.inf)
                silhouette_scores.append(0)
                ch_scores.append(0)
                
        logger.info(f"WCSS 值列表: {wcss}")
        logger.info(f"轮廓系数列表: {silhouette_scores}")


        


        # 检查轮廓系数
        if silhouette_scores and max(silhouette_scores) > 0.1:   # 只要有有效轮廓系数就优先使用
            max_sil_idx = np.argmax(silhouette_scores)
            best_k = max_sil_idx + 2
            logger.info(f"✅ 采用轮廓系数最高 K={best_k} (silhouette={silhouette_scores[max_sil_idx]:.4f})")
                
        else:
             # 肘部法
             elbow_idx = self._find_elbow_point(wcss)
             best_k = elbow_idx + 2
             logger.warning(f"轮廓系数无效，退回肘部法 K={best_k}")

        
        

        
        return best_k


    def _find_best_k_prototype(self, X: np.ndarray, cat_indices: List[int], max_k: int) -> int:
     """
    为 K-Prototypes 确定最佳 K 值（使用统一的 X + cat_indices 调用方式）
    
    Args:
        X: 完整数据的 numpy array (dtype=object)
        cat_indices: 分类列的索引列表（从 0 开始）
        max_k: 最大 K 值
    
    Returns:
        int: 建议的最佳 K 值
    """
     costs = []
     valid_ks = []

     logger.info(f"寻找最佳 K，范围 2 ~ {max_k}，分类列索引: {cat_indices}")

     for k in range(2, max_k + 1):
        try:
            kproto = KPrototypes(
                n_clusters=k,
                init='Cao',               # 改用 'Cao' 通常更稳定
                n_init=2,                 # 多跑几次初始化，减少随机性
                random_state=config.data.random_seed,
                n_jobs=1                  # 调试时先设 1，避免多进程问题
            )
            
            # 关键：统一使用 fit_predict(X, categorical=cat_indices)
            _ = kproto.fit_predict(X, categorical=cat_indices)
            cost = kproto.cost_
            
            if np.isfinite(cost):
                costs.append(cost)
                valid_ks.append(k)
                logger.debug(f"K={k} → cost={cost:.4f}")
            else:
                logger.warning(f"K={k} cost 为无效值 ({cost})，跳过")
                
        except Exception as e:
            logger.warning(f"K={k} K-Prototypes 失败: {str(e)}")
            continue

     if not costs:
         logger.error("所有 K 值均失败，无法确定最佳 K，使用默认值 3")
         return 3

    # 肘部法找拐点（使用已有的 _find_elbow_point，但注意索引从 0 开始）
     rel_idx = self._find_elbow_point(costs)
     rel_idx = min(rel_idx, len(valid_ks) - 1)
     best_k = valid_ks[rel_idx]

     logger.info(f"K-Prototypes 最佳 K = {best_k} (cost 列表长度={len(costs)})")

     return best_k

    def _find_elbow_point(self, values: List[float]) -> int:
        """
        找到肘部点

        Args:
            values: WCSS或成本值列表

        Returns:
            int: 肘部点对应的K值
        """
        if len(values) < 3:
            return 0

        # 计算每个点到直线的距离
        values = np.array(values, dtype=float)
        valid_mask = np.isfinite(values)
        if valid_mask.sum() < 3:
            return 0
        
        x = np.arange(len(values))[valid_mask]
        y = values[valid_mask]

        

        # 计算所有点到首尾连线的距离
        distances = []
        p1 = np.array([x[0], y[0]])
        p2 = np.array([x[-1], y[-1]])

        for i in range(len(x)):
            p3 = np.array([x[i], y[i]])
            line_vec = p2 - p1
            point_vec = p3 - p1

            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq == 0:
                distances.append(0)
                continue
            # 计算点到直线的距离
            t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
            projection = p1 + t * line_vec
            dist = np.linalg.norm(p3 - projection)
            distances.append(dist)
        elbow_idx = np.argmax(distances)
        logger.debug(f"肘部距离: {distances}")
        return int(np.argmax(distances))   

    def _evaluate_clustering(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        评估聚类质量

        Args:
            data: 数据
            labels: 聚类标签

        Returns:
            Dict: 评估指标
        """
        metrics = {}

        # 轮廓系数
        if len(np.unique(labels)) > 1:
            metrics['silhouette_score'] = silhouette_score(data, labels)
        else:
            metrics['silhouette_score'] = 0

        # Calinski-Harabasz指数
        if len(np.unique(labels)) > 1:
            metrics['calinski_harabasz'] = calinski_harabasz_score(data, labels)
        else:
            metrics['calinski_harabasz'] = 0

        # Davies-Bouldin指数
        if len(np.unique(labels)) > 1:
            metrics['davies_bouldin'] = davies_bouldin_score(data, labels)
        else:
            metrics['davies_bouldin'] = np.inf

        return metrics

    def _evaluate_clustering_mixed(self, data: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
        """
        评估混合数据聚类质量

        Args:
            data: 原始数据
            labels: 聚类标签

        Returns:
            Dict: 评估指标
        """
        metrics = {}

        # 内部指标（基于混合数据）
        try:
            # 转换分类变量为数值
            data_numeric = data.copy()
            for col in data.select_dtypes(include=['object', 'category']).columns:
                data_numeric[col] = pd.factorize(data[col])[0]

            # 计算轮廓系数
            if len(np.unique(labels)) > 1:
                metrics['silhouette_score'] = silhouette_score(data_numeric.fillna(0), labels)
            else:
                metrics['silhouette_score'] = 0

        except Exception as e:
            logger.warning(f"混合数据聚类评估失败: {str(e)}")
            metrics['silhouette_score'] = 0

        # Calinski-Harabasz指数（仅数值部分）
        numeric_data = data.select_dtypes(include=[np.number]).fillna(0)
        if len(np.unique(labels)) > 1:
            metrics['calinski_harabasz'] = calinski_harabasz_score(numeric_data, labels)
        else:
            metrics['calinski_harabasz'] = 0

        # Davies-Bouldin指数
        if len(np.unique(labels)) > 1:
            metrics['davies_bouldin'] = davies_bouldin_score(numeric_data, labels)
        else:
            metrics['davies_bouldin'] = np.inf

        return metrics

    @monitor_performance
    def stability_test(self, data: pd.DataFrame, algorithm: str, n_iterations: int = 10, progress_callback=None) -> Dict[str, Any]:
        """
        聚类稳定性测试

        Args:
            data: 数据集
            algorithm: 聚类算法
            n_iterations: 迭代次数
            progress_callback: 进度回调函数

        Returns:
            Dict: 稳定性测试结果
        """
        logger.info(f"开始{algorithm}聚类稳定性测试...")

        labels_list = []
        metrics_list = []

        for i in range(n_iterations):
            # 采样80%的数据
            sample_idx = np.random.choice(len(data), int(len(data) * 0.8), replace=False)
            sample_data = data.iloc[sample_idx]

            if algorithm == 'kmeans':
                result = self.kmeans_clustering(sample_data, max_clusters=5)
            elif algorithm == 'kprototypes':
                categorical_cols = sample_data.select_dtypes(include=['object', 'category']).columns.tolist()
                result = self.kprototypes_clustering(sample_data, categorical_cols, max_clusters=5)
            else:
                continue

            labels_list.append(result['labels'])
            metrics_list.append(result['metrics'])

            # 更新进度
            if progress_callback is not None:
                progress_callback()

        # 计算一致性
        consistency_scores = self._calculate_consistency(labels_list)

        stability_report = {
            'algorithm': algorithm,
            'n_iterations': n_iterations,
            'average_metrics': {
                'silhouette': np.mean([m['silhouette_score'] for m in metrics_list]),
                'calinski_harabasz': np.mean([m['calinski_harabasz'] for m in metrics_list]),
                'davies_bouldin': np.mean([m['davies_bouldin'] for m in metrics_list])
            },
            'consistency_scores': consistency_scores,
            'stability_level': self._assess_stability(consistency_scores)
        }

        logger.info(f"{algorithm}聚类稳定性测试完成 - 稳定性等级: {stability_report['stability_level']}")

        return stability_report

    def _calculate_consistency(self, labels_list: List[np.ndarray]) -> Dict[str, float]:
        """
        计算聚类结果的一致性

        Args:
            labels_list: 多次聚类的标签列表

        Returns:
            Dict: 一致性分数
        """
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        n_iterations = len(labels_list)
        consistency_scores = {
            'rand_index': 0,
            'mutual_info': 0
        }

        # 计算所有配对的一致性
        rand_scores = []
        mi_scores = []

        for i in range(n_iterations):
            for j in range(i + 1, n_iterations):
                # 确保标签对齐
                len_i = len(labels_list[i])
                len_j = len(labels_list[j])
                min_len = min(len_i, len_j)

                rand_score = adjusted_rand_score(
                    labels_list[i][:min_len],
                    labels_list[j][:min_len]
                )
                mi_score = normalized_mutual_info_score(
                    labels_list[i][:min_len],
                    labels_list[j][:min_len]
                )

                rand_scores.append(rand_score)
                mi_scores.append(mi_score)

        consistency_scores['rand_index'] = np.mean(rand_scores)
        consistency_scores['mutual_info'] = np.mean(mi_scores)

        return consistency_scores

    def _assess_stability(self, consistency_scores: Dict[str, float]) -> str:
        """
        评估稳定性等级

        Args:
            consistency_scores: 一致性分数

        Returns:
            str: 稳定性等级
        """
        avg_score = (consistency_scores['rand_index'] + consistency_scores['mutual_info']) / 2

        if avg_score > 0.8:
            return "非常稳定"
        elif avg_score > 0.6:
            return "稳定"
        elif avg_score > 0.4:
            return "中等"
        elif avg_score > 0.2:
            return "不稳定"
        else:
            return "非常不稳定"

    def analyze_cluster_characteristics(self, data: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """
        分析聚类特征

        Args:
            data: 原始数据
            labels: 聚类标签

        Returns:
            Dict: 聚类特征分析结果
        """
        cluster_characteristics = {}

        for cluster_id in np.unique(labels):
            cluster_data = data[labels == cluster_id]

            cluster_characteristics[f'cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(data) * 100,
                'features': {}
            }

            # 分析每个特征的统计信息
            for col in data.columns:
                if data[col].dtype in ['object', 'category']:
                    # 分类变量
                    value_counts = cluster_data[col].value_counts()
                    cluster_characteristics[f'cluster_{cluster_id}']['features'][col] = {
                        'type': 'categorical',
                        'top_values': value_counts.head(3).to_dict()
                    }
                else:
                    # 数值变量
                    cluster_characteristics[f'cluster_{cluster_id}']['features'][col] = {
                        'type': 'numeric',
                        'mean': cluster_data[col].mean(),
                        'median': cluster_data[col].median(),
                        'std': cluster_data[col].std()
                    }

        return cluster_characteristics


