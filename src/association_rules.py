"""
关联规则挖掘模块 - 优化的关联规则算法实现
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from config.settings import config
import psutil
from utils.performance import monitor_performance

logger = logging.getLogger(__name__)

class AssociationRuleMiner:
    """关联规则挖掘器 - 支持并行计算和参数优化"""

    def __init__(self):
        self.optimal_params = None
        self.best_rules = None
        self.frequent_itemsets_cache = {}

    @monitor_performance
    def mine_association_rules(self,
                             df: pd.DataFrame,
                             algorithm: str = 'fp_growth',
                             min_support: float = None,
                             min_confidence: float = None) -> Dict[str, Any]:
        """
        挖掘关联规则

        Args:
            df: 事务数据（每行是一个事务）
            algorithm: 使用的算法 ('apriori' 或 'fp_growth')
            min_support: 最小支持度
            min_confidence: 最小置信度

        Returns:
            Dict: 包含频繁项集和关联规则的字典
        """
        logger.info(f"开始使用 {algorithm} 算法挖掘关联规则...")

        if min_support is None:
            min_support = config.algorithm.min_support
        if min_confidence is None:
            min_confidence = config.algorithm.min_confidence

        # 1. 生成频繁项集
        frequent_itemsets = self._get_frequent_itemsets(df, algorithm, min_support)

        # 2. 生成关联规则
        rules = self._generate_rules(frequent_itemsets, min_confidence)

        # 3. 评估规则质量
        evaluated_rules = self._evaluate_rules(rules)

        result = {
            'algorithm': algorithm,
            'min_support': min_support,
            'min_confidence': min_confidence,
            'frequent_itemsets': frequent_itemsets,
            'rules': evaluated_rules,
            'rule_count': len(evaluated_rules),
            'itemset_count': len(frequent_itemsets)
        }

        logger.info(f"关联规则挖掘完成 - 生成 {len(evaluated_rules)} 条规则")

        return result

    def _get_frequent_itemsets(self, df: pd.DataFrame, algorithm: str, min_support: float) -> pd.DataFrame:
        """
        获取频繁项集（带缓存）

        Args:
            df: 事务数据
            algorithm: 算法名称
            min_support: 最小支持度

        Returns:
            DataFrame: 频繁项集
        """
        cache_key = f"{algorithm}_{min_support}_{hash(tuple(df.columns))}"

        if cache_key in self.frequent_itemsets_cache:
            logger.info("使用缓存的频繁项集")
            return self.frequent_itemsets_cache[cache_key]

        logger.info(f"开始计算频繁项集...")

        if algorithm == 'apriori':
            frequent_itemsets = apriori(
                df,
                min_support=min_support,
                use_colnames=True,
                verbose=0,  # 关闭 verbose 输出
                n_jobs=config.algorithm.n_jobs
            )
        elif algorithm == 'fp_growth':
            # FP-Growth不支持n_jobs参数
            frequent_itemsets = fpgrowth(
                df,
                min_support=min_support,
                use_colnames=True,
                verbose=0  # 关闭 verbose 输出
            )
        else:
            raise ValueError(f"不支持的算法: {algorithm}")

        # 缓存结果
        self.frequent_itemsets_cache[cache_key] = frequent_itemsets

        return frequent_itemsets

    def _generate_rules(self, frequent_itemsets: pd.DataFrame, min_confidence: float) -> pd.DataFrame:
        """
        生成关联规则

        Args:
            frequent_itemsets: 频繁项集
            min_confidence: 最小置信度

        Returns:
            DataFrame: 关联规则
        """
        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence,
            support_only=False
        )

        return rules

    def _evaluate_rules(self, rules: pd.DataFrame) -> pd.DataFrame:
        """
        评估规则质量

        Args:
            rules: 原始规则

        Returns:
            DataFrame: 带有评估指标的规则
        """
        if rules.empty:
            return rules

        # 计算额外的评估指标
        rules['lift'] = rules['lift']
        rules['leverage'] = rules['leverage']
        rules['conviction'] = rules['conviction']

        # 计算Kulczynski度量
        rules['kulczynski'] = (rules['support'] * (1 + rules['lift'])) / 2

        # 按提升度排序
        rules = rules.sort_values('lift', ascending=False)

        return rules

    @monitor_performance
    def optimize_parameters(self,
                          df: pd.DataFrame,
                          algorithm: str = 'fp_growth',
                          param_space: Dict[str, List] = None,
                          progress_callback=None) -> Dict[str, Any]:
        """
        优化算法参数

        Args:
            df: 事务数据
            algorithm: 算法名称
            param_space: 参数搜索空间
            progress_callback: 进度回调函数

        Returns:
            Dict: 最优参数和结果
        """
        if param_space is None:
            param_space = {
                'min_support': [0.01, 0.02, 0.03, 0.05],
                'min_confidence': [0.3, 0.4, 0.5, 0.6]
            }

        logger.info("开始参数优化...")

        best_score = 0
        best_params = None
        best_result = None

        # 使用随机搜索
        n_samples = min(20, len(param_space['min_support']) * len(param_space['min_confidence']))

        for i in range(n_samples):
            # 随机选择参数
            min_support = np.random.choice(param_space['min_support'])
            min_confidence = np.random.choice(param_space['min_confidence'])

            try:
                # 运行算法
                result = self.mine_association_rules(
                    df,
                    algorithm=algorithm,
                    min_support=min_support,
                    min_confidence=min_confidence
                )

                # 计算评分（综合考虑规则的提升度和数量）
                if len(result['rules']) > 0:
                    avg_lift = result['rules']['lift'].mean()
                    rule_count = len(result['rules'])
                    score = (avg_lift + np.log(rule_count + 1)) * 0.5
                else:
                    score = 0  # 没有规则得0分

                if score > best_score:
                    best_score = score
                    best_params = {
                        'min_support': min_support,
                        'min_confidence': min_confidence
                    }
                    best_result = result

            except Exception as e:
                logger.warning(f"参数组合失败: min_support={min_support}, min_confidence={min_confidence}")
                continue

            # 更新进度
            if progress_callback is not None:
                progress_callback()

        if best_params is None:
            raise ValueError("未能找到有效的参数组合")

        logger.info(f"参数优化完成 - 最优参数: {best_params}, 评分: {best_score:.4f}")

        self.optimal_params = best_params

        return {
            'optimal_params': best_params,
            'best_result': best_result,
            'best_score': best_score
        }

    def parallel_mining(self,
                       df_list: List[pd.DataFrame],
                       algorithm: str = 'fp_growth',
                       min_support: float = None,
                       min_confidence: float = None) -> List[Dict]:
        """
        并行挖掘多个数据集的关联规则

        Args:
            df_list: 数据集列表
            algorithm: 算法名称
            min_support: 最小支持度
            min_confidence: 最小置信度

        Returns:
            List[Dict]: 各数据集的挖掘结果
        """
        logger.info("开始并行关联规则挖掘...")

        results = []
        with ProcessPoolExecutor(max_workers=config.algorithm.n_jobs) as executor:
            # 提交任务
            futures = [
                executor.submit(
                    self.mine_association_rules,
                    df,
                    algorithm,
                    min_support,
                    min_confidence
                ) for df in df_list
            ]

            # 收集结果
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"并行挖掘任务失败: {str(e)}")
                    continue

        logger.info(f"并行挖掘完成 - 完成 {len(results)} 个数据集")

        return results

    def find_loss_related_rules(self, rules_df: pd.DataFrame) -> List[Dict]:
        """
        找到与客户流失相关的规则

        Args:
            rules_df: 关联规则DataFrame

        Returns:
            List[Dict]: 相关规则列表
        """
        if rules_df.empty:
            return []

        # 定义与流失相关的关键词
        loss_keywords = ['流失', '投诉', '不满', '差评', '取消', '退订', '低活跃']

        related_rules = []

        for _, rule in rules_df.iterrows():
            # 检查前项和后项
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])

            # 检查是否包含流失相关项
            loss_related = False
            for item in antecedents + consequents:
                if any(keyword in str(item) for keyword in loss_keywords):
                    loss_related = True
                    break

            if loss_related and rule['lift'] > 1.0:  # 只保留提升度大于1的规则
                related_rules.append({
                    'rule': f"{antecedents} -> {consequents}",
                    'support': rule['support'],
                    'confidence': rule['confidence'],
                    'lift': rule['lift'],
                    'kulczynski': rule.get('kulczynski', 0)
                })

        # 按提升度排序
        related_rules.sort(key=lambda x: x['lift'], reverse=True)

        return related_rules

    def rule_quality_report(self, rules_df: pd.DataFrame) -> Dict[str, Any]:
        """
        生成规则质量报告

        Args:
            rules_df: 关联规则DataFrame

        Returns:
            Dict: 质量报告
        """
        if rules_df.empty:
            return {'error': '没有规则可分析'}

        report = {
            '基本信息': {
                '规则总数': len(rules_df),
                '平均支持度': rules_df['support'].mean(),
                '平均置信度': rules_df['confidence'].mean(),
                '平均提升度': rules_df['lift'].mean()
            },
            '分布统计': {
                '支持度分布': rules_df['support'].describe().to_dict(),
                '置信度分布': rules_df['confidence'].describe().to_dict(),
                '提升度分布': rules_df['lift'].describe().to_dict()
            },
            '高质量规则': {
                '提升度>2': len(rules_df[rules_df['lift'] > 2]),
                '置信度>0.8': len(rules_df[rules_df['confidence'] > 0.8]),
                'Kulczynski>0.5': len(rules_df[rules_df.get('kulczynski', 0) > 0.5])
            },
            '前10强规则': rules_df.head(10)[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_dict('records')
        }

        return report


