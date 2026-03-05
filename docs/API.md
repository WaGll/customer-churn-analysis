客户流失分析系统 API 说明文档
本文件详细说明了 src/ 目录下各核心模块的功能接口，方便开发者调用或进行二次开发。

1. 数据加载模块 (src.data_loader)
负责从原始文件读取数据并进行初步的内存优化。

load_data(file_path)
功能: 加载 Excel 或 CSV 格式的原始数据集。

参数: file_path (str) - 数据文件的路径。

返回: pd.DataFrame - 加载后的原始数据。

optimize_memory(df)
功能: 通过转换数值类型（如 int64 转 int32）和对象类型（转为 category）减少内存占用。

参数: df (pd.DataFrame) - 需要优化的 DataFrame。

返回: pd.DataFrame - 优化后的数据。

2. 特征工程模块 (src.feature_engineering)
负责数据的清洗、编码及标准化处理。

preprocess_data(df)
功能: 自动识别数值和分类变量，执行缺失值处理。

返回: 包含三种编码方式的字典：one_hot, mixed, standardized。

3. 关联规则挖掘 (src.association_rules)
基于 FP-Growth 算法挖掘客户流失的潜在模式。

mine_association_rules(df, min_support=0.1, min_confidence=0.5)
功能: 生成频繁项集并推导关联规则。

参数:

min_support (float): 最小支持度阈值。

min_confidence (float): 最小置信度阈值。

返回: pd.DataFrame - 包含支持度、置信度和提升度（Lift）的规则列表。

4. 聚类分析模块 (src.clustering)
实现客户分群逻辑。

kmeans_clustering(data, n_clusters_range=(2, 6))
功能: 执行 K-Means 聚类，并通过轮廓系数（Silhouette Score）自动选择最优 K 值。

返回: tuple - (最优标签, 聚类模型, 最佳K值)。

kprototypes_clustering(df, categorical_indices)
功能: 处理混合类型数据的聚类。

参数: categorical_indices (list) - 分类特征在 DataFrame 中的索引位置。

5. 可视化模块 (src.visualization)
生成交互式报表和分析图表。

create_interactive_dashboard(df, rules, clusters)
功能: 整合分析结果，生成一个独立的 HTML 交互式仪表板。

保存路径: output/visualizations/interactive_dashboard.html

create_cluster_profile_plots(df, cluster_labels)
功能: 绘制各聚类群体的特征分布对比图（雷达图、箱线图）。

6. 工具类 (src.utils)
提供系统支持功能。

performance.py: 提供 @timer 装饰器和内存监控工具，用于记录各模块的耗时与内存变化。

logger.py: 统一配置日志输出格式，将日志同时记录到控制台及 logs/ 文件夹。

调用示例
Python
from src.data_loader import load_data
from src.feature_engineering import preprocess_data
from src.clustering import kmeans_clustering

# 1. 加载
df = load_data("data/customer_churn_data.xlsx")

# 2. 处理
data_dict = preprocess_data(df)

# 3. 聚类
labels, model, best_k = kmeans_clustering(data_dict['standardized'])
