场景 1：三步实现快速分析 (Quick Start)
如果你只想快速得到一个分析报告，只需运行以下核心逻辑。

Python
from src.data_loader import load_data
from src.feature_engineering import preprocess_data
from src.visualization import create_interactive_dashboard

# 1. 加载数据
df = load_data("data/customer_churn_data.xlsx")

# 2. 自动化特征工程（获取标准化后的数据）
processed_data = preprocess_data(df)
X = processed_data['standardized']

# 3. 直接生成可视化报告
# 假设已经有了初步的规则和标签（可由 main 函数自动生成）
create_interactive_dashboard(df, rules=[], clusters=[])
print("分析完成！请在 output/visualizations 查看报告。")
场景 2：深度流失诱因挖掘 (Root Cause Analysis)
这个场景专注于找出“为什么客户会走”。我们通过关联规则挖掘（Association Rules）来寻找高提升度（Lift）的特征组合。

Python
from src.data_loader import load_data
from src.association_rules import mine_association_rules

# 加载并选择流失相关的列
df = load_data("data/customer_churn_data.xlsx")

# 挖掘关联规则：设定最小支持度 0.05，置信度 0.6
rules = mine_association_rules(df, min_support=0.05, min_confidence=0.6)

# 过滤出结论为“流失”的规则，并按提升度排序
churn_rules = rules[rules['consequents'].apply(lambda x: '用户流失标签' in str(x))]
top_reasons = churn_rules.sort_values(by='lift', ascending=False).head(5)

print("导致流失的前 5 大特征组合：")
print(top_reasons[['antecedents', 'confidence', 'lift']])
场景 3：客户分群与精准营销 (Customer Segmentation)
利用聚类分析将客户分为不同的群体，以便运营团队进行差异化补救。

Python
import pandas as pd
from src.data_loader import load_data
from src.feature_engineering import preprocess_data
from src.clustering import kmeans_clustering

# 1. 数据预处理
df = load_data("data/customer_churn_data.xlsx")
data_dict = preprocess_data(df)

# 2. 执行聚类（系统会自动选择最优 K 值）
labels, kmeans_model, best_k = kmeans_clustering(data_dict['standardized'])

# 3. 将聚类结果合并回原始数据进行画像分析
df['cluster_id'] = labels
cluster_profile = df.groupby('cluster_id').agg({
    '上月投诉次数': 'mean',
    '使用App时间_时': 'mean',
    '用户流失标签': 'mean' # 计算该群体的流失率
}).reset_index()

print(f"系统发现的最优分群数: {best_k}")
print(cluster_profile)
场景 4：通过 Makefile 运行生产流水线
如果你不想写 Python 代码，直接使用我们配置好的 Makefile 工具：

Bash
# 安装环境
make install-dev

# 一键运行完整分析流程（加载 -> 特征工程 -> 关联规则 -> 聚类 -> 可视化）
make run

# 如果模型运行不稳定或产生太多临时文件，执行清理
make clean

# 部署到 Docker 环境查看 Web 仪表板
make deploy
💡 开发者贴士
性能监控：所有示例在运行时都会触发 src.utils.performance 模块，你可以在控制台实时看到每个步骤消耗了多少内存。

数据路径：示例默认数据存放于 data/ 目录。若使用自定义数据，请确保列名与系统要求的特征一致。