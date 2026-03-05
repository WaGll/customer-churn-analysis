开发者详细文档 (Technical Documentation)
欢迎阅读 客户流失分析系统 的深度文档。本目录包含了系统的核心设计逻辑、算法选择依据以及扩展指南。

📖 文档索引
API 参考手册 (API.md) - 详细的函数接口说明。

场景应用示例 (EXAMPLES.md) - 针对业务场景的代码片段。

原始数据说明 (如果存在) - 字段含义与数据字典。

🏗️ 核心架构说明
本系统采用模块化设计，确保各分析组件（关联规则、聚类、特征工程）可以独立运行或组合调用。

数据流向
Raw Data: 从 data/ 加载 Excel/CSV。

Preprocessing: 执行类型转换、内存优化及多种编码（One-Hot, Mixed, Standardized）。

Analysis Engines:

Association Rules: 挖掘高频流失路径。

Clustering: 划分客户群体画像。

Output: 结果持久化为 JSON/CSV，并生成交互式 HTML 报告。

🔬 算法逻辑深度解析
1. 关联规则挖掘 (FP-Growth)
系统放弃了传统的 Apriori 算法，采用了更高效的 FP-Growth。

目的: 发现类似 [频繁投诉] + [长久未登录] -> [极高流失风险] 的模式。

关键指标: 我们重点关注 Lift (提升度)。如果 Lift > 1，说明前件对后件（流失）有显著的正向触发作用。

2. 聚类分析 (K-Means & K-Prototypes)
K-Means: 用于标准化后的数值特征。

K-Prototypes: 用于处理同时包含分类（如“性别”）和数值（如“余额”）的原始数据。

稳定性测试: 系统会自动运行多次聚类并计算平均轮廓系数，以评估分群的可信度。

🛠️ 二次开发指南
如何添加新的特征处理逻辑？
在 src/feature_engineering.py 中新增一个处理函数。

在 preprocess_data 主函数中调用该方法。

确保输出格式依然保持字典结构以兼容后续模块。

如何调整可视化模板？
所有的 HTML 生成逻辑位于 src/visualization.py。

本系统使用 Plotly 进行交互渲染。

若需修改样式，请调整 export_all_plots_to_html 函数中的布局参数。

🧪 测试说明
我们使用 pytest 进行覆盖测试，并使用自定义的 performance.py 工具监控压力。

单元测试: 验证数据加载与编码的正确性。

性能测试: 模拟 10w+ 数据量下的内存峰值与耗时。

运行测试命令：

Bash
make test          # 基础测试
make perf-test     # 性能压测
🔗 相关资源
项目仓库: [GitHub Repo Link]

问题反馈: [Issue Tracker Link]

作者: 2026 Customer Churn Analysis Team