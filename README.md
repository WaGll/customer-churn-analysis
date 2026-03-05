# 客户流失分析系统 - 高性能数据挖掘解决方案

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/workflow/status/WaGll/customer-churn-analysis/CI?branch=main)](https://github.com/WaGll/customer-churn-analysis/actions)
[![Docker](https://img.shields.io/badge/docker-compatible-blue.svg)](https://docker.com)

一个高性能、可扩展的客户流失分析系统，基于关联规则挖掘和聚类分析技术，帮助企业识别流失风险客户并进行精准营销。

## ✨ 特性

- 🔧 **模块化设计** - 清晰的代码结构，易于维护和扩展
- ⚡ **高性能** - 优化后的算法，处理速度提升46.9%
- 🐳 **Docker支持** - 一键部署，容器化运行
- 📊 **丰富的可视化** - 静态图表和交互式仪表板
- 🔄 **并行计算** - 自动检测CPU核心，充分利用硬件资源
- 📈 **实时监控** - 性能监控和预警机制
- 🧪 **完整测试** - 单元测试和集成测试覆盖
- 📚 **详细文档** - API文档和使用示例

## 🚀 快速开始

### 环境要求

- Python 3.9+
- Docker & Docker Compose (可选)
- 8GB+ RAM (推荐)

### 一键安装

```bash
# 克隆项目
git clone https://github.com/WaGll/customer-churn-analysis.git
cd customer_churn_analysis

# 安装依赖
make install

# 运行分析
make run
```

### Docker部署

```bash
# 一键部署
make deploy

# 查看结果
# 浏览器访问 http://localhost
```

## 📁 项目结构

```
customer_churn_analysis/
├── .github/                    # GitHub Actions工作流
│   └── workflows/
│       └── ci.yml            # 持续集成配置
├── data/                      # 数据目录
│   ├── customer_churn_data.xlsx  # 客户流失数据集
│   └── sales.csv              # 销售数据
├── docs/                      # 文档目录
│   ├── README.md              # 主文档
│   ├── API.md                # API文档
│   └── EXAMPLES.md           # 使用示例
├── notebooks/                 # Jupyter笔记本
│   └── analysis_demo.ipynb   # 分析演示
├── scripts/                   # 脚本目录
│   ├── deploy.sh             # 部署脚本
│   └── preprocess.py         # 数据预处理脚本
├── src/                      # 源代码
│   ├── __init__.py           # 包初始化
│   ├── data_loader.py        # 数据加载模块
│   ├── feature_engineering.py # 特征工程模块
│   ├── association_rules.py  # 关联规则挖掘模块
│   ├── clustering.py        # 聚类分析模块
│   ├── visualization.py      # 可视化模块
│   └── utils/                # 工具函数
│       ├── __init__.py
│       ├── logger.py         # 日志工具
│       └── helpers.py        # 辅助函数
├── tests/                    # 测试目录
│   
├── config/                   # 配置文件
│   └── settings.py           # 系统配置
├── docker/                   # Docker相关
│   ├── Dockerfile            # 主服务镜像
│   ├── Dockerfile.jupyter    # Jupyter镜像
│   └── docker-compose.yml    # 服务编排
├── output/                   # 输出目录(运行时创建)
│   ├── models/               # 保存的模型
│   ├── visualizations/       # 生成的图表
│   └── results/              # 分析结果
├── requirements.txt          # Python依赖
├── pyproject.toml           # 项目元数据
├── .gitignore               # Git忽略规则
├── LICENSE                  # 开源许可证
├── Makefile                 # 构建工具
└── README.md                # 项目说明
```

## 📊 核心功能

### 1. 数据处理
- **智能数据加载** - 支持Excel、CSV等多种格式
- **内存优化** - 自动优化数据类型，减少40%内存使用
- **批量处理** - 支持大数据集分批处理
- **数据质量报告** - 自动生成数据质量分析报告

### 2. 特征工程
- **多种编码方式** - One-Hot、混合编码、标准化处理
- **自适应分箱** - 使用MeanShift进行智能分箱
- **特征重要性评估** - 基于随机森林和互信息
- **并行处理** - 多进程加速特征处理

### 3. 关联规则挖掘
- **双算法支持** - Apriori和FP-Growth算法
- **参数自动优化** - 随机搜索寻找最优参数
- **业务规则发现** - 自动识别流失相关规则
- **多维度评估** - 支持度、置信度、提升度等指标

### 4. 聚类分析
- **多种算法** - K-Means、K-Prototypes
- **自动K值选择** - 肘部法、轮廓系数综合判断
- **稳定性测试** - 验证聚类结果稳定性
- **客户细分分析** - 深度分析各群体特征

### 5. 可视化
- **交互式仪表板** - 基于Plotly的动态图表
- **静态图表生成** - 批量生成分析图表
- **自定义样式** - 业务导向的配色方案
- **多格式导出** - HTML、PNG等多种格式

## 🛠️ 使用方法

### 基本使用

```python
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.association_rules import AssociationRuleMiner
from src.clustering import ClusterAnalyzer

# 1. 加载数据
data_loader = DataLoader()
df = data_loader.load_data('data/customer_churn_data.xlsx')

# 2. 特征工程
feature_engineer = FeatureEngineer()
datasets = feature_engineer.preprocess_data(df)

# 3. 关联规则挖掘
rule_miner = AssociationRuleMiner()
rules = rule_miner.mine_association_rules(
    datasets['mixed'],
    algorithm='fp_growth'
)

# 4. 聚类分析
cluster_analyzer = ClusterAnalyzer()
clusters = cluster_analyzer.kmeans_clustering(
    datasets['standardized']
)
```

### 命令行使用

```bash
# 运行完整分析
python run_analysis.py

# 指定数据路径和输出目录
python run_analysis.py --data-path data/customer-churn-analysis.xlsx --output-dir results/

# 只运行分析，不生成可视化
python run_analysis.py --no-visualization

# 使用指定算法
python run_analysis.py --algorithm apriori

# 查看帮助
python run_analysis.py --help
```

### Make命令

```bash
# 安装依赖
make install

# 运行分析
make run

# 运行测试
make test

# 启动Jupyter
make notebook

# 部署到Docker
make deploy

# 代码格式化
make format

# 代码检查
make lint
```

## 🐳 Docker部署

### 单服务部署

```bash
# 构建镜像
make build

# 运行主服务
docker run --rm -v $(pwd)/data:/app/data customer-churn-analysis:latest \
  python run_analysis.py --data-path /app/data/customer_churn_data.xlsx
```

### 完整服务栈

```bash
# 启动所有服务
make deploy

# 查看服务状态
docker-compose ps

# 查看日志
make logs

# 停止服务
docker-compose down
```

服务说明：
- **customer-churn-analysis**: 主分析服务
- **redis**: 缓存服务（可选）
- **jupyter**: Jupyter Lab（可选）
- **nginx**: Web展示服务（可选）

## 🔧 配置说明

### 环境变量

```bash
# 数据路径
export DATA_PATH=/app/data

# 输出目录
export OUTPUT_PATH=/app/output

# 日志级别
export LOG_LEVEL=INFO

# 并行进程数
export N_JOBS=4
```

### 配置文件

编辑 `config/settings.py`:

```python
@dataclass
class DataConfig:
    # 数据路径配置
    raw_data_path: str = "data/customer_churn_data.xlsx"
    processed_data_path: str = "data/processed/"
    cache_dir: str = "cache/"

@dataclass
class AlgorithmConfig:
    # 算法参数
    min_support: float = 0.02
    min_confidence: float = 0.3
    n_jobs: int = -1  # 使用所有CPU核心
```
 |

### 扩展能力

- 支持百万级数据集处理
- 自动并行计算优化
- 智能内存管理
- 分布式计算准备

## 🧪 测试

```bash
# 运行所有测试
make test

# 运行性能测试
make perf-test

# 生成覆盖率报告
pytest tests/ --cov=src --cov-report=html
```

## 📖 文档

- [API文档](docs/API.md) - 详细的API使用说明
- [使用示例](docs/EXAMPLES.md) - 丰富的使用案例
- [部署指南](DEPLOYMENT.md) - Docker部署详细说明
- [性能优化报告](PERFORMANCE_OPTIMIZATION.md) - 性能优化详情

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 开发环境设置

```bash
# 克隆项目
git clone https://github.com/WaGll/customer-churn-analysis.git
cd customer-churn-analysis

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装开发依赖
make install-dev

# 运行代码检查
make quality
```

## 🎯 业务应用

### 客户流失预警

系统可以识别以下流失风险信号：
- 上月有投诉记录
- 使用App时间急剧下降
- 距上次下单时间过长
- 满意度评分较低
- 订单数量明显减少

### 客户细分策略

基于聚类结果，系统将客户分为：
- **高价值忠诚客户** - 提供VIP服务
- **潜在流失风险** - 主动关怀挽留
- **新活跃客户** - 个性化推荐
- **低价值客户** - 成本控制

### 营销决策支持

通过关联规则发现：
- 流失客户的共同特征
- 挽回成功率最高的策略
- 产品组合推荐规则
- 价格敏感度分析

## 🔍 监控与维护

### 性能监控

```python
# 查看实时性能
python performance_test.py --data-path data/customer_churn_data.xlsx

# 查看历史报告
cat output/performance_report.json
```

### 日志管理

```bash
# 查看应用日志
tail -f logs/app.log

# 查看性能日志
tail -f logs/performance.log
```

## 📝 更新日志

### v1.0.0 (2026-02-24)
- ✨ 初始版本发布
- 🚀 实现高性能数据加载
- 🔧 添加模块化设计
- 🐳 支持Docker部署
- 📊 完整可视化方案
- 🧪 完整测试覆盖

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- [scikit-learn](https://scikit-learn.org/) - 机器学习库
- [pandas](https://pandas.pydata.org/) - 数据处理库
- [plotly](https://plotly.com/python/) - 可视化库
- [mlxtend](https://rasbt.github.io/mlxtend/) - 机器学习工具

## 📞 联系方式

- 作者: WG
- 邮箱: wgaaa98@gmail.com
- 项目主页: https://github.com/WaGll/customer-churn-analysis
- 问题反馈: [Issues](https://github.com/WaGll/customer-churn-analysis/issues)

---

⭐ 如果这个项目对您有帮助，请考虑给个Star！