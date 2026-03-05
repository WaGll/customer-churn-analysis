# 变量定义
PYTHON := python3
PIP := $(PYTHON) -m pip
PROJECT_NAME := customer-churn-analysis

# 默认目标
.PHONY: help
help:
	@echo "🔍 可用的命令："
	@echo "  install      - 安装项目必要依赖"
	@echo "  install-dev  - 安装开发和测试依赖"
	@echo "  run          - 运行客户流失全流程分析"
	@echo "  test         - 运行单元测试并生成覆盖率报告"
	@echo "  quality      - 运行 lint (检查) 和 format (格式化)"
	@echo "  notebook     - 启动 Jupyter Notebook 交互式环境"
	@echo "  clean        - 清理临时文件和缓存"
	@echo "  deploy       - 使用 Docker Compose 部署服务"

# 1. 安装依赖
.PHONY: install
install:
	@echo "🚀 正在安装核心依赖..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

.PHONY: install-dev
install-dev: install
	@echo "🛠️ 正在安装开发工具 (black, isort, pytest)..."
	$(PIP) install black isort pytest pytest-cov

# 2. 运行分析
.PHONY: run
run:
	@echo "📈 正在启动流失分析主程序..."
	@mkdir -p output/models output/visualizations output/results
	$(PYTHON) run_analysis.py

# 3. 质量与格式化
.PHONY: format
format:
	@echo "✨ 正在自动格式化代码..."
	black src/ tests/
	isort src/ tests/

.PHONY: lint
lint:
	@echo "🔎 正在执行静态代码检查..."
	black --check src/ tests/
	isort --check-only src/ tests/

.PHONY: quality
quality: format lint
	@echo "✅ 代码质量检查完成！"

# 4. 测试
.PHONY: test
test:
	@echo "🧪 正在运行测试套件..."
	pytest tests/ --cov=src --cov-report=term-missing

# 5. 清理 (增加了 -f 和目录保护)
.PHONY: clean
clean:
	@echo "🧹 正在清理缓存和临时文件..."
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache
	# 清理输出但保留文件夹结构
	rm -f output/models/* output/visualizations/* output/results/*
	@echo "✨ 清理完成"

# 6. 环境工具
.PHONY: notebook
notebook:
	@echo "📓 启动 Jupyter Notebook..."
	jupyter notebook

.PHONY: deploy
deploy:
	@echo "🐳 正在通过 Docker 部署..."
	docker-compose up -d --build