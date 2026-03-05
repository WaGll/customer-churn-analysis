#!/bin/bash

# 客户流失分析系统部署脚本

set -e

echo "=================================================="
echo "  客户流失分析系统部署"
echo "=================================================="

# 检查Docker和Docker Compose是否安装
check_docker() {
    echo "检查Docker环境..."

    if ! command -v docker &> /dev/null; then
        echo "错误: Docker未安装"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        echo "错误: Docker Compose未安装"
        exit 1
    fi

    echo "Docker环境检查通过"
}

# 构建Docker镜像
build_images() {
    echo "构建Docker镜像..."

    # 构建主分析服务镜像
    docker-compose build customer-churn-analysis

    # 构建Jupyter镜像
    docker-compose build jupyter

    echo "Docker镜像构建完成"
}

# 创建必要的目录
create_directories() {
    echo "创建必要的目录..."

    mkdir -p data/processed
    mkdir -p cache
    mkdir -p logs
    mkdir -p output
    mkdir -p visualizations
    mkdir -p notebooks

    echo "目录创建完成"
}

# 设置权限
set_permissions() {
    echo "设置文件权限..."

    chmod +x run_analysis.py
    chmod +x deploy.sh

    echo "权限设置完成"
}

# 启动服务
start_services() {
    echo "启动服务..."

    # 仅启动分析服务
    docker-compose up -d customer-churn-analysis

    # 可选：启动所有服务
    # docker-compose up -d

    echo "服务启动完成"
}

# 检查服务状态
check_status() {
    echo "检查服务状态..."

    docker-compose ps

    echo "\n访问以下地址:"
    echo "- 分析日志: docker-compose logs -f customer-churn-analysis"
    echo "- Jupyter Lab: http://localhost:8888 (如果启动了jupyter服务)"
    echo "- 可视化报告: http://localhost (如果启动了nginx服务)"
}

# 停止服务
stop_services() {
    echo "停止所有服务..."

    docker-compose down

    echo "服务已停止"
}

# 清理环境
clean() {
    echo "清理环境..."

    docker-compose down -v --remove-orphans

    read -p "是否删除数据目录？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf data/processed cache logs output visualizations notebooks
        echo "数据目录已删除"
    fi

    echo "清理完成"
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [命令]"
    echo ""
    echo "可用命令:"
    echo "  install     - 安装并部署系统"
    echo "  start       - 启动服务"
    echo "  stop        - 停止服务"
    echo "  restart     - 重启服务"
    echo "  status      - 查看服务状态"
    echo "  logs        - 查看服务日志"
    echo "  clean       - 清理环境"
    echo "  help        - 显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 install   # 首次部署"
    echo "  $0 start     # 启动服务"
    echo "  $0 logs      # 查看日志"
}

# 主函数
main() {
    case "${1:-}" in
        install)
            check_docker
            create_directories
            set_permissions
            build_images
            start_services
            check_status
            ;;
        start)
            start_services
            check_status
            ;;
        stop)
            stop_services
            ;;
        restart)
            stop_services
            start_services
            check_status
            ;;
        status)
            check_status
            ;;
        logs)
            docker-compose logs -f customer-churn-analysis
            ;;
        clean)
            clean
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "错误: 未知命令 '$1'"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

main "$@"