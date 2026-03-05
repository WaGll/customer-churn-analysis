FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（尽量精简）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    fonts-wqy-microhei \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 用宿主机 UID/GID 创建用户
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appgroup && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash appuser && \
    chown -R appuser:appgroup /app

USER appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge  # 清理 pip 缓存，进一步减小层

COPY --chown=appuser:appgroup . .

# 创建输出目录
RUN mkdir -p /app/output/visualizations /app/logs /app/data/processed

CMD ["python", "run_analysis.py"]