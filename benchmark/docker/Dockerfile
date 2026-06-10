# fwi-benchmark:cu118 - 项目推荐 Benchmark 容器（PyTorch 2.3.1+cu118）
# 构建: docker build -t fwi-benchmark:cu118 -f benchmark/docker/Dockerfile .
# 详见: benchmark/docker/BUILD_AND_RUN.md
FROM docker.1ms.run/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git curl wget ca-certificates \
    build-essential pkg-config \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

# PyTorch pinned to CUDA 11.8 build for T4
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
      torch==2.3.1+cu118 \
      torchvision==0.18.1+cu118 \
      torchaudio==2.3.1+cu118 \
      --index-url https://download.pytorch.org/whl/cu118

COPY benchmark/docker/requirements.txt /tmp/requirements.txt
# 使用清华镜像加速国内构建（PyTorch 仍从官方 index 安装）
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install -i ${PIP_INDEX} -r /tmp/requirements.txt

# LPIPS warmup 已移除：构建时下载 AlexNet 权重易因网络失败；首次评测时由 lpips 库按需下载（--lpips-mode real）

WORKDIR /workspace/repo
