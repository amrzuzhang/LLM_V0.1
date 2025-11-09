#!/usr/bin/env bash
# 安装 LandGPT 项目的依赖环境
set -euo pipefail

ENV_NAME=${1:-landgpt}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] 未检测到 conda，请先安装 Miniconda 或 Anaconda。" >&2
  exit 1
fi

echo "[INFO] 创建/更新 conda 环境: ${ENV_NAME} (Python ${PYTHON_VERSION})"
conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}" >/dev/null

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

PIP=${PIP:-pip}

if [ ! -f "requirements.txt" ]; then
  echo "[ERROR] 未找到 requirements.txt，请确保在项目根目录运行此脚本。" >&2
  exit 1
fi

echo "[INFO] 安装 Python 依赖..."
${PIP} install --upgrade pip
${PIP} install -r requirements.txt

echo "[INFO] 环境 ${ENV_NAME} 已准备就绪。"
