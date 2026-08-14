#!/usr/bin/env bash
set -Eeuo pipefail

# One-time installer for an NVIDIA Linux cloud instance.
# Optional overrides:
#   INSTALL_ROOT=/data/qwen3vl MODEL_SOURCE=modelscope bash install.sh
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 bash install.sh
#   DOWNLOAD_MODEL=0 bash install.sh
#   TRANSFORMERS_VERSION=5.6.0 bash install.sh

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-${PROJECT_DIR}/runtime}"
VENV_DIR="${INSTALL_ROOT}/venv"
REPO_DIR="${INSTALL_ROOT}/LlamaFactory"
MODEL_DIR="${MODEL_DIR:-${INSTALL_ROOT}/models/Qwen3-VL-4B-Instruct}"
HF_HOME="${HF_HOME:-${INSTALL_ROOT}/cache/huggingface}"
MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${INSTALL_ROOT}/cache/modelscope}"
LLAMAFACTORY_REF="${LLAMAFACTORY_REF:-v0.9.5}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.6.0}"
NUMPY_VERSION="${NUMPY_VERSION:-1.26.4}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
DOWNLOAD_MODEL="${DOWNLOAD_MODEL:-1}"
MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"

log() { printf '\n>>> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || die "仅支持 Linux 云服务器。"
[[ "$(uname -m)" == "x86_64" ]] || die "当前安装包仅验证了 Linux x86_64。"
command -v git >/dev/null || die "缺少 git，请先用系统包管理器安装。"
command -v curl >/dev/null || die "缺少 curl，请先用系统包管理器安装。"
command -v nvidia-smi >/dev/null || die "未检测到 NVIDIA 驱动。请选用带 NVIDIA GPU/驱动的云镜像。"

mkdir -p "${INSTALL_ROOT}" "${HF_HOME}" "${MODELSCOPE_CACHE}" "${PROJECT_DIR}/outputs" "${PROJECT_DIR}/logs"
printf '%s\n' "${INSTALL_ROOT}" > "${PROJECT_DIR}/.install_root"
export HF_HOME MODELSCOPE_CACHE HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

log "GPU 信息"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

log "准备 Python 3.11 虚拟环境"
if command -v uv >/dev/null; then
    UV_BIN="$(command -v uv)"
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
    UV_BIN="${HOME}/.local/bin/uv"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV_BIN="${HOME}/.local/bin/uv"
fi
[[ -x "${UV_BIN}" ]] || die "uv 安装失败。"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${UV_BIN}" python install 3.11
    "${UV_BIN}" venv --python 3.11 --seed "${VENV_DIR}"
fi
if grep -Eiq '^include-system-site-packages[[:space:]]*=[[:space:]]*true' "${VENV_DIR}/pyvenv.cfg"; then
    die "检测到旧版非隔离虚拟环境。请删除 runtime/ 后重装，或改用新的 INSTALL_ROOT。"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

log "检查 CUDA 版 PyTorch"
if ! python - <<'PY'
import sys
try:
    import torch
    ok = torch.cuda.is_available() and tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) >= (2, 4)
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
then
    echo "未找到可用的 CUDA PyTorch，安装默认 CUDA 12.4 构建。"
    echo "如服务器需要其他版本，请设置 TORCH_INDEX_URL 后重新运行。"
    python -m pip install --index-url "${TORCH_INDEX_URL}" torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
fi

log "安装固定版本 LlamaFactory ${LLAMAFACTORY_REF}"
if [[ ! -d "${REPO_DIR}/.git" ]]; then
    git clone --branch "${LLAMAFACTORY_REF}" --depth 1 https://github.com/hiyouga/LlamaFactory.git "${REPO_DIR}"
else
    current_ref="$(git -C "${REPO_DIR}" describe --tags --always 2>/dev/null || true)"
    [[ "${current_ref}" == "${LLAMAFACTORY_REF}" ]] || die "已有仓库版本为 ${current_ref}，期望 ${LLAMAFACTORY_REF}。请更换 INSTALL_ROOT。"
fi

log "安装锁定依赖（Transformers ${TRANSFORMERS_VERSION}, NumPy ${NUMPY_VERSION}）"
python -m pip install \
    -e "${REPO_DIR}" \
    "transformers==${TRANSFORMERS_VERSION}" \
    "numpy==${NUMPY_VERSION}" \
    qwen-vl-utils \
    "deepspeed==0.18.0"

cat > "${INSTALL_ROOT}/env.sh" <<EOF
export QWEN3VL_PROJECT_DIR="${PROJECT_DIR}"
export QWEN3VL_INSTALL_ROOT="${INSTALL_ROOT}"
export QWEN3VL_MODEL_DIR="${MODEL_DIR}"
export HF_HOME="${HF_HOME}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE}"
export HF_ENDPOINT="${HF_ENDPOINT}"
export LLAMAFACTORY_DIR="${REPO_DIR}"
source "${VENV_DIR}/bin/activate"
EOF

if [[ "${DOWNLOAD_MODEL}" == "1" ]]; then
    log "下载 Qwen3-VL-4B-Instruct"
    MODEL_DIR="${MODEL_DIR}" MODEL_SOURCE="${MODEL_SOURCE}" bash "${PROJECT_DIR}/download_model.sh"
else
    echo "已跳过模型下载；稍后运行 bash download_model.sh。"
fi

log "验证 Qwen3-VL 训练环境"
python - <<'PY'
import torch
import transformers
import llamafactory
import numpy
import deepspeed
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: F401

assert torch.cuda.is_available(), "torch 无法使用 CUDA"
major, minor = torch.cuda.get_device_capability()
print("llamafactory:", getattr(llamafactory, "__version__", "installed"))
print("transformers:", transformers.__version__)
print("numpy:", numpy.__version__)
print("deepspeed:", deepspeed.__version__)
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0), "capability:", f"{major}.{minor}")
if not torch.cuda.is_bf16_supported():
    print("WARNING: GPU 不支持 bf16；train.sh 将自动切换为 fp16。")
print("✅ Qwen3-VL LoRA 训练环境安装完成")
PY

printf '\n下一步：\n  1. 将完整图片目录放入 data/wsi_train/（训练/测试 JSON 已就位）\n  2. 运行 bash "%s/train.sh"\n' "${PROJECT_DIR}"
