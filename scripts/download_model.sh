#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${INSTALL_ROOT:-}" && -f "${PROJECT_DIR}/.install_root" ]]; then
    INSTALL_ROOT="$(<"${PROJECT_DIR}/.install_root")"
fi
INSTALL_ROOT="${INSTALL_ROOT:-${PROJECT_DIR}/runtime}"
[[ -f "${INSTALL_ROOT}/env.sh" ]] && source "${INSTALL_ROOT}/env.sh"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-4B-Instruct}"
MODEL_DIR="${MODEL_DIR:-${QWEN3VL_MODEL_DIR:-${INSTALL_ROOT}/models/Qwen3-VL-4B-Instruct}}"
MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"
mkdir -p "${MODEL_DIR}"
export MODEL_ID MODEL_DIR MODEL_SOURCE

echo ">>> 使用 ${MODEL_SOURCE} 下载 ${MODEL_ID} 到 ${MODEL_DIR}"
python - <<'PY'
import os
from pathlib import Path

model_id = os.environ["MODEL_ID"]
model_dir = os.environ["MODEL_DIR"]
source = os.environ["MODEL_SOURCE"].lower()

if source == "modelscope":
    from modelscope import snapshot_download
    snapshot_download(model_id, local_dir=model_dir)
elif source == "huggingface":
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=model_id, local_dir=model_dir)
else:
    raise SystemExit("MODEL_SOURCE 只能是 modelscope 或 huggingface")

from transformers import AutoConfig, AutoProcessor
config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
if config.model_type != "qwen3_vl":
    raise RuntimeError(f"模型类型错误，期望 qwen3_vl，实际为 {config.model_type}")
if not any(Path(model_dir).glob("*.safetensors")):
    raise RuntimeError("模型权重不完整：未找到 safetensors 文件")
print("model_type:", config.model_type)
print("✅ 模型文件和处理器校验通过:", model_dir)
PY
