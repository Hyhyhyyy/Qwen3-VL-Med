#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${INSTALL_ROOT:-}" && -f "${PROJECT_DIR}/.install_root" ]]; then
    INSTALL_ROOT="$(<"${PROJECT_DIR}/.install_root")"
fi
INSTALL_ROOT="${INSTALL_ROOT:-${PROJECT_DIR}/runtime}"
ENV_FILE="${INSTALL_ROOT}/env.sh"
[[ -f "${ENV_FILE}" ]] || { echo "未安装环境，请先运行 bash install.sh" >&2; exit 1; }
# shellcheck disable=SC1090
source "${ENV_FILE}"

TRAIN_MODE="${TRAIN_MODE:-lora}"
case "${TRAIN_MODE}" in
    lora)
        DEFAULT_CONFIG="${PROJECT_DIR}/configs/qwen3vl_lora_sft.yaml"
        DEFAULT_OUTPUT_DIR="${PROJECT_DIR}/outputs/qwen3vl-4b-lora-wsi"
        ;;
    full)
        DEFAULT_CONFIG="${PROJECT_DIR}/configs/qwen3vl_full_sft.yaml"
        DEFAULT_OUTPUT_DIR="${PROJECT_DIR}/outputs/qwen3vl-4b-full-wsi"
        export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
        ;;
    *)
        echo "TRAIN_MODE 只支持 lora 或 full，当前值：${TRAIN_MODE}" >&2
        exit 1
        ;;
esac
CONFIG="${CONFIG:-${DEFAULT_CONFIG}}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
MODEL_PATH="${MODEL_PATH:-${QWEN3VL_MODEL_DIR}}"

[[ -f "${CONFIG}" ]] || { echo "配置不存在：${CONFIG}" >&2; exit 1; }
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "模型不存在：${MODEL_PATH}；请先运行 download_model.sh" >&2; exit 1; }
if [[ "${TRAIN_MODE}" == "full" ]]; then
    python -c 'import deepspeed' >/dev/null 2>&1 || {
        echo "全量微调需要 DeepSpeed；请重新运行 bash install.sh，或在虚拟环境中安装 deepspeed==0.18.0" >&2
        exit 1
    }
    GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
    if (( GPU_COUNT < 2 )); then
        echo ">>> WARNING: 全量微调仅检测到 ${GPU_COUNT} 张 GPU；Qwen3-VL-4B 全参训练通常需要多张高显存 GPU。"
    fi
fi
python "${PROJECT_DIR}/scripts/validate_dataset.py" --dataset-dir "${DATASET_DIR}" --name wsi_train --check-images
python "${PROJECT_DIR}/scripts/validate_dataset.py" --dataset-dir "${DATASET_DIR}" --name wsi_test --check-images
mkdir -p "${OUTPUT_DIR}" "${PROJECT_DIR}/logs"

PRECISION_ARGS=(bf16=true fp16=false)
if ! python -c 'import torch; raise SystemExit(0 if torch.cuda.is_bf16_supported() else 1)'; then
    echo ">>> 当前 GPU 不支持 bf16，自动使用 fp16"
    PRECISION_ARGS=(bf16=false fp16=true)
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${LLAMAFACTORY_DIR}"
echo ">>> 配置: ${CONFIG}"
echo ">>> 模型: ${MODEL_PATH}"
echo ">>> 输出: ${OUTPUT_DIR}"
echo ">>> 训练模式: ${TRAIN_MODE}"

# LlamaFactory 会使用所有可见 GPU。可用 CUDA_VISIBLE_DEVICES=0 限定单卡。
exec llamafactory-cli train "${CONFIG}" \
    model_name_or_path="${MODEL_PATH}" \
    dataset_dir="${DATASET_DIR}" \
    output_dir="${OUTPUT_DIR}" \
    "${PRECISION_ARGS[@]}" \
    "$@"
