#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${PROJECT_DIR}/.workflow"
mkdir -p "${STATE_DIR}" "${PROJECT_DIR}/logs" "${PROJECT_DIR}/outputs"

if [[ -f "${PROJECT_DIR}/configs/qwen3vl_lora_sft.yaml" ]]; then
    MODE="lora"
    TRAIN_OUTPUT="${TRAIN_OUTPUT:-${PROJECT_DIR}/outputs/qwen3vl-4b-lora-wsi}"
else
    MODE="full"
    TRAIN_OUTPUT="${TRAIN_OUTPUT:-${PROJECT_DIR}/outputs/qwen3vl-4b-full-wsi}"
fi

if [[ -z "${INSTALL_ROOT:-}" && -f "${PROJECT_DIR}/.install_root" ]]; then
    INSTALL_ROOT="$(<"${PROJECT_DIR}/.install_root")"
fi
INSTALL_ROOT="${INSTALL_ROOT:-${PROJECT_DIR}/runtime}"
ENV_FILE="${INSTALL_ROOT}/env.sh"
[[ -f "${ENV_FILE}" ]] || { echo "未安装环境，请先运行 bash install.sh" >&2; exit 1; }
# shellcheck disable=SC1090
source "${ENV_FILE}"

timestamp() { date -u +%Y%m%dT%H%M%SZ; }
record_state() { printf '%s\n' "$2" > "${STATE_DIR}/$1"; }
read_state() { [[ -f "${STATE_DIR}/$1" ]] && cat "${STATE_DIR}/$1"; }

precision_args() {
    if python -c 'import torch; raise SystemExit(0 if torch.cuda.is_bf16_supported() else 1)'; then
        printf '%s\n' "bf16=true" "fp16=false"
    else
        printf '%s\n' "bf16=false" "fp16=true"
    fi
}

ensure_trained_model() {
    [[ -d "${TRAIN_OUTPUT}" ]] || { echo "正式训练目录不存在：${TRAIN_OUTPUT}" >&2; exit 1; }
    [[ -f "${TRAIN_OUTPUT}/trainer_state.json" ]] || {
        echo "缺少 trainer_state.json，不能确认训练成功：${TRAIN_OUTPUT}" >&2
        exit 1
    }
    if [[ "${MODE}" == "lora" ]]; then
        [[ -f "${TRAIN_OUTPUT}/adapter_config.json" ]] || { echo "缺少 LoRA adapter_config.json" >&2; exit 1; }
        compgen -G "${TRAIN_OUTPUT}/adapter_model.*" >/dev/null || { echo "缺少 LoRA adapter 权重" >&2; exit 1; }
    else
        [[ -f "${TRAIN_OUTPUT}/config.json" ]] || { echo "缺少全量模型 config.json" >&2; exit 1; }
        compgen -G "${TRAIN_OUTPUT}/*.safetensors" >/dev/null || { echo "缺少全量模型 safetensors 权重" >&2; exit 1; }
    fi
}

run_smoke() {
    local output log
    output="${PROJECT_DIR}/outputs/smoke-$(timestamp)"
    log="${PROJECT_DIR}/logs/smoke-$(timestamp).log"
    echo ">>> 小样本训练冒烟测试：${output}"
    VALIDATE_MAX_RECORDS="${SMOKE_SAMPLES:-8}" TRAIN_LOG="${log}" OUTPUT_DIR="${output}" \
        bash "${PROJECT_DIR}/train.sh" \
        max_samples="${SMOKE_SAMPLES:-8}" max_steps=1 logging_steps=1 \
        save_strategy=steps save_steps=1 save_total_limit=1 overwrite_output_dir=true
    record_state smoke_output "${output}"
    echo ">>> 冒烟测试完成：${output}"
}

run_train() {
    local log
    log="${PROJECT_DIR}/logs/formal-$(timestamp).log"
    local gpu_count="${EXPECTED_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
    local model_path="${MODEL_PATH:-${QWEN3VL_MODEL_DIR}}"
    echo ">>> 正式训练：${TRAIN_OUTPUT}"
    set +e
    OUTPUT_DIR="${TRAIN_OUTPUT}" TRAIN_LOG="${log}" python "${PROJECT_DIR}/train_guard.py" run launch \
        --output-dir "${TRAIN_OUTPUT}" --log-file "${log}" \
        --framework llamafactory --training-type "${MODE}" \
        --expected-gpus "${gpu_count}" --model-path "${model_path}" --strict-preflight \
        -- bash "${PROJECT_DIR}/train.sh" overwrite_output_dir=true
    local code=$?
    set -e
    if (( code >= 2 )); then
        echo "正式训练或 Train Guard 验收失败，退出码：${code}" >&2
        exit "${code}"
    fi
    ensure_trained_model
    record_state train_output "${TRAIN_OUTPUT}"
    echo ">>> 正式训练完成并通过产物检查"
}

run_prediction() {
    local sample_limit="$1"
    local stage="$2"
    local output
    output="${PROJECT_DIR}/outputs/${stage}-$(timestamp)"
    local config="${PROJECT_DIR}/configs/qwen3vl_predict.yaml"
    local precision=()
    mapfile -t precision < <(precision_args)
    ensure_trained_model
    python "${PROJECT_DIR}/scripts/validate_dataset.py" --dataset-dir "${PROJECT_DIR}/data" \
        --name wsi_test --check-images --max-records "${sample_limit}" \
        --issues-jsonl "${PROJECT_DIR}/logs/${stage}-data-issues.jsonl"
    local model_args=(model_name_or_path="${TRAIN_OUTPUT}")
    if [[ "${MODE}" == "lora" ]]; then
        model_args=(model_name_or_path="${MODEL_PATH:-${QWEN3VL_MODEL_DIR}}" adapter_name_or_path="${TRAIN_OUTPUT}")
    fi
    mkdir -p "${output}"
    (
        cd "${LLAMAFACTORY_DIR}"
        llamafactory-cli train "${config}" \
            "${model_args[@]}" dataset_dir="${PROJECT_DIR}/data" output_dir="${output}" \
            max_samples="${sample_limit}" "${precision[@]}"
    ) 2>&1 | tee -a "${PROJECT_DIR}/logs/${stage}-$(timestamp).log"
    local predictions="${output}/generated_predictions.jsonl"
    [[ -s "${predictions}" ]] || { echo "未生成预测文件：${predictions}" >&2; exit 1; }
    python "${PROJECT_DIR}/scripts/evaluate_predictions.py" \
        --predictions "${predictions}" --output-dir "${output}/evaluation"
    record_state "${stage}_output" "${output}"
    echo ">>> ${stage} 完成：${output}"
}

run_merge() {
    ensure_trained_model
    if [[ "${MODE}" == "full" ]]; then
        record_state merged_output "${TRAIN_OUTPUT}"
        echo ">>> 全量训练结果已是完整模型，无需合并：${TRAIN_OUTPUT}"
        return
    fi
    local output
    output="${PROJECT_DIR}/outputs/merged-model-$(timestamp)"
    (
        cd "${LLAMAFACTORY_DIR}"
        llamafactory-cli export "${PROJECT_DIR}/configs/qwen3vl_merge.yaml" \
            model_name_or_path="${MODEL_PATH:-${QWEN3VL_MODEL_DIR}}" \
            adapter_name_or_path="${TRAIN_OUTPUT}" export_dir="${output}"
    ) 2>&1 | tee -a "${PROJECT_DIR}/logs/merge-$(timestamp).log"
    [[ -f "${output}/config.json" ]] || { echo "合并模型缺少 config.json" >&2; exit 1; }
    compgen -G "${output}/*.safetensors" >/dev/null || { echo "合并模型缺少 safetensors 权重" >&2; exit 1; }
    record_state merged_output "${output}"
    echo ">>> LoRA 合并完成：${output}"
}

run_export() {
    bash "${PROJECT_DIR}/scripts/export_results.sh" "${1:-}"
}

usage() {
    cat <<'EOF'
用法：bash scripts/workflow.sh COMMAND [云盘目录]

命令：
  smoke       前 N 条数据 + 1 step 小样本训练验证
  train       Train Guard 守护的正式训练
  quick-test  正式训练后快速生成并评估前 8 条测试样本
  evaluate    对完整测试集批量生成并评估
  merge       LoRA 合并；全量模型执行完整性确认
  export      导出模型、日志、预测和报告；未给路径时终端询问
  all         依次执行以上全部阶段
EOF
}

command_name="${1:-help}"
case "${command_name}" in
    smoke) run_smoke ;;
    train) run_train ;;
    quick-test) run_prediction "${QUICK_TEST_SAMPLES:-8}" quick-test ;;
    evaluate) run_prediction "${EVAL_SAMPLES:-100000}" batch-eval ;;
    merge) run_merge ;;
    export) run_export "${2:-}" ;;
    all)
        run_smoke
        if [[ "${AUTO_CONFIRM:-0}" != "1" ]]; then
            read -r -p "冒烟测试成功。继续正式训练？输入 YES：" answer
            [[ "${answer}" == "YES" ]] || { echo "已停止。"; exit 0; }
        fi
        run_train
        run_prediction "${QUICK_TEST_SAMPLES:-8}" quick-test
        run_prediction "${EVAL_SAMPLES:-100000}" batch-eval
        run_merge
        run_export "${2:-}"
        ;;
    help|-h|--help) usage ;;
    *) usage >&2; exit 2 ;;
esac
