#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${PROJECT_DIR}/.workflow"
if [[ -f "${PROJECT_DIR}/configs/qwen3vl_lora_sft.yaml" ]]; then MODE="lora"; else MODE="full"; fi

CLOUD_ROOT="${1:-${CLOUD_ROOT:-}}"
if [[ -z "${CLOUD_ROOT}" ]]; then
    read -r -p "请输入云盘挂载目录（例如 /data 或 /mnt/cloud）：" CLOUD_ROOT
fi
[[ -n "${CLOUD_ROOT}" ]] || { echo "云盘目录不能为空" >&2; exit 2; }
[[ "${CLOUD_ROOT}" != "/" ]] || { echo "拒绝把系统根目录作为导出位置" >&2; exit 2; }
mkdir -p "${CLOUD_ROOT}"
CLOUD_ROOT="$(cd "${CLOUD_ROOT}" && pwd)"
[[ -w "${CLOUD_ROOT}" ]] || { echo "云盘目录不可写：${CLOUD_ROOT}" >&2; exit 1; }

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
DEST="${CLOUD_ROOT}/qwen3vl-${MODE}-${stamp}"
mkdir -p "${DEST}"

read_state() { [[ -f "${STATE_DIR}/$1" ]] && cat "${STATE_DIR}/$1"; }
latest_matching_dir() {
    local pattern="$1"
    local matches=()
    mapfile -t matches < <(compgen -G "${pattern}" | sort)
    if (( ${#matches[@]} )); then
        printf '%s\n' "${matches[${#matches[@]} - 1]}"
    fi
}
TRAIN_OUTPUT="$(read_state train_output || true)"
QUICK_OUTPUT="$(read_state quick-test_output || true)"
EVAL_OUTPUT="$(read_state batch-eval_output || true)"
MERGED_OUTPUT="$(read_state merged_output || true)"
if [[ -z "${TRAIN_OUTPUT}" ]]; then
    if [[ "${MODE}" == "lora" ]]; then
        candidate="${PROJECT_DIR}/outputs/qwen3vl-4b-lora-wsi"
    else
        candidate="${PROJECT_DIR}/outputs/qwen3vl-4b-full-wsi"
    fi
    [[ -d "${candidate}" ]] && TRAIN_OUTPUT="${candidate}"
fi
[[ -n "${QUICK_OUTPUT}" ]] || QUICK_OUTPUT="$(latest_matching_dir "${PROJECT_DIR}/outputs/quick-test-*" || true)"
[[ -n "${EVAL_OUTPUT}" ]] || EVAL_OUTPUT="$(latest_matching_dir "${PROJECT_DIR}/outputs/batch-eval-*" || true)"
if [[ -z "${MERGED_OUTPUT}" ]]; then
    if [[ "${MODE}" == "lora" ]]; then
        MERGED_OUTPUT="$(latest_matching_dir "${PROJECT_DIR}/outputs/merged-model-*" || true)"
    else
        MERGED_OUTPUT="${TRAIN_OUTPUT}"
    fi
fi

copy_dir() {
    local source="$1" name="$2"
    if [[ -n "${source}" && -d "${source}" ]]; then
        echo ">>> 导出 ${name}: ${source}"
        cp -a "${source}" "${DEST}/${name}"
    else
        echo ">>> WARNING: 跳过不存在的 ${name}: ${source:-未记录}" >&2
    fi
}

required_kb=0
sources=("${TRAIN_OUTPUT}" "${QUICK_OUTPUT}" "${EVAL_OUTPUT}" "${PROJECT_DIR}/logs" "${PROJECT_DIR}/reports")
if [[ -n "${MERGED_OUTPUT}" && "${MERGED_OUTPUT}" != "${TRAIN_OUTPUT}" ]]; then
    sources+=("${MERGED_OUTPUT}")
fi
for source in "${sources[@]}"; do
    if [[ -n "${source}" && -e "${source}" ]]; then
        size="$(du -sk "${source}" | awk '{print $1}')"
        required_kb=$((required_kb + size))
    fi
done
free_kb="$(df -Pk "${CLOUD_ROOT}" | awk 'NR==2 {print $4}')"
reserve_kb=$((1024 * 1024))
if (( free_kb < required_kb + reserve_kb )); then
    echo "云盘空间不足：预计需要 $((required_kb / 1024)) MiB，另保留 1024 MiB 安全余量" >&2
    exit 1
fi

copy_dir "${TRAIN_OUTPUT}" training_output
copy_dir "${QUICK_OUTPUT}" quick_test
copy_dir "${EVAL_OUTPUT}" batch_evaluation
if [[ -n "${MERGED_OUTPUT}" && "${MERGED_OUTPUT}" != "${TRAIN_OUTPUT}" ]]; then
    copy_dir "${MERGED_OUTPUT}" merged_model
fi
copy_dir "${PROJECT_DIR}/logs" logs
copy_dir "${PROJECT_DIR}/reports" reports
copy_dir "${PROJECT_DIR}/local_tools" local_tools
cp -a "${PROJECT_DIR}/configs" "${DEST}/configs"
cp -a "${PROJECT_DIR}/README.md" "${DEST}/SERVER_README.md"
cp -a "${PROJECT_DIR}/data/dataset_info.json" "${DEST}/dataset_info.json"
cp -a "${PROJECT_DIR}/data/excluded_records.jsonl" "${DEST}/excluded_records.jsonl"

{
    echo "mode=${MODE}"
    echo "exported_at=${stamp}"
    echo "source_project=${PROJECT_DIR}"
    echo "training_output=${TRAIN_OUTPUT}"
    echo "quick_output=${QUICK_OUTPUT}"
    echo "evaluation_output=${EVAL_OUTPUT}"
    echo "merged_output=${MERGED_OUTPUT}"
} > "${DEST}/EXPORT_INFO.txt"
nvidia-smi > "${DEST}/nvidia-smi.txt" 2>&1 || true
python -m pip freeze > "${DEST}/python-environment.txt" 2>&1 || true
python "${PROJECT_DIR}/scripts/build_export_manifest.py" --root "${DEST}" --mode "${MODE}"
printf '%s\n' "${DEST}" > "${STATE_DIR}/last_export_path"
sync || true
echo ">>> 导出完成：${DEST}"
echo ">>> 下载整个目录到本地后运行 local_tools/analyze_results.py 进行复盘。"
