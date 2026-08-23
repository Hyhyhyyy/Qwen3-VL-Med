#!/usr/bin/env bash
# =============================================================
#  A100 全量微调 启动脚本（脱离 SSH，崩溃可续训）
#  用法：
#    bash scripts/train_full.sh          # 正常启动（数据到位后）
#    FRESH=1 bash scripts/train_full.sh  # 改了分辨率/cutoff/template 后清缓存重 tokenize
# =============================================================
set -euo pipefail

# Non-interactive SSH/setsid sessions do not inherit `conda activate myconda`.
# LLaMA-Factory launches `torchrun` by name, so keep the environment bin first.
CONDA_ENV_BIN="${CONDA_ENV_BIN:-/opt/conda/envs/qwen3vl/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# ===== 定位项目根目录 =====
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# ===== 读取并应用路径（env.sh 是唯一需要手动改的地方）=====
# shellcheck disable=SC1091
source ./env.sh
bash ./apply_paths.sh

MODEL_DIR=$TRAIN_ROOT/models/Qwen3-VL-4B-Instruct
DATA_DIR=${DATA_DIR:-$TRAIN_ROOT/data}   # 默认 TRAIN_ROOT/data；env.sh 的 DATA_DIR 指向网盘数据目录
CFG=$ROOT/configs/qwen3vl_a100_full_sft.yaml
LOG=$TRAIN_ROOT/output/a100_fullft.log
mkdir -p "$TRAIN_ROOT/output"

grep -qE '^stage:[[:space:]]*sft([[:space:]]|$)' "$CFG" || {
  echo "[X] 配置缺少 stage: sft，拒绝以非训练模式启动"; exit 1;
}
grep -qE '^do_train:[[:space:]]*true([[:space:]]|$)' "$CFG" || {
  echo "[X] 配置缺少 do_train: true，拒绝只加载模型后 exit 0"; exit 1;
}

# ---------- 0) GPU 显存自检（防"包说 80G、实际 40G"导致 CUDA OOM/首步崩）----------
GPU_MEM_GB="${GPU_MEM_GB:-80}"
# 注意：不要用 `nvidia-smi ... | head -1`，在 set -o pipefail 下 head 提前关管道会让
# nvidia-smi 收 SIGPIPE(141)，整条管道以 141 退出触发 set -e 误杀脚本。先抓到变量再处理。
ACTUAL_MB_RAW=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)
ACTUAL_MB=$(printf '%s' "$ACTUAL_MB_RAW" | tr -d ' ')
ACTUAL_MB=${ACTUAL_MB%%$'\n'*}
if [ -n "$ACTUAL_MB" ]; then
  ACTUAL_GB=$(( (ACTUAL_MB + 1023) / 1024 ))
  echo "[i] 实际 GPU 显存 = ${ACTUAL_GB}GB（env.sh GPU_MEM_GB=${GPU_MEM_GB}）"
  if [ "$GPU_MEM_GB" = "80" ] && [ "$ACTUAL_GB" -lt 70 ]; then
    echo "[X] 显存不符：env.sh 设 GPU_MEM_GB=80（走 ZeRO-2），但本机仅 ${ACTUAL_GB}GB。"
    echo "    40GB 卡跑 ZeRO-2 必 CUDA OOM。请编辑 env.sh 改 GPU_MEM_GB=40（自动切 FSDP+8bit 路线）。"
    exit 1
  fi
  if [ "$GPU_MEM_GB" = "40" ] && [ "$ACTUAL_GB" -ge 70 ]; then
    echo "[!] 提示：env.sh 设 GPU_MEM_GB=40（FSDP+8bit），但本机有 ${ACTUAL_GB}GB，可改回 80 用更稳的 ZeRO-2。"
  fi
else
  echo "[!] 无法读取 GPU 显存，跳过自检（请确认 nvidia-smi 可用）"
fi

# ---------- 0.5) Python 环境快速自检（防盲启导致 torch/flash/deepspeed 连环错）----------
python - <<PY || { echo "[X] 环境自检失败，请勿启动训练，先跑 bash setup_env.sh"; exit 1; }
import importlib.metadata as m, sys, os
ok = True

try:
    import torch
    print("[i] torch       :", torch.__version__, "| cuda", torch.version.cuda,
          "| avail", torch.cuda.is_available(), "| devices", torch.cuda.device_count())
    if not torch.cuda.is_available():
        print("[X] CUDA 不可用"); ok = False
    if not torch.__version__.startswith("2.5.1"):
        print("[!] 警告: torch 不是镜像原装 2.5.1+cu121（当前 %s），flash-attn 可能失效" % torch.__version__)
except Exception as e:
    print("[X] torch       :", e); ok = False

try:
    import flash_attn; print("[i] flash_attn  :", flash_attn.__version__)
except Exception as e:
    print("[X] flash_attn  :", e); ok = False

try:
    import deepspeed; ver = m.version("deepspeed")
    print("[i] deepspeed   :", deepspeed.__version__, "(metadata OK:", ver, ")")
except Exception as e:
    print("[X] deepspeed   :", e); ok = False

gpu_mem_gb = os.environ.get("GPU_MEM_GB", "80")
gpu_count  = os.environ.get("GPU_COUNT", "1")
if gpu_mem_gb == "40" and gpu_count == "1":
    try:
        import bitsandbytes; print("[i] bitsandbytes:", bitsandbytes.__version__)
    except Exception as e:
        print("[X] bitsandbytes:", e); ok = False

sys.exit(0 if ok else 1)
PY

# 同理：先抓到变量再处理，避免 nvidia-smi | head -1 在 pipefail 下触发 SIGPIPE(141) 误杀脚本
ACTUAL_COUNT_RAW=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null)
ACTUAL_COUNT=$(printf '%s' "$ACTUAL_COUNT_RAW" | tr -d ' ')
ACTUAL_COUNT=${ACTUAL_COUNT%%$'\n'*}
if [ -n "$ACTUAL_COUNT" ] && [ "$ACTUAL_COUNT" != "${GPU_COUNT:-1}" ]; then
  echo "[X] GPU 数量不符: env.sh GPU_COUNT=${GPU_COUNT:-1}, 实际 nvidia-smi 看到 ${ACTUAL_COUNT} 张"
  echo "    请修改 env.sh 中的 GPU_COUNT 为 ${ACTUAL_COUNT} 后再启动"
  exit 1
fi

# 2×40G 是已知陷阱：ZeRO-3 无 offload 单卡峰值 ~40GB 必 OOM；CPU offload 又吃 90GB 内存且慢
if [ "${GPU_COUNT:-1}" -eq 2 ] && [ "${GPU_MEM_GB:-80}" -lt 70 ]; then
  echo "[!] 警告: GPU_COUNT=2 且单卡 <70GB —— 2×40G 是已知陷阱，ZeRO-3 无 offload 单卡峰值≈40GB 必 OOM。"
  echo "    强烈建议租 4 张 A100-40GB（GPU_COUNT=4）再跑，否则首步即 CUDA OOM。是否继续由你决定。"
fi

# ---------- 1) 数据到位检查（传输完成后才启动）----------
if [ ! -f "$DATA_DIR/wsi_train.json" ]; then
  echo "[X] 数据未到位: $DATA_DIR/wsi_train.json 不存在，请先确认传输完成"
  exit 1
fi
echo "[i] 数据已到位: wsi_train.json 存在 ($(wc -c < "$DATA_DIR/wsi_train.json") 字节)"

# ---------- 1.5) 图像路径校验（防 images 指向不存在文件 → 0 样本 / 静默 exit 0）----------
python - "$DATA_DIR" <<'PY'
import json, os, sys
base = sys.argv[1]
jp = os.path.join(base, "wsi_train.json")
if not os.path.isfile(jp):
    print("[!] wsi_train.json 不存在，跳过图像校验"); sys.exit(0)
try:
    data = json.load(open(jp))
except Exception as e:
    print("[X] wsi_train.json 解析失败:", e); sys.exit(1)
miss = 0; tot = 0
for rec in data[:200]:                     # 抽前 200 条判断路径格式
    for im in rec.get("images", []):
        tot += 1
        p = im if os.path.isabs(im) else os.path.join(base, im)
        if not os.path.exists(p):
            miss += 1
            if miss <= 5:
                print("    [缺失] %s -> %s" % (im, p))
if tot == 0:
    print("[!] 前 200 条都没有 images 字段，请确认数据格式")
elif miss == 0:
    print("[i] 图像路径校验通过：抽样 %d 张全部存在" % tot)
else:
    print("[X] %d/%d 张图像缺失（前 200 条抽样），LLaMA-Factory 会因找不到图得到 0 样本/静默退出" % (miss, tot))
    sys.exit(1)
PY

# ---------- 2) 自愈前置：统一 preprocessor min/max_pixels（防 image-token mismatch）----------
PIX=262144
python - "$MODEL_DIR" "$PIX" <<'PY'
import json, sys, pathlib
d, mp = sys.argv[1], int(sys.argv[2])
p = pathlib.Path(d)/"preprocessor_config.json"
if not p.exists():
    print("[!] preprocessor_config.json 不存在，跳过 patch")
    sys.exit(0)
c = json.loads(p.read_text(encoding="utf-8"))
c["max_pixels"]=mp; c["min_pixels"]=mp
c["size"]={"longest_edge":mp,"shortest_edge":mp}
p.write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")
print("patched preprocessor_config.json -> max/min pixels =", mp)
PY

# ---------- 3) 可选清 tokenize 缓存 ----------
export FRESH="${FRESH:-0}"
if [ "$FRESH" = "1" ]; then
  rm -rf "$HOME/.cache/llamafactory" "$HOME/.cache/huggingface/datasets"
  echo "[i] FRESH=1 已清 tokenize 缓存"
fi

# ---------- 3.5) 陈旧「已完成」trainer_state 防护 ----------
# 若 output_dir 下已有标记训练「已完成」的 trainer_state.json，resume_from_checkpoint:true
# 会误判「训完了」→ 进程立刻 exit 0（无任何报错，极易误判为平台 kill）。必须先拦截。
STATE=$TRAIN_ROOT/output/full_sft/trainer_state.json
MAX_EPOCH=$(grep -E '^num_train_epochs' "$CFG" | head -1 | grep -oE '[0-9.]+' | head -1)
MAX_EPOCH=${MAX_EPOCH:-3.0}
if [ -f "$STATE" ]; then
  python - "$STATE" "$MAX_EPOCH" <<'PY'
import json, sys
sp, max_ep = sys.argv[1], float(sys.argv[2])
try:
    d = json.load(open(sp))
except Exception as e:
    print("[!] 无法读取 trainer_state.json:", e); sys.exit(0)
ep = d.get("epoch"); gs = d.get("global_step", 0); ms = d.get("max_steps")
done = False
if ep is not None and float(ep) >= max_ep: done = True
if ms and gs and int(gs) >= int(ms): done = True
if done:
    print("[X] 检测到 output/full_sft/trainer_state.json 标记训练『已完成』(epoch=%s, global_step=%s, max_steps=%s)" % (ep, gs, ms))
    print("    当前 resume_from_checkpoint:true + overwrite_output_dir:false 会误判『训完了』→ 进程立刻 exit 0。")
    print("    解决（二选一）：")
    print("      rm -rf /path/to/cluster   # 清空后重跑（从头训）")
    print("      或改 configs/qwen3vl_a100_full_sft.yaml.in 的 overwrite_output_dir: true")
    sys.exit(1)
else:
    print("[i] trainer_state 未标记完成 (epoch=%s, global_step=%s)，将从最近 checkpoint 续训" % (ep, gs))
PY
  if [ $? -ne 0 ]; then exit 1; fi
fi

# ---------- 4) 脱离 SSH 启动 ----------
#  LLaMA-Factory 0.9.5 硬性规定：配置里带 deepspeed 时，必须用 FORCE_TORCHRUN=1 启动，
#  否则直接抛 "Please use `FORCE_TORCHRUN=1` to launch DeepSpeed training."。
#  FORCE_TORCHRUN=1 会让 llamafactory-cli 内部自动 torchrun 拉起 nproc_per_node=可见卡数
#  的进程（多卡自动分片 ZeRO-3，单卡则 nproc=1 跑 ZeRO-2），无需我们手写 torchrun。
#  只有 FSDP 路线（单卡 40G）不含 deepspeed 字段，走普通 llamafactory-cli。
if [ "${GPU_COUNT:-1}" -ge 2 ]; then
  LAUNCH="FORCE_TORCHRUN=1 llamafactory-cli train $CFG"
  echo "[i] 多卡模式: GPU_COUNT=${GPU_COUNT} -> FORCE_TORCHRUN=1 让 LLaMA-Factory 自动 torchrun 拉起 ${GPU_COUNT} 进程 (ZeRO-3 跨卡分片)"
elif grep -q '^deepspeed:' "$CFG"; then
  LAUNCH="FORCE_TORCHRUN=1 llamafactory-cli train $CFG"
  echo "[i] 单卡 ZeRO-2 模式: FORCE_TORCHRUN=1 (LLaMA-Factory 内部 torchrun nproc=1)"
else
  LAUNCH="llamafactory-cli train $CFG"
  echo "[i] 单卡 FSDP 模式: llamafactory-cli train (无 deepspeed)"
fi

setsid nohup bash -c "$LAUNCH > $LOG 2>&1" &
disown
echo "[i] 已后台启动。日志: $LOG"
echo "[i] 监测: tail -f $LOG"
echo "[i] 进程监测: pgrep -af 'llamafactory.cli train'"
