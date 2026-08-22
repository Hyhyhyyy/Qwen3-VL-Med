#!/usr/bin/env bash
set -u

PY=/opt/conda/envs/qwen3vl/bin/python
echo '=== ENV ==='
"$PY" - <<'PY'
import sys
import torch
import transformers
import deepspeed
import llamafactory
print('python', sys.version.replace('\n', ' '))
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('cuda_available', torch.cuda.is_available(), 'gpu_count', torch.cuda.device_count())
print('transformers', transformers.__version__)
print('deepspeed', deepspeed.__version__)
print('llamafactory', getattr(llamafactory, '__version__', 'unknown'))
PY

echo '=== FILES ==='
find /path/to/cluster -maxdepth 3 -type f -printf '%TY-%Tm-%Td %TH:%TM %10s %p\n' 2>/dev/null | sort

echo '=== CONFIG ==='
sed -n '1,300p' /path/to/cluster 2>/dev/null

echo '=== DEEPSPEED CONFIGS ==='
for file in /path/to/cluster do
  echo "FILE: $file"
  cat "$file"
done

echo '=== TRAIN SCRIPT ==='
sed -n '1,420p' /path/to/cluster 2>/dev/null

echo '=== ARTIFACTS ==='
find /path/to/cluster /path/to/cluster /path/to/cluster \
  -maxdepth 5 -type f -printf '%TY-%Tm-%Td %TH:%TM %12s %p\n' 2>/dev/null | sort

echo '=== LOG TAILS ==='
for file in /path/to/cluster /path/to/cluster do
  [[ -f "$file" ]] || continue
  echo "----- $file (last 160 lines) -----"
  tail -160 "$file"
done

echo '=== DATA ==='
ls -lah /path/to/cluster 2>/dev/null
wc -l /path/to/cluster 2>/dev/null || true
du -sh /path/to/cluster /path/to/cluster 2>/dev/null

echo '=== NETWORK ==='
ip -br addr 2>/dev/null || true
ip route 2>/dev/null || true
ls -l /sys/class/net 2>/dev/null || true
echo '=== NCCL LIBS ==='
"$PY" -m pip show nvidia-nccl-cu12 torch deepspeed transformers llamafactory 2>/dev/null | sed -n '1,260p'
