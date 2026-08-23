# 配置与 JSON 预制要求（PREFLIGHT）

本文件说明 `configs/` 下各 YAML / DeepSpeed JSON **使用前必须预置**的项。JSON 不支持注释，
因此所有预制要求集中写在这里；YAML 模板里的 `{{...}}` 占位符也必须先替换再使用。

> 安全边界：本仓库**不包含**任何模型权重、checkpoint、训练数据、逐样本结果、内部主机名、
> 账户或凭据。所有路径均为占位符，请在你自己的隔离环境中替换为真实本地路径。

---

## 0. 通用环境变量（本地 `env.sh`，不进仓库）

以下脚本（`scripts/train_full_remote.sh`、`scripts/remote_qwen3vl_diagnose.sh`、
`configs/full_sft/qwen3vl_a100_full_sft.yaml.in`）依赖一个**本地** `env.sh`，它**不提交到仓库**。
请在你自己的训练机上创建，至少包含：

```bash
export TRAIN_ROOT="/path/to/your/training/root"   # 训练根目录（放 configs/ data/ output/）
export MODEL_DIR="${TRAIN_ROOT}/models/Qwen3-VL-4B-Instruct"  # 已下载的基础模型目录
export DATA_DIR="${TRAIN_ROOT}/data"             # 含 wsi_train.json / wsi_test.json 的目录
export DEEPSPEED="${TRAIN_ROOT}/configs/ds_a100_z3_nooffload.json"  # DeepSpeed 配置绝对路径
export GPU_MEM_GB="40"        # 单卡显存；80GB 卡用 ZeRO-2，40GB 卡用 FSDP+8bit 或 ZeRO-3 无 offload
export GPU_COUNT="4"          # 实际可见 GPU 数量（与 YAML 的梯度累积配合，保证有效 batch 一致）
```

`scripts/train.sh`（Environment 版）不依赖 `env.sh`，改用 `INSTALL_ROOT/.install_root` +
`install.sh` 生成的 `env.sh`，路径全部参数化，更适合作隔离安装包发布。

---

## 1. `configs/ds_a100_z3_nooffload.json`（DeepSpeed）

- **作用**：ZeRO-3、优化器/参数均**不 offload**（`offload_optimizer.device: none`、
  `offload_param.device: none`）。适合多张高显存 GPU（如 4×A100-40GB）做全量微调。
- **预制要求**：
  - 仅当 YAML 里带 `deepspeed:` 字段时生效；FSDP 路线（单卡 40GB）不含该字段，走普通启动。
  - `stage3_gather_16bit_weights_on_model_save: true` 会在保存时聚合 16bit 权重，确保产出完整模型。
  - 单卡 **40GB** 显存跑 ZeRO-3 无 offload 极易 OOM；此类机器请改用 ZeRO-2 配置或 FSDP+8bit 路线。
  - 本文件为**通用 DeepSpeed 模板**，无需修改即可被 LLaMA-Factory 引用；所有 `auto` 字段由
    LLaMA-Factory 在启动时按 YAML 自动填充。

---

## 2. `configs/full_sft/qwen3vl_full_sft.yaml`（全量微调基础配置）

- **预制要求**：
  - `model_name_or_path: Qwen/Qwen3-VL-4B-Instruct` 为基础模型 Hugging Face ID；若已下载到本地，
    改为本地绝对路径（如 `${MODEL_DIR}`）。
  - `dataset: wsi_train` 需在 `dataset_dir` 下的 `dataset_info.json` 中注册；**数据集文件不进仓库**，
    请替换为你的数据。
  - `deepspeed: examples/deepspeed/ds_z3_config.json` 指向 LLaMA-Factory 自带示例；如需本仓库的
    ZeRO-3 无 offload 配置，改为 `configs/ds_a100_z3_nooffload.json`（或绝对路径 `${DEEPSPEED}`）。
  - 全量微调会解冻语言模型、视觉塔、投影器；请确保多卡 + 充足显存。
  - **关键四件套**（消除 image-token mismatch）：`image_max_pixels` / `image_min_pixels` 保持一致、
    不写 `image_max_token_num`、预处理 batch 不要过大。

---

## 3. `configs/full_sft/qwen3vl_a100_full_sft.yaml.in`（A100 全量微调模板）

- **预制要求**：本文件含 `{{TRAIN_ROOT}}`、`{{DATA_DIR}}`、`{{DEEPSPEED}}` 三个占位符，**必须替换**后
  才能作为 LLaMA-Factory 配置使用：
  - `{{TRAIN_ROOT}}` → 训练根目录绝对路径（见第 0 节）。
  - `{{DATA_DIR}}` → 含 `dataset_info.json` 与 `wsi_train.json` 的目录。
  - `{{DEEPSPEED}}` → DeepSpeed JSON 绝对路径（如 `configs/ds_a100_z3_nooffload.json` 或 `${DEEPSPEED}`）。
  - 原项目用 `apply_paths.sh` 从 `env.sh` 自动展开；本仓库不提供 `apply_paths.sh`，请手动 `sed` 替换或
    直接编辑成绝对路径后再用。
  - 单卡 A100-80GB 用 ZeRO-2，多卡 A100 用 ZeRO-3 无 offload（见 `deepspeed:` 字段）。
  - `flash_attn: sdpa` 是针对 Transformers 5.6.0 + Qwen3-VL 的稳定选择（FA2 曾因 `s_aux=None` 崩溃）。

---

## 4. `configs/lora/qwen3vl_lora_sft.yaml` 与 `qwen3vl_lora_sft_high_vram.yaml`

- **预制要求**：
  - 两者均为 LoRA 微调（`finetuning_type: lora`，`lora_target: all`）。
  - `qwen3vl_lora_sft.yaml`：单卡 `batch=1`、梯度累积 8（稳妥默认）。
  - `qwen3vl_lora_sft_high_vram.yaml`：高显存卡 `batch=2`、梯度累积 4（建议 40GB 以上）。
  - `model_name_or_path` / `dataset` 同第 2 节要求。
  - LoRA 不量化基础模型；显存随分辨率、上下文、batch 增大而上升。

---

## 5. 硬件前提（来自 `docs/environment_config.csv` / `docs/environment_snapshot.md`）

- 验证环境：4× NVIDIA A100 40GB、约 360GiB RAM、Python 3.12、torch 2.5.1+cu121、
  Transformers 5.6.0、LLaMA-Factory 0.9.5、DeepSpeed 0.18.0、attention=SDPA。
- 全量微调建议多张高显存 GPU；单卡 40GB 走 FSDP+8bit 或 ZeRO-2，避免 ZeRO-3 无 offload 的 OOM 陷阱。
- 随机种子默认 42；如需严格可复现，请显式固定并验证每轮运行的确定性设置（见环境快照待办）。

---

## 6. 预制检查清单（训练前）

- [ ] 基础模型已下载到 `MODEL_DIR`（或改为 HF ID）。
- [ ] `dataset_info.json` 已注册 `wsi_train` / `wsi_test`，且数据文件就位（**非本仓库数据**）。
- [ ] YAML 中的 `{{...}}` 占位符已全部替换为真实本地绝对路径。
- [ ] DeepSpeed JSON 路径正确，且与单卡/多卡显存匹配（ZeRO-2 vs ZeRO-3 无 offload）。
- [ ] 已运行 `scripts/validate_dataset.py` 校验图片路径与 `<image>` 数量一致。
- [ ] 已运行 1-step 冒烟测试确认加载/预处理/反向/保存链路完整。
- [ ] 确认仓库与本地均无 token / 密码 / SSH 私钥 / 内部主机名（见 `SECURITY.md`）。
