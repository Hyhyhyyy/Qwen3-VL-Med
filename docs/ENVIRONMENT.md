# Qwen3-VL-4B LoRA 与全量微调训练包

把整个目录上传到一台 **Linux x86_64 + NVIDIA GPU** 云服务器，然后完成安装、数据替换和训练。脚本不依赖特定云盘挂载路径，默认把环境放在本目录的 `runtime/`。

> “任何云服务器”仍需满足硬件前提：NVIDIA 驱动可用（`nvidia-smi` 正常）、有足够的显存和磁盘。脚本不会安装内核级 GPU 驱动。

## 1. 一次性安装

```bash
cd /path/to/this-folder
chmod +x install.sh download_model.sh train.sh
bash install.sh
```

安装脚本会：创建不继承系统包的 Python 3.11 虚拟环境、安装 CUDA PyTorch 2.6.0、固定安装 LlamaFactory v0.9.5、Transformers 5.6.0、NumPy 1.26.4、DeepSpeed 0.18.0 和视觉依赖，下载模型，并验证 `Qwen3VLForConditionalGeneration`、模型类型、权重、处理器和 CUDA。LoRA 是普通 BF16/FP16 LoRA，不使用 bitsandbytes 量化；全量微调使用 DeepSpeed ZeRO-3。

默认 PyTorch 使用 CUDA 12.4。若云镜像要求其他构建：

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 bash install.sh
```

模型默认从 ModelScope 下载；也可以使用 Hugging Face 镜像：

```bash
MODEL_SOURCE=huggingface bash install.sh
```

将环境和模型放到独立数据盘：

```bash
INSTALL_ROOT=/data/qwen3vl-runtime bash install.sh
```

安装位置会记录在项目根目录的 `.install_root`，之后直接运行 `bash train.sh` 即可，无需重复设置 `INSTALL_ROOT`。

## 2. 准备视觉训练数据

把你现有的整个 `wsi_train` 图片文件夹放到 `data/wsi_train/`。训练和测试标注均由原始 train JSON 生成，并分别放在同级的 `data/wsi_train.json`、`data/wsi_test.json`。

原始 train 中编号大于 7000 的记录已排除。编号 1–7000 范围内共有 19082 条可用记录，其中按患者编号整体、固定哈希顺序抽取 656 条作为测试集，剩余 18426 条作为训练集。同一患者编号、患者目录或图片不会同时出现在两个集合中。

推荐目录结构：

```text
data/
├── dataset_info.json       # 数据集注册文件，一般无需修改
├── wsi_train.json          # 训练标注（JSON 数组）
├── wsi_test.json           # 测试标注（JSON 数组）
└── wsi_train/              # 直接放入你已有的 wsi_train 文件夹内容
    ├── 001.jpg
    └── ...
```

每条记录格式：

```json
{"messages":[{"role":"user","content":"<image>描述病理图像并给出诊断依据。"},{"role":"assistant","content":"模型应该学习的答案。"}],"images":["wsi_train/001.jpg"]}
```

规则：

- 每条数据中 `<image>` 的数量必须与 `images` 数组长度完全一致。
- 相对图片路径以 `data/` 为基准，所以 `data/wsi_train/001.jpg` 在 JSON 中写成 `wsi_train/001.jpg`；不要写成 `data/wsi_train/001.jpg`。
- `messages` 从 `user` 开始，训练目标放在 `assistant` 消息中。
- 数据集注册已经写入 `data/dataset_info.json`：训练集名称为 `wsi_train`，测试集名称为 `wsi_test`。

手动检查数据：

```bash
runtime/venv/bin/python scripts/validate_dataset.py --dataset-dir data --name wsi_train --check-images
runtime/venv/bin/python scripts/validate_dataset.py --dataset-dir data --name wsi_test --check-images
```

## 3. 先跑 LoRA

默认使用较稳妥的单卡 batch=1、梯度累积 8：

```bash
bash train.sh
```

建议第一次先跑 1 个训练 step，确认模型加载、图片预处理、反向传播和保存路径完整可用：

```bash
OUTPUT_DIR="$PWD/outputs/smoke-test" bash train.sh \
  max_samples=8 max_steps=1 logging_steps=1 save_strategy=no overwrite_output_dir=true
```

冒烟测试成功后再执行正式训练命令。`train.sh` 会在启动前校验训练集和测试集的全部路径及图片可读性；不支持 bf16 的 GPU 会自动切换 fp16。

高显存 GPU 使用原始 batch=2、梯度累积 4：

```bash
CONFIG="$PWD/configs/qwen3vl_lora_sft_high_vram.yaml" bash train.sh
```

断点续训或临时覆盖参数：

```bash
bash train.sh resume_from_checkpoint=/path/to/checkpoint learning_rate=3e-5
```

限定 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh
```

输出默认写入 `outputs/qwen3vl-4b-lora-wsi/`。主配置为 `configs/qwen3vl_lora_sft.yaml`。

## 4. 再跑全量微调

LoRA 跑通并确认数据、损失和保存均正常后，再独立启动全量微调：

```bash
TRAIN_MODE=full OUTPUT_DIR="$PWD/outputs/full-smoke-test" bash train.sh \
  max_samples=8 max_steps=1 logging_steps=1 save_strategy=no overwrite_output_dir=true
```

冒烟测试通过后开始正式全量训练：

```bash
TRAIN_MODE=full bash train.sh
```

全量配置为 `configs/qwen3vl_full_sft.yaml`，默认输出到 `outputs/qwen3vl-4b-full-wsi/`。该配置会解冻语言模型、视觉塔和多模态投影器，并使用 DeepSpeed ZeRO-3；建议多张 80GB 级 GPU。多卡有效 batch 等于 `per_device_train_batch_size × gradient_accumulation_steps × GPU 数量`，GPU 数量变化时应相应调整梯度累积，避免实验间有效 batch 不一致。

“先 LoRA、后全量”默认是两次独立实验，都从同一个基础模型开始。不要把 LoRA checkpoint 直接填入全量配置的 `resume_from_checkpoint`，因为优化器和可训练参数集合不同。如果需要让全量训练继承 LoRA 效果，应先把 LoRA adapter 合并到基础模型，再用 `MODEL_PATH=/path/to/merged-model TRAIN_MODE=full bash train.sh` 启动。

## 显存说明

LoRA 并不量化基础模型。高分辨率图片、长上下文和 batch=2 都会明显增加显存占用。默认配置对齐官方 Qwen3-VL LoRA 示例的核心设置：`lora_rank: 8`、`cutoff_len: 2048`、图片最多 262144 像素、batch=1；如果仍然 OOM，可在命令行覆盖：

```bash
bash train.sh cutoff_len=1024 image_max_pixels=131072 per_device_train_batch_size=1
```

建议使用至少 24GB 显存，32GB 更稳妥；高显存 batch=2 配置建议 40GB 以上。显存需求仍会随实际 JPG 尺寸和文本长度变化。
