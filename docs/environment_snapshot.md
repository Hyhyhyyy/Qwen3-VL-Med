# 环境快照

- 远端：<GPU cloud provider>，4× NVIDIA A100 40GB，约 360GiB RAM。
- Python：3.12。
- torch：2.5.1+cu121。
- Transformers：5.6.0。
- LLaMA-Factory：0.9.5。
- DeepSpeed：0.18.0。
- attention：SDPA；Qwen3-VL 当前环境下 FlashAttention2 曾因 `s_aux=None` 不兼容。

最终封版还需保存：`nvidia-smi`、驱动/CUDA、`pip freeze`、conda 显式列表、关键源码 commit、OS、磁盘类型、环境变量白名单、随机种子与确定性设置。不得保存 token、密码和 SSH 私钥。

