# Token—视觉token原始解释性归档规范

## 固定比较协议

- 首批固定24个代表病例；病例ID、图像顺序和输入处理一经锁定不得按Run结果更换。
- 病例只按任务无关元数据分层选择，不按某个候选模型的正确/错误选择。
- Qwen3-VL-4B统一使用最后一个语言解码层（0-based Layer 35）和全部32个attention head。
- target采用teacher-forced reference token，排除模板、padding、特殊token和EOS。
- 不同任务使用独立target span：完整报告诊断span、直接诊断span、形态证据span、联合模型证据span与诊断span。
- target span不同的Run不得进入同一个解释性排名。

## 每病例NPZ

主要矩阵使用float16和`numpy.savez_compressed`：

- `attention_heads[T,H,V]`
- `attention_link_abs_gradient_heads[T,H,V]`
- `abs_gradient[T,V]`：目标token log-probability对视觉token输入表征的梯度范数
- `grad_x_attention_heads[T,H,V]`
- `attention[T,V]`
- `grad_x_attention[T,V]`
- target token ID、字符offset、视觉token序列位置、图像索引、网格坐标和原图bbox映射

Gradient×Attention按`abs_gradient[:,None,:] × attention_heads`计算。继续保留历史病例级Gradient/Attention标量，并由原始数组确定性重算；不得以attention-link gradient替代历史视觉表征gradient。

## 多图映射

每张图必须记录原始/处理后尺寸、图像SHA256、视觉token起止区间、patch size、spatial merge size、token网格的`t/h/w`以及token到原图`xyxy`坐标映射。所有区间连续、不重叠且总和等于V。

## Manifest与图件

每个NPZ配套JSON manifest，记录Run、病例、comparison group、target span、target文本/ID、层/head、数组shape/dtype、聚合公式、代码/配置/图像/NPZ哈希和QC状态。自动生成：

- target-token×visual-token矩阵图；
- 逐token Attention、Abs-Gradient和Gradient×Attention图；
- 每张图空间热图；
- 全head注意力图；
- 原图叠加图。

公共工具`code/standard_eval/archive_interpretability.py`负责shape验证、float16 NPZ、manifest、视觉token坐标映射和基础论文图件。模型侧提取器必须将原始张量传入该工具，不得先求和或均值后再归档。

## 验收

- 24例完整率100%，无NaN/Inf；
- 同组Run的病例、target token与解码层完全一致；
- token区间覆盖V且空间坐标位于原图范围；
- 从NPZ重算兼容标量，与13项表绝对误差不超过预注册容差；
- NPZ、临床图像、病例manifest和渲染热图均禁止进入公共Git。

公开schema见`docs/schemas/interpretability_manifest.schema.json`与`fixed_case.schema.json`。
