# Qwen3-VL-Med

面向 Qwen3-VL 医疗领域微调研究的代码级公开版本，已执行隐私筛查与内部数据隔离。

维护者：**Hyhyhyyy**

## 隐私边界

本公开仓库**不包含任何临床图像、病例级记录、报告、预测结果、模型权重、LoRA adapter、内部主机名、访问凭据或机构存储路径**。仓库中的 JSON 示例完全由人工合成，不对应任何真实人员。

真实临床数据必须保存在具备访问控制的院内环境中。请勿将加密后的临床数据压缩包提交到公开仓库：Git 历史具有长期可追溯性，未来一旦密钥泄露，密文仍可能被解密。

## 仓库包含内容

- 清洗后多图训练、单图训练、病例级诊断及“证据到诊断”任务的全量微调配置模板。
- 四组受控 LoRA 冻结实验：
  - R04：视觉 LoRA＋多模态投影层全量训练＋语言 LoRA；
  - R05：冻结视觉塔；
  - R06：冻结多模态投影层；
  - R07：冻结语言主干。
- 数据清洗与任务转换代码。
- 单图到病例级聚合及评测代码。
- adapter 组件审计和配对 bootstrap 比较工具。
- 提交前默认拒绝敏感文件的隐私审计脚本。

## 仓库明确不包含

- 医院数据集或由其产生的逐行数据产物；
- 图像、直接或间接标识符、自由文本报告、排除记录、审计明细及预测文件；
- 基于私有队列计算的实验结果；
- 模型、adapter、优化器或 checkpoint 权重；
- 机器专用部署脚本、远端地址和运行状态清单。

## 目录结构

```text
code/                         数据构建、清洗、推理与评测代码
configs/full_sft/             全量微调配置模板
configs/eval/                 评测配置示例
experiments/lora_freezing/    四组 LoRA 冻结实验与审计工具
examples/                     仅含合成数据示例
docs/                         实验设计、数据治理与隐私审计报告
tools/privacy_audit.ps1       提交前隐私门禁
```

## 快速开始

1. 在隔离环境中安装 Qwen3-VL、LLaMA-Factory、PyTorch、Transformers，以及 `requirements.txt` 中的可选依赖。
2. 将私有数据集保存在本仓库之外。
3. 复制所需 YAML 模板，在受保护环境中替换 `/path/to/...` 占位路径；不要提交包含真实路径的本地配置。
4. 在院内环境中向 LLaMA-Factory 注册私有数据集。
5. 每次提交前运行隐私门禁：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\privacy_audit.ps1
```

6. 使用 LLaMA-Factory 启动配置，例如：

```bash
FORCE_TORCHRUN=1 llamafactory-cli train experiments/lora_freezing/lora_01_r04_all_components.yaml
```

仓库内 YAML 均为公开模板，其中的占位路径不会指向任何真实数据。

## 研究设计

详细实验矩阵见 [实验设计](docs/EXPERIMENT_DESIGN.md)。数据处理、院内加密和密钥分离要求见 [数据治理](docs/DATA_GOVERNANCE.md)。公开包的审计范围见 [隐私审计报告](docs/PRIVACY_AUDIT_REPORT.md)。

## 临床使用警告

本仓库仅用于研究，不属于医疗器械。基于参考答案计算的指标不能证明临床安全性或诊断性能。在任何临床使用前，必须进行独立的病理专家复核，并遵守所在机构的数据治理和伦理审批要求。
