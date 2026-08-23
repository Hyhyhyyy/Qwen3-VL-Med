# 复现与核验流程

## 1. 数据边界

私有数据必须位于Git仓库之外。只允许提交字段结构、人工合成示例和不含真实路径的模板。训练、验证、测试必须按病例隔离，避免同病例的不同图像跨集合泄漏。

## 2. 环境门槛

正式训练前依次验证：

1. Python、CUDA、PyTorch、Transformers、PEFT和训练框架版本；
2. GPU数量、可用显存、磁盘和共享内存；
3. 模型与processor可加载；
4. 训练数据格式、图像存在性和输入输出字段；
5. 10-step smoke test可以前向、反向和更新参数；
6. checkpoint可以保存、重新加载并继续训练；
7. 评测脚本可对小样本端到端运行。

## 3. 参数状态审计

训练前记录视觉塔、投影器、语言主干各自处于Full、LoRA还是冻结状态；训练后检查adapter配置、可训练参数和实际保存张量，避免仅根据文件夹名称判断。

## 4. 评测血缘

每次正式评测应保存：

- 代码提交、配置文件和评测版本；
- 模型/adapter身份和文件哈希；
- 数据manifest的受控环境哈希；
- 固定解码参数和随机种子；
- 逐记录基础指标、聚合指标和置信区间；
- 失败样本索引（仅在受控环境内）；
- 生成时延、token、显存和异常日志。

## 5. 可复现级别

- 固定seed只表示随机源尽量受控。
- 若未启用并验证确定性算法，不声明位级复现。
- 分布式训练、CUDA kernel和依赖版本差异可能导致数值漂移。
- 对关键结论使用同病例配对比较；若结果接近边界，应增加随机种子和外部队列。

## 6. 公开发布

公开包从空目录按白名单构建，不直接在私有工程树执行`git add -A`。发布前运行隐私门禁、Python编译、单元测试和哈希生成；确认Git diff后再提交。

默认CI运行不依赖大型训练框架的指标规则和视野聚合测试。`experiments/lora_freezing/test_audit_adapter_components.py`需要PyTorch，应在安装训练依赖的隔离环境中额外执行；不得把“缺少PyTorch导致未启动”记录为测试通过。

## 7. 证据鲁棒训练

使用`code/evidence_pipeline/build_robust_evidence_mix.py`从受控环境中的Oracle训练记录、上游训练集预测和显式case manifest构建1:1混合数据。构建器强制split过滤、Oracle索引连续性、图像链接一致性、上游覆盖率和输出SHA-256。测试记录不得参与构建。

若每个病例由一条Oracle记录扩增为Oracle+生成证据两条记录，应相应调整epoch以保持与Oracle-only基线相同的样本暴露量，避免把训练步数翻倍误解释为数据策略收益。

## 8. 原始解释性归档

模型侧提取器必须先保留完整target×head×visual数组，再调用`code/standard_eval/archive_interpretability.py`生成float16 NPZ、空间映射、manifest和热图。不得只将已有病例级均值写入新格式。不同target span使用独立comparison group。
