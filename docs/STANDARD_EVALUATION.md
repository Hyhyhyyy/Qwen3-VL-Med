# 统一 13 项评测机制与证据边界

## 输出指标

统一评测机制固定输出以下 13 类结果：ROUGE-L F1、BERTScore F1、BLEU-4、METEOR、CIDEr、严格诊断匹配准确率、病例级 bootstrap 95% CI、ECE、熵代理、目标 token 对视觉 token 的梯度份额、目标 token 到视觉 token 的注意力质量、MMBench 严格循环准确率及 alignment tax、医学文本嵌入余弦相似度。

## 固定协议

- 使用固定测试清单、图像预处理、提示模板、确定性解码参数与指标实现版本。
- 文本、诊断、校准和解释性结果均保留逐病例内部记录，公开仓库只保留代码、协议和聚合结论。
- 严格诊断匹配通过病例级 bootstrap 报告不确定性；同病例 Run 比较优先使用配对检验。
- ECE 使用固定分箱；熵代理只表示确定程度，不能脱离正确率单独解释。
- 公开 benchmark 与临床任务预测必须使用隔离的输出目录，防止结果覆盖或混用。
- 每次运行输出配置摘要、依赖版本、随机种子、样本计数、错误计数和哈希清单。

## 梯度与注意力的当前记录层级

R01、R02、R04–R08 与 R14 的既有机制保存了固定病例样本上的病例级标量，以及每张输入图像的贡献份额。这些数据可以用于跨 Run 分布、箱线图、误差条、相关性和逐图贡献比较，但不能重建空间热图。

要生成论文级词—区域对齐图，下一阶段必须额外保存：

1. 目标 token ID、文本及统一 target span 定义；
2. `target_token × visual_token` 的 attention、绝对 gradient 与 Gradient × Attention 原始矩阵；
3. 最终解码层各 attention head 的独立矩阵；
4. 每张图像的视觉 token 起止区间、patch 网格尺寸及 token 到原图坐标映射；
5. 运行、层号、head、归一化方法、图像顺序与样本索引等 manifest 元数据。

Phase 4固定首批24个代表病例，统一teacher-forced reference target、相同图像顺序和相同解码层。矩阵用float16 NPZ存在受控环境，并自动渲染逐token、逐图和多头热图。原始矩阵、图像和病例manifest不进入公共仓库。

公开归档工具`code/standard_eval/archive_interpretability.py`要求保存`attention_heads[T,H,V]`、视觉表征`abs_gradient[T,V]`、attention-link梯度、逐head Gradient×Attention、target token ID/offset以及视觉token到原图bbox映射。完整协议见[解释性归档规范](INTERPRETABILITY_ARCHIVE.md)。

## 验收门禁

- 测试样本数、病例唯一性、图像数量和错误数满足预注册要求；
- calibration bins 计数总和等于有效病例数；
- benchmark 数据版本与摘要一致，选项解析率达到预设阈值；
- 指标 JSON 可被模式校验，所有有限数值处于合法范围；
- 训练配置、最终产物、重载推理和评测结果均生成独立 SHA-256；
- 发布前执行当前树隐私审计、完整 Git 历史扫描、Python 语法测试和发布哈希核验。
- Phase 4 Run还必须通过24病例原始矩阵完整率、shape/NaN/Inf、token空间映射和兼容标量重算门禁。
