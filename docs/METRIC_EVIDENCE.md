# 指标、计算依据与证据等级

## 分级原则

- **A：临床终点。** 同目标病理任务，由病理医师盲评并完成外部验证。
- **B：相邻医学领域专家对齐指标。** 例如经放射科医师验证的报告指标，迁移到病理仍须重新验证。
- **C：公认通用方法。** BLEU、ROUGE、chrF、BERTScore、P/R/F1、MCC、Cohen's kappa及bootstrap CI。
- **D：项目规则或简化变体。** Clinical Fact、Hallucination/Omission、HARE-style、CRQS-style、METEOR-exact和CIDEr-lite。
- **E：工程指标。** 延迟、token、显存、长度、重复率和结构字段率。

当前自动评测中没有A等级临床终点。

## 核心公式

- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2PR / (P + R)
- Hallucination reference proxy = 参考文本未支持的预测事实数 / 预测事实数
- Omission reference proxy = 预测未覆盖的参考事实数 / 参考事实数
- Clinical discordance proxy = 共同实体中的状态冲突数 / 实体并集数
- 项目CRQS-style = [Clinical Fact Recall + Key Fact Recall + (1-Hallucination) + (1-Discordance)] / 4

项目CRQS-style不是PathReportEval官方CRQS的等价实现，权重也未经过肝病理临床校准。

## 重要实现差异

- `METEOR-exact`只使用中文精确token匹配，不包含官方METEOR的同义词、词干等资源。
- `CIDEr-lite`使用单参考1-4 gram TF-IDF余弦相似度并乘10，不等同于完整多参考CIDEr/CIDEr-D。
- BERTScore即使采用公认算法，通用中文编码器也不等于肝病理专用语义模型。
- 参考报告规则代理不查看原始图像；正确但参考报告未写出的发现可能被误判为幻觉。

## 推荐临床主终点

面向4-8图病理报告，模型选择优先级应为：病理医师盲评重大错误率、关键疾病病例级敏感度、图像不支持事实率、关键遗漏率、报告可接受率和医师+AI后的重大错误残留率。自动文本指标仅作为次级回归指标。

## 主要方法来源

- BLEU: Papineni et al., ACL 2002, https://aclanthology.org/P02-1040/
- ROUGE: Lin, ACL Workshop 2004, https://aclanthology.org/W04-1013/
- METEOR: Banerjee and Lavie, ACL Workshop 2005, https://aclanthology.org/W05-0909/
- chrF: Popovic, WMT 2015, https://aclanthology.org/W15-3049/
- CIDEr: Vedantam et al., CVPR 2015
- BERTScore: Zhang et al., ICLR 2020, https://openreview.net/forum?id=SkeHuCVFDr

