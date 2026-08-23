# Experiment design

## Full fine-tuning branches

- Cleaned multi-image SFT establishes the full-parameter baseline.
- Single-view SFT isolates view-level morphology learning.
- Case-diagnosis SFT maps multiple images directly to one diagnosis.
- Evidence-to-diagnosis SFT freezes the visual path and trains the language backbone on model-produced evidence.

## Controlled LoRA freezing study

All four LoRA configurations keep the base model, dataset split, epochs, global batch, learning rate, rank, alpha, dropout and seed constant.

| Run | Vision | Projector | Language | Comparison |
|---|---|---|---|---|
| R04 | LoRA | Full, saved with adapter | LoRA | LoRA baseline |
| R05 | Frozen | Full, saved with adapter | LoRA | Effect of freezing vision |
| R06 | LoRA | Frozen | LoRA | Effect of freezing projector |
| R07 | LoRA | Full, saved with adapter | Frozen | Effect of freezing language |

R04 is compared with the corresponding Full-SFT task only for model quality under matched data and optimization settings. Runtime comparisons must report hardware, world size and normalized GPU-hours.

## Evaluation

The public code implements exact and concept-level diagnosis metrics, text-overlap metrics, hallucination/omission proxies, numeric consistency, latency and paired bootstrap comparisons. These are reference-based research proxies and do not replace expert image-grounded review.
