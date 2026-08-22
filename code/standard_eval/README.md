# Standard evaluation components

This directory contains the reusable, data-agnostic components used by the internal 13-metric evaluation workflow:

- `confidence_teacherforce.py` and `calibration_ece.py`: teacher-forced confidence, ECE and entropy proxy;
- `interpretability_grad_attn.py`: case-level visual gradient and attention shares;
- `benchmark_mmbench.py`: isolated public MMBench evaluation;
- `external_similarity.py`: optional embedding similarity backends;
- `paired_stats.py` and `kfold_cv.py`: paired statistics and split checks;
- `export_13_metrics.py` and `validate_standard_results.py`: one-row export and acceptance gates.

All paths and private data remain external to the repository. The scripts consume explicit command-line inputs and must be run in an authorized environment.

## Interpretability limitation

The current `interpretability_grad_attn.py` exports case-level and per-image aggregate shares. It does not persist the full target-token × visual-token × attention-head matrices required for spatial heatmaps. Follow the next-phase archive contract in [`docs/STANDARD_EVALUATION.md`](../../docs/STANDARD_EVALUATION.md) before claiming token-to-region visualization support.
