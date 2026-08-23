# Standard evaluation components

This directory contains the reusable, data-agnostic components used by the internal 13-metric evaluation workflow:

- `confidence_teacherforce.py` and `calibration_ece.py`: teacher-forced confidence, ECE and entropy proxy;
- `interpretability_grad_attn.py`: case-level visual gradient and attention shares;
- `archive_interpretability.py`: full float16 token-to-visual arrays, per-head matrices, image-space token mapping, manifests and basic heatmaps;
- `benchmark_mmbench.py`: isolated public MMBench evaluation;
- `external_similarity.py`: optional embedding similarity backends;
- `paired_stats.py` and `kfold_cv.py`: paired statistics and split checks;
- `export_13_metrics.py` and `validate_standard_results.py`: one-row export and acceptance gates.

All paths and private data remain external to the repository. The scripts consume explicit command-line inputs and must be run in an authorized environment.

## Interpretability limitation

`interpretability_grad_attn.py` remains the compatibility path for the historical 13-metric table. A Phase 4 model-side extractor must pass complete tensors to `archive_interpretability.py` before aggregation; old scalar summaries cannot reconstruct or substitute for raw matrices. Follow [`docs/INTERPRETABILITY_ARCHIVE.md`](../../docs/INTERPRETABILITY_ARCHIVE.md) before claiming token-to-region visualization support.
