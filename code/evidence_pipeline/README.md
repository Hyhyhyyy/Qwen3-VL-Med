# Evidence-robust diagnosis data

`build_robust_evidence_mix.py` constructs a controlled 1:1 mixture of Oracle evidence and archived upstream-model evidence for the same training case. Both inputs share the same diagnosis target.

The case manifest must explicitly contain split, stable case ID, Oracle array index, upstream sample indices and corresponding image paths. The builder rejects non-contiguous Oracle linkage, failed predictions, image mismatches and incomplete upstream coverage.

```bash
python code/evidence_pipeline/build_robust_evidence_mix.py \
  --predictions /private/run/upstream_train_predictions.jsonl \
  --case-manifest /private/data/case_manifest.jsonl \
  --oracle-train /private/data/oracle_train.json \
  --output-dir /private/data/robust_mix \
  --expected-cases EXPECTED_PRIVATE_CASE_COUNT
```

All arguments point outside the repository. Do not commit generated mix, audit, manifest, predictions or case linkage. Keep the held-out test split untouched. When the mixture doubles records per case, adjust epochs so optimization exposure remains comparable to the Oracle-only baseline.
