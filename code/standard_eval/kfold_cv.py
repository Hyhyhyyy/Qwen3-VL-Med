#!/usr/bin/env python3
"""k-fold evaluation stability (variance under test-set partitioning).

Partitions the per-case metric table into k folds by case_index and recomputes
the mean of each numeric metric per fold, then reports mean ± std across folds.
This estimates how stable each metric is under different test partitions — it is
NOT model-training CV (no retraining), and it lets you fill the sheet's
"k-fold CV" column with a concrete variability number instead of "需补".

Note: metrics that are themselves case-means (rougeL_f1, diagnosis_exact,
clinical_fact_f1, ...) are exactly the right quantities to summarize this way.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260810)
    args = ap.parse_args()
    cfg = json.load(open(args.config, "r", encoding="utf-8"))
    out_dir = Path(cfg["output_dir"])
    per_case = out_dir / "per_case_metrics.csv"
    if not per_case.is_file():
        raise SystemExit(f"缺少 {per_case}，请先跑 compute_metrics.py")

    with per_case.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)

    # Keep only fully-numeric columns
    numeric_cols = []
    for col in fields:
        if col in ("case_index",):
            continue
        vals = []
        ok = True
        for r in rows:
            v = r.get(col, "")
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                ok = False
                break
        if ok and vals:
            numeric_cols.append(col)

    import random
    rng = random.Random(args.seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    folds = [indices[i::args.k] for i in range(args.k)]

    per_metric = {}
    for col in numeric_cols:
        fold_means = []
        for f in folds:
            if not f:
                continue
            vals = [float(rows[i][col]) for i in f]
            fold_means.append(statistics.fmean(vals))
        if len(fold_means) >= 2:
            per_metric[col] = {
                "mean": round(statistics.fmean(fold_means), 6),
                "std": round(statistics.pstdev(fold_means), 6),
                "folds": [round(x, 6) for x in fold_means],
            }
        else:
            per_metric[col] = {"mean": round(fold_means[0], 6) if fold_means else None, "std": 0.0, "folds": fold_means}

    result = {"k": args.k, "n_cases": len(rows), "seed": args.seed, "per_metric": per_metric}
    (out_dir / "kfold_cv.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # Compact representative lines
    rep_cols = ["rougeL_f1", "diagnosis_exact", "clinical_fact_f1", "crqs_style_proxy", "key_fact_recall"]
    lines = [f"# k-fold CV (k={args.k}, n={len(rows)})", ""]
    lines += ["| 指标 | 跨折均值 | 跨折 std |", "|---|---:|---:|"]
    for col in rep_cols:
        if col in per_metric:
            m = per_metric[col]["mean"]
            s = per_metric[col]["std"]
            lines.append(f"| {col} | {m:.4f} | {s:.4f} |")
    (out_dir / "kfold_cv.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
