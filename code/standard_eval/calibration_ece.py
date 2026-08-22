#!/usr/bin/env python3
"""Confidence calibration (ECE) + entropy-based confidence proxy.

Reads predictions.jsonl (needs `mean_token_confidence`, produced by
generate_predictions.py --record-confidence) and per_case_metrics.csv
(diagnosis_exact correctness). Bins predictions by confidence and computes
Expected Calibration Error (ECE) against diagnosis-exact correctness.

HONEST CAVEAT: free-text generation has no defined per-diagnosis probability.
This is a *proxy*: confidence = mean max-softmax over generated tokens;
correctness = normalized exact diagnosis match. Use for ranking runs, not as a
clinically calibrated probability.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    cfg = json.load(open(args.config, "r", encoding="utf-8"))
    out_dir = Path(cfg["output_dir"])
    pred_path = out_dir / "predictions.jsonl"
    per_case_path = out_dir / "per_case_metrics.csv"

    if not pred_path.is_file():
        raise SystemExit(f"缺少 {pred_path}，请先跑 generate_predictions.py")
    if not per_case_path.is_file():
        raise SystemExit(f"缺少 {per_case_path}，请先跑 compute_metrics.py")

    rows = [json.loads(l) for l in pred_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    has_conf = any(isinstance(r.get("mean_token_confidence"), (int, float)) for r in rows)
    if not has_conf:
        result = {
            "available": False,
            "reason": "predictions.jsonl 不含 mean_token_confidence；请重跑 generate_predictions.py 加 --record-confidence",
            "ece": None, "mean_confidence": None, "mean_entropy_proxy": None, "reliability": [],
        }
        (out_dir / "calibration.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    conf_by_idx = {int(r["case_index"]): float(r["mean_token_confidence"]) for r in rows}

    import csv
    correct_by_idx = {}
    with per_case_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            correct_by_idx[int(row["case_index"])] = float(row["diagnosis_exact"])

    pairs = [(conf_by_idx[i], correct_by_idx[i]) for i in conf_by_idx if i in correct_by_idx]
    if not pairs:
        raise SystemExit("predictions 与 per_case_metrics 的 case_index 无法对齐")
    confs = [p[0] for p in pairs]
    corrects = [p[1] for p in pairs]
    n = len(pairs)
    bins = args.bins
    edges = [i / bins for i in range(bins + 1)]
    reliability = []
    ece = 0.0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        sel = [
            k for k in range(n)
            if (lo <= confs[k] < hi) or (b == bins - 1 and lo <= confs[k] <= hi)
        ]
        if not sel:
            continue
        bin_conf = statistics.fmean(confs[k] for k in sel)
        bin_acc = statistics.fmean(corrects[k] for k in sel)
        frac = len(sel) / n
        ece += frac * abs(bin_conf - bin_acc)
        reliability.append({
            "bin_low": lo, "bin_high": hi,
            "confidence": bin_conf, "accuracy": bin_acc, "count": len(sel),
        })
    mean_conf = statistics.fmean(confs)
    # entropy proxy: treat confidence as p, entropy = -p log p - (1-p) log(1-p) per case, averaged
    import math
    entropies = []
    for c in confs:
        c = min(max(c, 1e-6), 1 - 1e-6)
        entropies.append(-(c * math.log(c) + (1 - c) * math.log(1 - c)) / math.log(2))
    result = {
        "available": True,
        "caveat": "proxy: confidence=mean max-softmax over generated tokens; correctness=diagnosis exact match",
        "ece": round(ece, 6),
        "mean_confidence": round(mean_conf, 6),
        "mean_entropy_proxy": round(statistics.fmean(entropies), 6),
        "n_cases": n,
        "reliability": [
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in item.items()}
            for item in reliability
        ],
    }
    (out_dir / "calibration.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
