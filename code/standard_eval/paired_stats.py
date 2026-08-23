#!/usr/bin/env python3
"""Paired significance: McNemar test + Holm step-down correction.

Compares each run's per-case correctness against a baseline run on the SAME
test set (our config forces clean_v2 for all active runs, so case_index aligns).
Default correctness = `diagnosis_exact`; pass --metric-col to use e.g.
`clinical_fact_f1` with --threshold.

McNemar needs the paired disagreement counts (baseline wrong & compared right
vs baseline right & compared wrong). Holm correction is applied across all
pairwise comparisons to control family-wise error.

Output:
  <output_root>/paired_stats.json  (per-comparison p, Holm q, conclusions map)
  <output_root>/paired_stats_report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def chi2_survival(x: float) -> float:
    """Upper-tail p-value of chi-square with 1 df = erfc(sqrt(x)/sqrt(2))."""
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x) / math.sqrt(2.0))


def mcnemar_p(n01: int, n10: int) -> float:
    """n01 = baseline wrong & compared right; n10 = baseline right & compared wrong."""
    total = n01 + n10
    if total == 0:
        return 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / total  # continuity-corrected
    return chi2_survival(stat)


def load_correctness(out_dir: Path, col: str, threshold: float) -> dict[int, int]:
    path = out_dir / "per_case_metrics.csv"
    if not path.is_file():
        raise SystemExit(f"缺少 {path}，请先跑 compute_metrics.py")
    out: dict[int, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            idx = int(row["case_index"])
            val = float(row[col])
            out[idx] = 1 if (val >= threshold) else 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--baseline", default="R04")
    ap.add_argument("--compare", default=None, help="逗号分隔的对比 run，缺省=发现的所有其它 run")
    ap.add_argument("--metric-col", default="diagnosis_exact")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    root = Path(args.output_root)
    discovered = sorted(p.name for p in root.iterdir() if (p / "per_case_metrics.csv").is_file())
    if args.baseline not in discovered:
        raise SystemExit(f"基线 {args.baseline} 未找到 per_case_metrics.csv；已发现：{discovered}")
    compares = [c.strip() for c in args.compare.split(",")] if args.compare else [d for d in discovered if d != args.baseline]
    compares = [c for c in compares if c in discovered]
    if not compares:
        raise SystemExit("没有可对比的 run")

    base = load_correctness(root / args.baseline, args.metric_col, args.threshold)
    comparisons = []
    for c in compares:
        comp = load_correctness(root / c, args.metric_col, args.threshold)
        common = sorted(set(base) & set(comp))
        n01 = sum(1 for i in common if base[i] == 0 and comp[i] == 1)
        n10 = sum(1 for i in common if base[i] == 1 and comp[i] == 0)
        n00 = sum(1 for i in common if base[i] == 0 and comp[i] == 0)
        n11 = sum(1 for i in common if base[i] == 1 and comp[i] == 1)
        p = mcnemar_p(n01, n10)
        delta = (sum(comp[i] for i in common) - sum(base[i] for i in common)) / len(common)
        comparisons.append({
            "compared": c, "baseline": args.baseline, "metric_col": args.metric_col,
            "n_common": len(common), "n01": n01, "n10": n10, "n00": n00, "n11": n11,
            "mcnemar_p": round(p, 6), "delta_correct": round(delta, 6),
        })

    # Holm step-down
    ordered = sorted(comparisons, key=lambda x: x["mcnemar_p"])
    m = len(ordered)
    for rank, item in enumerate(ordered, start=1):
        holm_q = min(1.0, item["mcnemar_p"] * (m - rank + 1))
        item["holm_q"] = round(holm_q, 6)
        item["significant_at_alpha"] = holm_q <= args.alpha
    # enforce monotonic non-decreasing q
    for i in range(1, len(ordered)):
        ordered[i]["holm_q"] = round(max(ordered[i]["holm_q"], ordered[i - 1]["holm_q"]), 6)

    conclusions = {}
    for item in comparisons:
        if item["compared"] == args.baseline:
            conclusions[item["compared"]] = "主基线（无配对对比）"
            continue
        sig = "显著" if item["significant_at_alpha"] else "不显著"
        conclusions[item["compared"]] = (
            f"vs {args.baseline}（{args.metric_col}）：McNemar p={item['mcnemar_p']:.4f}，"
            f"Holm q={item['holm_q']:.4f}（α={args.alpha}）→ {sig}；"
            f"正确率差 Δ={item['delta_correct']:+.4f}（n={item['n_common']}）"
        )

    result = {
        "baseline": args.baseline, "metric_col": args.metric_col, "threshold": args.threshold,
        "alpha": args.alpha, "comparisons": comparisons, "conclusions": conclusions,
    }
    (root / "paired_stats.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 配对显著性（McNemar + Holm）", "",
             f"- 基线：{args.baseline}｜指标列：{args.metric_col}（阈值 {args.threshold}）｜α={args.alpha}", ""]
    lines += ["| 对比 run | n | n01 | n10 | McNemar p | Holm q | 显著 | Δ正确率 |", "|---|---:|---:|---:|---:|---:|:--:|---:|"]
    for item in comparisons:
        lines.append(f"| {item['compared']} | {item['n_common']} | {item['n01']} | {item['n10']} | "
                     f"{item['mcnemar_p']:.4f} | {item['holm_q']:.4f} | "
                     f"{'是' if item['significant_at_alpha'] else '否'} | {item['delta_correct']:+.4f} |")
    (root / "paired_stats_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
