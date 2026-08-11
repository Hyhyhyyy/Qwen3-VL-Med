#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


AGGREGATES = [
    ("诊断严格准确率", "diagnostic.accuracy", True),
    ("诊断概念 Micro-F1", "diagnostic.concept_micro_f1", True),
    ("诊断概念 Macro-F1", "diagnostic.concept_macro_f1_supported", True),
    ("Clinical Fact F1（代理）", "clinical_reference_proxies.clinical_fact_f1", True),
    ("幻觉事实率（代理）", "clinical_reference_proxies.hallucinated_fact_rate_reference_proxy", False),
    ("遗漏率（代理）", "clinical_reference_proxies.omission_rate_reference_proxy", False),
    ("临床不一致率（代理）", "clinical_reference_proxies.clinical_discordance_rate_reference_proxy", False),
    ("ROUGE-L F1", "lexical.mean_rougeL_f1", True),
    ("chrF", "lexical.corpus_chrf", True),
    ("BERTScore F1", "lexical.bertscore_f1", True),
]
PAIRED = [
    ("诊断严格准确率", "diagnosis_exact", True),
    ("逐病例诊断概念 F1", "diagnosis_concept_f1", True),
    ("Clinical Fact F1（代理）", "clinical_fact_f1", True),
    ("幻觉事实率（代理）", "hallucinated_fact_rate_proxy", False),
    ("遗漏率（代理）", "omission_rate_proxy", False),
    ("临床不一致率（代理）", "clinical_discordance_rate_proxy", False),
    ("ROUGE-L F1", "rougeL_f1", True),
    ("chrF", "chrf", True),
    ("BERTScore F1", "bertscore_f1", True),
]


def nested(obj: dict, path: str):
    current = obj
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def load_rows(path: Path) -> dict[int, dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {int(row["case_index"]): row for row in csv.DictReader(handle)}


def finite(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def paired_ci(old_rows, new_rows, column, samples, seed):
    differences = []
    for case_index in sorted(set(old_rows) & set(new_rows)):
        old = finite(old_rows[case_index].get(column))
        new = finite(new_rows[case_index].get(column))
        if old is not None and new is not None:
            differences.append(new - old)
    if not differences:
        return None
    rng = random.Random(seed)
    count = len(differences)
    estimates = sorted(
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    return {
        "n": count,
        "mean_delta": sum(differences) / count,
        "ci95_low": estimates[int(0.025 * (samples - 1))],
        "ci95_high": estimates[int(0.975 * (samples - 1))],
    }


def fmt(value) -> str:
    return "—" if value is None else f"{float(value):.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_metrics = json.loads((args.baseline_dir / "metrics.json").read_text(encoding="utf-8"))
    candidate_metrics = json.loads((args.candidate_dir / "metrics.json").read_text(encoding="utf-8"))
    baseline_rows = load_rows(args.baseline_dir / "per_case_metrics.csv")
    candidate_rows = load_rows(args.candidate_dir / "per_case_metrics.csv")
    common = sorted(set(baseline_rows) & set(candidate_rows))
    if len(common) != len(baseline_rows) or len(common) != len(candidate_rows):
        raise SystemExit("Refusing unpaired comparison: case indices differ")

    result = {
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "paired_case_count": len(common),
        "aggregates": [],
        "paired_bootstrap": [],
    }
    for label, path, higher_is_better in AGGREGATES:
        old = nested(baseline_metrics, path)
        new = nested(candidate_metrics, path)
        delta = None if old is None or new is None else float(new) - float(old)
        result["aggregates"].append({
            "metric": label,
            "path": path,
            "baseline": old,
            "candidate": new,
            "delta": delta,
            "higher_is_better": higher_is_better,
        })
    for offset, (label, column, higher_is_better) in enumerate(PAIRED):
        interval = paired_ci(
            baseline_rows,
            candidate_rows,
            column,
            args.bootstrap_samples,
            args.seed + offset,
        )
        if interval:
            interval.update(
                metric=label,
                column=column,
                higher_is_better=higher_is_better,
                ci_excludes_zero=interval["ci95_low"] > 0 or interval["ci95_high"] < 0,
            )
            result["paired_bootstrap"].append(interval)

    (args.output_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {args.baseline_label} vs {args.candidate_label}",
        "",
        f"- 完全配对病例：{len(common)}。",
        "- 差值方向：candidate - baseline。",
        "- 临床事实、幻觉、遗漏和不一致均为参考驱动规则代理，不等于专家阅片结论。",
        "",
        "## 汇总指标",
        "",
        f"| 指标 | {args.baseline_label} | {args.candidate_label} | 差值 |",
        "|---|---:|---:|---:|",
    ]
    for row in result["aggregates"]:
        lines.append(
            f"| {row['metric']} | {fmt(row['baseline'])} | {fmt(row['candidate'])} | {fmt(row['delta'])} |"
        )
    lines.extend([
        "", "## 病例级配对 Bootstrap", "",
        "| 指标 | n | 平均差值 | 95% CI | CI跨0 |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in result["paired_bootstrap"]:
        crosses = "否" if row["ci_excludes_zero"] else "是"
        lines.append(
            f"| {row['metric']} | {row['n']} | {fmt(row['mean_delta'])} | "
            f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] | {crosses} |"
        )
    (args.output_dir / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"paired_case_count": len(common), "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
