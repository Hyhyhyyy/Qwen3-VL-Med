#!/usr/bin/env python3
"""Compare baseline and clean-v2 models on the identical clean test set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


AGGREGATES = [
    ("诊断全文严格准确率", "diagnostic.accuracy", True),
    ("诊断概念 Micro-F1", "diagnostic.concept_micro_f1", True),
    ("诊断概念 Macro-F1（有支持类）", "diagnostic.concept_macro_f1_supported", True),
    ("Clinical Fact F1（代理）", "clinical.clinical_fact_f1", True),
    ("HARE 实体 F1（代理）", "clinical.hare_style_entity_f1_proxy", True),
    ("HARE 关系 F1（代理）", "clinical.hare_style_relation_f1_proxy", True),
    ("关键事实召回率", "clinical.key_fact_recall", True),
    ("幻觉事实率（代理）", "clinical.hallucinated_fact_rate_reference_proxy", False),
    ("遗漏率（代理）", "clinical.omission_rate_reference_proxy", False),
    ("临床不一致率（代理）", "clinical.clinical_discordance_rate_reference_proxy", False),
    ("逐病例否定翻转率", "clinical.negation_flip_rate_per_case", False),
    ("Corpus BLEU-4", "lexical.corpus_bleu4", True),
    ("ROUGE-L F1", "lexical.mean_rougeL_f1", True),
    ("chrF", "lexical.corpus_chrf", True),
    ("BERTScore F1", "lexical.bertscore_f1", True),
]

PAIRED = [
    ("诊断严格准确率", "diagnosis_exact", True),
    ("逐病例诊断概念 F1", "diagnosis_concept_f1", True),
    ("Clinical Fact F1（代理）", "clinical_fact_f1", True),
    ("HARE 实体 F1（代理）", "hare_style_entity_f1_proxy", True),
    ("HARE 关系 F1（代理）", "hare_style_relation_f1_proxy", True),
    ("关键事实召回率", "key_fact_recall", True),
    ("幻觉事实率（代理）", "hallucinated_fact_rate_proxy", False),
    ("遗漏率（代理）", "omission_rate_proxy", False),
    ("临床不一致率（代理）", "clinical_discordance_rate_proxy", False),
    ("ROUGE-L F1", "rougeL_f1", True),
    ("chrF", "chrf", True),
    ("BERTScore F1", "bertscore_f1", True),
]


def nested(obj: dict, path: str):
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def load_rows(path: Path) -> dict[int, dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {int(r["case_index"]): r for r in csv.DictReader(f)}


def finite_number(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def paired_ci(old_rows, new_rows, column, samples, seed):
    diffs = []
    for idx in sorted(set(old_rows) & set(new_rows)):
        old = finite_number(old_rows[idx].get(column))
        new = finite_number(new_rows[idx].get(column))
        if old is not None and new is not None:
            diffs.append(new - old)
    if not diffs:
        return None
    rng = random.Random(seed)
    n = len(diffs)
    boots = []
    for _ in range(samples):
        boots.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    lo = boots[int(0.025 * (samples - 1))]
    hi = boots[int(0.975 * (samples - 1))]
    return {"n": n, "mean_delta": sum(diffs) / n, "ci95_low": lo, "ci95_high": hi}


def fmt(x):
    return "—" if x is None else f"{float(x):.6f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", required=True, type=Path)
    p.add_argument("--clean-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--bootstrap-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260810)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    old = json.loads((args.baseline_dir / "metrics.json").read_text(encoding="utf-8"))
    new = json.loads((args.clean_dir / "metrics.json").read_text(encoding="utf-8"))
    old_rows = load_rows(args.baseline_dir / "per_case_metrics.csv")
    new_rows = load_rows(args.clean_dir / "per_case_metrics.csv")
    common = sorted(set(old_rows) & set(new_rows))
    if len(common) != len(old_rows) or len(common) != len(new_rows):
        raise SystemExit("Refusing unpaired comparison: case indices differ")

    result = {"paired_case_count": len(common), "aggregates": [], "paired_bootstrap": []}
    for label, path, higher_better in AGGREGATES:
        ov, nv = nested(old, path), nested(new, path)
        delta = None if ov is None or nv is None else float(nv) - float(ov)
        result["aggregates"].append({
            "metric": label, "path": path, "baseline": ov, "clean_v2": nv,
            "delta": delta, "higher_is_better": higher_better,
            "improved": None if delta is None else (delta > 0 if higher_better else delta < 0),
        })
    for offset, (label, column, higher_better) in enumerate(PAIRED):
        ci = paired_ci(old_rows, new_rows, column, args.bootstrap_samples, args.seed + offset)
        if ci:
            ci.update(metric=label, column=column, higher_is_better=higher_better,
                      improved=ci["mean_delta"] > 0 if higher_better else ci["mean_delta"] < 0,
                      ci_excludes_zero=ci["ci95_low"] > 0 or ci["ci95_high"] < 0)
            result["paired_bootstrap"].append(ci)

    (args.output_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Qwen3-VL 原始全量微调 vs clean_v2 全量微调",
        "",
        f"- 公平性：两个模型均在完全相同的 {len(common)} 条 clean_v2 测试病例上评分。",
        "- 位置词处理：评分前统一移除‘左上图/左中图’等位置元数据。",
        "- 安全声明：临床事实、幻觉、遗漏和不一致均为参考答案驱动的规则代理指标，不等同于病理专家阅片结论。",
        "",
        "## 汇总指标",
        "",
        "| 指标 | 原始模型 | clean_v2 | 差值（新-旧） | 方向 |",
        "|---|---:|---:|---:|---|",
    ]
    for x in result["aggregates"]:
        direction = "↑ 越高越好" if x["higher_is_better"] else "↓ 越低越好"
        lines.append(f"| {x['metric']} | {fmt(x['baseline'])} | {fmt(x['clean_v2'])} | {fmt(x['delta'])} | {direction} |")
    lines += [
        "", "## 配对病例 Bootstrap（95% CI）", "",
        "| 指标 | 有效病例 | 平均差值 | 95% CI | CI 是否跨 0 |",
        "|---|---:|---:|---:|---|",
    ]
    for x in result["paired_bootstrap"]:
        cross = "否" if x["ci_excludes_zero"] else "是"
        lines.append(f"| {x['metric']} | {x['n']} | {fmt(x['mean_delta'])} | [{fmt(x['ci95_low'])}, {fmt(x['ci95_high'])}] | {cross} |")
    (args.output_dir / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"paired_case_count": len(common), "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
