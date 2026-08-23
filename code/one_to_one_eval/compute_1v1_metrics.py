#!/usr/bin/env python3
"""Compute view-level morphology metrics for R03 predictions.

All clinical fact scores are reference-driven rule proxies.  The cross-view
supplementation score is also a proxy: it flags predicted facts absent from the
current view reference but present in another expert view from the same case.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


POSITION_RE = re.compile(r"左上图|右上图|左中图|右中图|左下图|右下图|上图|下图")


def load_metric_helpers(metric_root: Path):
    sys.path.insert(0, str(metric_root))
    from compute_metrics import (  # type: ignore
        bootstrap_ci, char_tokens, contradiction_stats, fact_set, rouge_l, set_prf,
    )
    return bootstrap_ci, char_tokens, contradiction_stats, fact_set, rouge_l, set_prf


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def mean(rows: list[dict], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--metric-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
    args = parser.parse_args()
    bootstrap_ci, char_tokens, contradiction_stats, fact_set, rouge_l, set_prf = load_metric_helpers(args.metric_root)

    predictions = read_jsonl(args.predictions)
    audit_all = read_jsonl(args.audit)
    audit = {int(x["one_to_one_index"]): x for x in audit_all if x.get("split") == "test"}
    indexed = {int(x["case_index"]): x for x in predictions}
    expected = set(audit)
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))[:20]
        extra = sorted(set(indexed) - expected)[:20]
        raise SystemExit(f"prediction/audit index mismatch: missing={missing} extra={extra}")

    case_reference_facts: dict[int, dict[int, set[str]]] = defaultdict(dict)
    for idx, row in indexed.items():
        case_id = int(audit[idx]["clean_case_index"])
        case_reference_facts[case_id][idx] = fact_set(str(row.get("reference") or ""))

    per_view = []
    for idx in sorted(indexed):
        row, meta = indexed[idx], audit[idx]
        if row.get("error") or not str(row.get("prediction") or "").strip():
            raise SystemExit(f"invalid prediction at view {idx}: {row.get('error')}")
        if list(row.get("images") or []) != [meta["image"]]:
            raise SystemExit(f"image/audit mismatch at view {idx}")
        reference = str(row.get("reference") or "")
        prediction = str(row.get("prediction") or "")
        ref_facts, pred_facts = fact_set(reference), fact_set(prediction)
        fact_p, fact_r, fact_f1 = set_prf(ref_facts, pred_facts)
        contradiction, comparable, _ = contradiction_stats(reference, prediction)
        discordance = contradiction / comparable if comparable else 0.0
        rl = rouge_l(char_tokens(reference), char_tokens(prediction))[2]
        case_id = int(meta["clean_case_index"])
        other_facts = set().union(*(facts for other, facts in case_reference_facts[case_id].items() if other != idx))
        other_only = (pred_facts - ref_facts) & other_facts
        per_view.append({
            "view_index": idx,
            "clean_case_index": case_id,
            "source_index": int(meta["source_index"]),
            "image": meta["image"],
            "char_rougeL_f1": rl,
            "clinical_fact_precision_proxy": fact_p,
            "clinical_fact_recall_proxy": fact_r,
            "clinical_fact_f1_proxy": fact_f1,
            "hallucinated_fact_rate_proxy": 1.0 - fact_p,
            "omission_rate_proxy": 1.0 - fact_r,
            "clinical_discordance_rate_proxy": discordance,
            "cross_view_supplementation_rate_proxy": len(other_only) / len(pred_facts) if pred_facts else 0.0,
            "output_chars": len(prediction),
            "schema_ok": prediction.startswith("图像分析："),
            "position_token_present": bool(POSITION_RE.search(prediction)),
            "case_diagnosis_phrase_present": "病理诊断" in prediction,
            "hit_max_new_tokens": bool(row.get("hit_max_new_tokens")),
        })

    main_keys = [
        "char_rougeL_f1", "clinical_fact_f1_proxy", "hallucinated_fact_rate_proxy",
        "omission_rate_proxy", "clinical_discordance_rate_proxy",
        "cross_view_supplementation_rate_proxy",
    ]
    metrics = {
        "metadata": {
            "view_count": len(per_view),
            "case_count": len({x["clean_case_index"] for x in per_view}),
            "warning": "Clinical and cross-view metrics are reference-driven rule proxies, not expert image-grounded endpoints.",
        },
        "view_level": {key: mean(per_view, key) for key in main_keys},
        "structure": {
            "schema_rate": statistics.fmean(float(x["schema_ok"]) for x in per_view),
            "position_token_rate": statistics.fmean(float(x["position_token_present"]) for x in per_view),
            "case_diagnosis_phrase_rate": statistics.fmean(float(x["case_diagnosis_phrase_present"]) for x in per_view),
            "hit_max_new_tokens_rate": statistics.fmean(float(x["hit_max_new_tokens"]) for x in per_view),
            "mean_output_chars": statistics.fmean(float(x["output_chars"]) for x in per_view),
        },
        "confidence_intervals": {
            key: bootstrap_ci([float(x[key]) for x in per_view], args.bootstrap_samples, args.bootstrap_seed + i)
            for i, key in enumerate(main_keys)
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics_1v1.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "per_view_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_view[0]))
        writer.writeheader()
        writer.writerows(per_view)
    lines = [
        "# R03 单图形态评测", "", f"- 单图数：{len(per_view)}", f"- 病例数：{metrics['metadata']['case_count']}",
        "- 临床事实与跨视野补写指标均为参考答案规则代理，不是专家图像落地结论。", "",
        "| 指标 | 数值 |", "|---|---:|",
    ]
    for key, value in metrics["view_level"].items():
        lines.append(f"| {key} | {value:.6f} |")
    for key, value in metrics["structure"].items():
        lines.append(f"| {key} | {value:.6f} |")
    (args.output_dir / "metrics_1v1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics["metadata"], ensure_ascii=False))


if __name__ == "__main__":
    main()
