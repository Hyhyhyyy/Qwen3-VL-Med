#!/usr/bin/env python3
"""Add BERTScore with an independent Chinese encoder to existing metrics."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

from bert_score import score


def load_core(root: Path):
    spec = importlib.util.spec_from_file_location("metric_core", root / "compute_metrics.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = json.load(open(config_path, "r", encoding="utf-8"))
    root = Path(cfg.get("metric_root", config_path.parent)).resolve()
    if not (root / "compute_metrics.py").is_file() and (config_path.parent.parent / "compute_metrics.py").is_file():
        root = config_path.parent.parent
    core = load_core(root)
    core.STRIP_POSITION_TOKENS_FOR_SCORING = bool(cfg.get("strip_position_tokens_for_scoring", False))
    model_type = cfg.get("bert_score_model")
    if not model_type:
        raise SystemExit("bert_score_model is not configured")

    out_dir = Path(cfg["output_dir"])
    prediction_rows = [
        json.loads(line)
        for line in (out_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    references = [core.normalize_text(x["reference"]) for x in prediction_rows]
    predictions = [core.normalize_text(x["prediction"]) for x in prediction_rows]

    precision, recall, f1 = score(
        predictions,
        references,
        model_type=model_type,
        lang="zh",
        batch_size=args.batch_size,
        device="cuda:0",
        verbose=True,
        rescale_with_baseline=False,
    )
    p_values = [float(x) for x in precision.cpu().tolist()]
    r_values = [float(x) for x in recall.cpu().tolist()]
    f_values = [float(x) for x in f1.cpu().tolist()]

    metrics_path = out_dir / "metrics.json"
    metrics = json.load(open(metrics_path, "r", encoding="utf-8"))
    metrics["lexical"].update({
        "bertscore_precision": sum(p_values) / len(p_values),
        "bertscore_recall": sum(r_values) / len(r_values),
        "bertscore_f1": sum(f_values) / len(f_values),
        "bertscore_model": model_type,
        "bertscore_rescale_with_baseline": False,
        "bertscore_domain_warning": "general Chinese encoder, not a pathology-specific encoder",
    })
    samples = int(cfg.get("bootstrap_samples", 1000))
    seed = int(cfg.get("bootstrap_seed", 20260810))
    metrics["confidence_intervals"]["bertscore_precision"] = core.bootstrap_ci(p_values, samples, seed + 101)
    metrics["confidence_intervals"]["bertscore_recall"] = core.bootstrap_ci(r_values, samples, seed + 102)
    metrics["confidence_intervals"]["bertscore_f1"] = core.bootstrap_ci(f_values, samples, seed + 103)
    metrics["metric_availability"]["bert_score"] = {
        "available": True,
        "reason": f"computed with {model_type}; general Chinese rather than pathology-specific encoder",
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metric_availability.json").write_text(
        json.dumps(metrics["metric_availability"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    per_case_path = out_dir / "per_case_metrics.csv"
    with per_case_path.open("r", encoding="utf-8-sig", newline="") as f:
        per_case = list(csv.DictReader(f))
    if len(per_case) != len(f_values):
        raise SystemExit("per-case row count does not match BERTScore output")
    for row, p, r, f in zip(per_case, p_values, r_values, f_values):
        row["bertscore_precision"] = p
        row["bertscore_recall"] = r
        row["bertscore_f1"] = f
    fieldnames = list(per_case[0])
    with per_case_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_case)

    (out_dir / "semantic_metrics.json").write_text(
        json.dumps({
            "model": model_type,
            "precision": metrics["lexical"]["bertscore_precision"],
            "recall": metrics["lexical"]["bertscore_recall"],
            "f1": metrics["lexical"]["bertscore_f1"],
            "domain_warning": metrics["lexical"]["bertscore_domain_warning"],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "metrics_report.md").write_text(core.build_markdown_report(metrics), encoding="utf-8")
    print(json.dumps(metrics["lexical"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
