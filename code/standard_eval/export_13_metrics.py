#!/usr/bin/env python3
"""Export the 13 metrics required by the project summary table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def nested(data: dict, *keys, default=None):
    value = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-benchmark")
    args = parser.parse_args()
    root = Path(args.output_dir)
    metrics = load(root / "metrics.json")
    calibration = load(root / "calibration.json")
    interpretation = load(root / "interpretability.json")
    benchmark = load(root / "benchmark" / "benchmark.json") or load(root / "benchmark.json")
    base_benchmark = load(Path(args.base_benchmark)) if args.base_benchmark else {}
    external = load(root / "external_similarity.json")
    lexical = metrics.get("lexical", {})
    diagnostic = metrics.get("diagnostic", {})
    ci = metrics.get("confidence_intervals", {})
    ci_block = ci.get("diagnosis_exact") or ci.get("rougeL_f1") or {}
    bleu4 = lexical.get("corpus_bleu4")
    if bleu4 is None:
        bleu4 = lexical.get("mean_sentence_bleu4")
    record = {
        "run": args.run_id,
        "rougeL_f1": lexical.get("mean_rougeL_f1"),
        "bertscore_f1": lexical.get("bertscore_f1"),
        "bleu4": bleu4,
        "meteor": lexical.get("mean_meteor_exact_chinese"),
        "cider": lexical.get("cider_lite"),
        "exact_diagnosis_accuracy": diagnostic.get("accuracy"),
        "bootstrap_95ci_low": ci_block.get("ci95_low"),
        "bootstrap_95ci_high": ci_block.get("ci95_high"),
        "ece": calibration.get("ece"),
        "entropy_proxy": calibration.get("mean_entropy_proxy"),
        "gradient_visual_share": interpretation.get("gradient_visual_share"),
        "attention_visual_mass": interpretation.get("attention_visual_mass"),
        "mmbench_strict_circular_accuracy": benchmark.get("strict_circular_accuracy"),
        "mmbench_vanilla_circ0_accuracy": benchmark.get("vanilla_circ0_accuracy"),
        "alignment_tax_pp_vs_base": (
            100 * (benchmark["strict_circular_accuracy"] - base_benchmark["strict_circular_accuracy"])
            if benchmark.get("strict_circular_accuracy") is not None
            and base_benchmark.get("strict_circular_accuracy") is not None
            else None
        ),
        "biobert_similarity": external.get("biobert_similarity"),
    }
    (root / "metrics_13.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "metrics_13.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
