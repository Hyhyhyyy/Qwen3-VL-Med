#!/usr/bin/env python3
"""Fail-fast acceptance checks for a completed standard evaluation run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    if not path.is_file():
        raise AssertionError(f"missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--require-benchmark", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_dir)

    result = load(root / "metrics_13.json")
    required = [
        "rougeL_f1", "bertscore_f1", "bleu4", "meteor", "cider",
        "exact_diagnosis_accuracy", "bootstrap_95ci_low", "bootstrap_95ci_high",
        "ece", "entropy_proxy", "gradient_visual_share", "attention_visual_mass",
        "biobert_similarity",
    ]
    missing = [key for key in required if result.get(key) is None]
    assert not missing, f"metrics_13 has unavailable values: {missing}"
    assert 0 <= result["bootstrap_95ci_low"] <= result["bootstrap_95ci_high"] <= 1

    calibration = load(root / "calibration.json")
    assert calibration.get("n_errors") == 0, "calibration contains errors"
    assert calibration.get("n_cases", 0) > 0, "calibration has no valid cases"
    assert sum(item["count"] for item in calibration.get("reliability", [])) == calibration["n_cases"]

    interpretation = load(root / "interpretability.json")
    assert interpretation.get("n_errors") == 0, "interpretability contains errors"
    assert interpretation.get("n_cases", 0) > 0, "interpretability has no valid cases"

    if args.require_benchmark:
        benchmark = load(root / "benchmark" / "benchmark.json")
        assert benchmark.get("n_errors") == 0, "benchmark contains errors"
        assert benchmark.get("parse_rate") == 1.0, "benchmark parse rate is not 100%"
        assert benchmark.get("n_rows") == 4876, "unexpected MMBench row count"
        assert benchmark.get("n_circular_groups") == 1292, "unexpected circular group count"
        assert result.get("mmbench_strict_circular_accuracy") is not None

    print("PASS: standard evaluation acceptance checks")


if __name__ == "__main__":
    main()
