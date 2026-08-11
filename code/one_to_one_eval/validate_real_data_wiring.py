#!/usr/bin/env python3
"""Dry-run R03 evaluation wiring with references as structural mock outputs.

This validates indices, images, grouping and output schemas only.  The resulting
oracle-like metrics are temporary and must never be reported as model results.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def assistant(record: dict) -> str:
    return next(str(x.get("content", "")) for x in record["messages"] if x.get("role") == "assistant")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--metric-root", required=True, type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    records = json.loads(args.test.read_text(encoding="utf-8"))
    audit = [json.loads(x) for x in args.audit.read_text(encoding="utf-8").splitlines() if x.strip()]
    test_audit = [x for x in audit if x.get("split") == "test"]
    indices = [int(x["one_to_one_index"]) for x in test_audit]
    if indices != list(range(len(records))):
        raise SystemExit("test audit indices are not contiguous/aligned")
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        prediction_path = tmp / "predictions.jsonl"
        with prediction_path.open("w", encoding="utf-8") as handle:
            for idx, record in enumerate(records):
                reference = assistant(record)
                handle.write(json.dumps({
                    "case_index": idx, "images": record.get("images", []), "reference": reference,
                    "prediction": reference, "error": None, "hit_max_new_tokens": False,
                }, ensure_ascii=False) + "\n")
        metric_out = tmp / "view_metrics"
        subprocess.run([
            sys.executable, str(here / "compute_1v1_metrics.py"), "--predictions", str(prediction_path),
            "--audit", str(args.audit), "--metric-root", str(args.metric_root), "--output-dir", str(metric_out),
            "--bootstrap-samples", "20",
        ], check=True)
        case_path = tmp / "case_inputs.json"
        subprocess.run([
            sys.executable, str(here / "build_case_aggregation_inputs.py"), "--predictions", str(prediction_path),
            "--audit", str(args.audit), "--output", str(case_path),
        ], check=True)
        metrics = json.loads((metric_out / "metrics_1v1.json").read_text(encoding="utf-8"))
        cases = json.loads(case_path.read_text(encoding="utf-8"))
        if metrics["metadata"]["view_count"] != 709 or len(cases) != 253:
            raise SystemExit(f"unexpected real-data wiring counts: {metrics['metadata']} cases={len(cases)}")
        if sum(len(x["view_indices"]) for x in cases) != 709:
            raise SystemExit("case aggregation lost or duplicated views")
        print(json.dumps({
            "status": "PASS", "structural_mock_only": True,
            "view_count": 709, "case_count": 253, "all_views_grouped_once": True,
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
