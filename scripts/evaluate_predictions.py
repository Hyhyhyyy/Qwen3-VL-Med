#!/usr/bin/env python3
"""Evaluate LlamaFactory generated_predictions.jsonl without third-party packages."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


PREDICTION_KEYS = ("predict", "prediction", "generated_text", "output")
REFERENCE_KEYS = ("label", "reference", "target", "answer")


def pick(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = record.get(key)
        if value is not None:
            if isinstance(value, list):
                value = value[0] if value else ""
            return str(value)
    return None


def normalize(text: str) -> str:
    return "".join(text.lower().split())


def char_f1(prediction: str, reference: str) -> tuple[float, float, float]:
    pred = normalize(prediction)
    ref = normalize(reference)
    if not pred and not ref:
        return 1.0, 1.0, 1.0
    if not pred or not ref:
        return 0.0, 0.0, 0.0
    common = sum((Counter(pred) & Counter(ref)).values())
    precision = common / len(pred)
    recall = common / len(ref)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def rouge_l(prediction: str, reference: str, limit: int = 512) -> float:
    pred = normalize(prediction)[:limit]
    ref = normalize(reference)[:limit]
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    if len(pred) > len(ref):
        pred, ref = ref, pred
    previous = [0] * (len(pred) + 1)
    for ref_char in ref:
        current = [0]
        for index, pred_char in enumerate(pred, 1):
            if pred_char == ref_char:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def load_records(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSON root is not an object")
                records.append(value)
            except (json.JSONDecodeError, ValueError) as error:
                malformed.append({"line": line_number, "error": type(error).__name__})
    return records, malformed


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.predictions.is_file():
        raise SystemExit(f"预测文件不存在：{args.predictions}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, malformed = load_records(args.predictions)
    rows: list[dict[str, Any]] = []
    missing_prediction = 0
    missing_reference = 0
    for index, record in enumerate(records, 1):
        prediction = pick(record, PREDICTION_KEYS)
        reference = pick(record, REFERENCE_KEYS)
        if prediction is None:
            missing_prediction += 1
            prediction = ""
        if reference is None:
            missing_reference += 1
            reference = ""
        precision, recall, f1 = char_f1(prediction, reference)
        row = {
            "index": index,
            "exact_match": float(normalize(prediction) == normalize(reference)),
            "char_precision": precision,
            "char_recall": recall,
            "char_f1": f1,
            "rouge_l": rouge_l(prediction, reference),
            "prediction_length": len(prediction),
            "reference_length": len(reference),
            "prediction": prediction,
            "reference": reference,
        }
        rows.append(row)

    scored = rows if missing_reference < len(rows) else []
    summary = {
        "prediction_file": args.predictions.name,
        "records": len(records),
        "malformed_lines": len(malformed),
        "missing_predictions": missing_prediction,
        "missing_references": missing_reference,
        "nonempty_prediction_rate": (
            sum(bool(str(row["prediction"]).strip()) for row in rows) / len(rows) if rows else 0.0
        ),
        "exact_match": mean([row["exact_match"] for row in scored]),
        "char_precision": mean([row["char_precision"] for row in scored]),
        "char_recall": mean([row["char_recall"] for row in scored]),
        "char_f1": mean([row["char_f1"] for row in scored]),
        "rouge_l": mean([row["rouge_l"] for row in scored]),
        "median_prediction_length": (
            statistics.median([row["prediction_length"] for row in rows]) if rows else 0
        ),
        "metric_note": "字符级指标用于批量回归比较；ROUGE-L按每段前512个非空白字符计算，不能替代病理专家评审。",
    }
    if any(isinstance(value, float) and not math.isfinite(value) for value in summary.values()):
        raise SystemExit("评估结果包含非有限数值")

    with (args.output_dir / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    with (args.output_dir / "malformed_lines.jsonl").open("w", encoding="utf-8") as stream:
        for row in malformed:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "metrics_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "value"))
        writer.writerows(summary.items())
    markdown = ["# 批量评估摘要", ""]
    for key, value in summary.items():
        markdown.append(f"- {key}: {value}")
    (args.output_dir / "metrics_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if records and not malformed and not missing_prediction else 1


if __name__ == "__main__":
    raise SystemExit(main())
