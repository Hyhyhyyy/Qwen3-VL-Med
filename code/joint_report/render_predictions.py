#!/usr/bin/env python3
"""Validate joint JSON predictions and render them for common text metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REQUIRED_FIELDS = {
    "overall_diagnosis",
    "per_image_findings",
    "case_level_findings",
    "integrated_evidence",
}


def parse_object(text: str):
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I | re.S).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def validate(record, expected_images: int) -> tuple[bool, list[str]]:
    if not isinstance(record, dict):
        return False, ["not_object"]
    errors = []
    if set(record) != REQUIRED_FIELDS:
        errors.append("top_level_fields")
    overall = record.get("overall_diagnosis", {})
    if not isinstance(overall, dict) or not str(overall.get("diagnosis", "")).strip():
        errors.append("diagnosis")
    findings = record.get("per_image_findings", [])
    ids = [item.get("image_id") for item in findings if isinstance(item, dict)] if isinstance(findings, list) else []
    if ids != [f"image_{index}" for index in range(1, expected_images + 1)]:
        errors.append("image_id_coverage_or_order")
    integrated = record.get("integrated_evidence", {})
    if not isinstance(integrated, dict) or not isinstance(integrated.get("cross_image_summary"), str):
        errors.append("cross_image_summary")
    return not errors, errors


def render(record: dict, diagnosis_prefix: str) -> str:
    narrative = record["integrated_evidence"]["cross_image_summary"].strip()
    diagnosis = str(record["overall_diagnosis"]["diagnosis"]).strip()
    return f"{narrative}\n{diagnosis_prefix}{diagnosis}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnosis-prefix", default="Diagnosis:")
    args = parser.parse_args()
    stats = {"rows": 0, "reference_valid": 0, "prediction_valid": 0}
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as destination:
        for line in source:
            row = json.loads(line)
            count = len(row.get("images", []))
            reference = parse_object(row["reference"])
            prediction = parse_object(row["prediction"])
            ref_valid, ref_errors = validate(reference, count)
            pred_valid, pred_errors = validate(prediction, count)
            if not ref_valid:
                raise ValueError(f"invalid reference: {ref_errors}")
            stats["rows"] += 1
            stats["reference_valid"] += 1
            stats["prediction_valid"] += int(pred_valid)
            row.update(
                structure_reference_valid=ref_valid,
                structure_prediction_valid=pred_valid,
                structure_prediction_errors=pred_errors,
                reference=render(reference, args.diagnosis_prefix),
                prediction=render(prediction, args.diagnosis_prefix) if pred_valid else row["prediction"],
            )
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats["reference_valid_rate"] = stats["reference_valid"] / max(stats["rows"], 1)
    stats["prediction_valid_rate"] = stats["prediction_valid"] / max(stats["rows"], 1)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
