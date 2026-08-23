#!/usr/bin/env python3
"""Audit train/test overlap and label distribution without touching the model."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


DIAG_RE = re.compile(r"病理诊断\s*[：:]\s*(.+)", re.S)


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).strip().lower()


def assistant_text(record: dict) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def diagnosis(text: str) -> str:
    match = DIAG_RE.search(text)
    if not match:
        return "__MISSING__"
    value = match.group(1).splitlines()[0]
    value = re.sub(r"[。；;].*$", "", value)
    return normalize(value) or "__EMPTY__"


def case_key(image_path: str) -> str:
    parts = Path(image_path.replace("\\", "/")).parts
    return "/".join(parts[:-1]) if len(parts) > 1 else image_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.load(open(args.config, "r", encoding="utf-8"))
    train = json.load(open(cfg["train_json"], "r", encoding="utf-8"))
    test = json.load(open(cfg["test_json"], "r", encoding="utf-8"))

    train_reports = [normalize(assistant_text(x)) for x in train]
    test_reports = [normalize(assistant_text(x)) for x in test]
    train_report_set = set(train_reports)
    train_images = {p for x in train for p in x.get("images", [])}
    test_images = {p for x in test for p in x.get("images", [])}
    train_cases = {case_key(p) for p in train_images}
    test_cases = {case_key(p) for p in test_images}

    exact_report_overlap = [i for i, report in enumerate(test_reports) if report in train_report_set]
    trivial_overlap = [
        i for i in exact_report_overlap
        if test_reports[i] in {"病理诊断：未找到", "病理诊断:未找到", "未找到", ""}
    ]
    nontrivial_overlap = [i for i in exact_report_overlap if i not in set(trivial_overlap)]
    image_path_overlap = sorted(train_images & test_images)
    case_path_overlap = sorted(train_cases & test_cases)
    test_report_counts = Counter(test_reports)
    duplicate_test_reports = {k: v for k, v in test_report_counts.items() if v > 1}
    diag_counts = Counter(diagnosis(assistant_text(x)) for x in test)

    result = {
        "train_cases": len(train),
        "test_cases": len(test),
        "train_image_paths": len(train_images),
        "test_image_paths": len(test_images),
        "exact_test_reports_present_in_train_count": len(exact_report_overlap),
        "exact_test_reports_present_in_train_indices": exact_report_overlap,
        "trivial_placeholder_report_overlap_count": len(trivial_overlap),
        "trivial_placeholder_report_overlap_indices": trivial_overlap,
        "nontrivial_exact_report_overlap_count": len(nontrivial_overlap),
        "nontrivial_exact_report_overlap_indices": nontrivial_overlap,
        "exact_image_path_overlap_count": len(image_path_overlap),
        "exact_image_path_overlap": image_path_overlap,
        "case_directory_overlap_count": len(case_path_overlap),
        "case_directory_overlap": case_path_overlap,
        "duplicate_test_report_group_count": len(duplicate_test_reports),
        "duplicate_test_report_case_count": sum(duplicate_test_reports.values()),
        "test_diagnosis_distribution": dict(diag_counts.most_common()),
        "near_duplicate_image_audit": None,
        "near_duplicate_image_audit_na_reason": "perceptual image hashing is not enabled in the base audit",
        "audit_sha256": hashlib.sha256(
            ("\n".join(train_reports) + "\n---TEST---\n" + "\n".join(test_reports)).encode("utf-8")
        ).hexdigest(),
    }
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "dataset_audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
