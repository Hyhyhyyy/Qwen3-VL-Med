#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


POSITIONS = re.compile(r"左上图|右上图|左中图|右中图|左下图|右下图|上图|下图")
PUNCT = re.compile(r"[\s，。；;：:、\\（）()【】\[\]<>《》“”\"'—-]+")


def assistant(record):
    return next((str(x.get("content", "")) for x in record.get("messages", [])
                 if x.get("role") == "assistant"), "")


def user(record):
    return next((str(x.get("content", "")) for x in record.get("messages", [])
                 if x.get("role") == "user"), "")


def canonical(text):
    return PUNCT.sub("", text).lower()


def inspect(path: Path, image_root: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    errors, images, targets = [], [], []
    for i, row in enumerate(rows):
        ims = row.get("images") or []
        prompt, target = user(row), assistant(row)
        if len(ims) != 1:
            errors.append([i, "image_count_not_one"])
        if prompt.count("<image>") != 1:
            errors.append([i, "placeholder_count_not_one"])
        if not target.startswith("图像分析："):
            errors.append([i, "invalid_target_schema"])
        if "病理诊断" in target:
            errors.append([i, "case_diagnosis_copied_to_single_view"])
        if POSITIONS.search(target):
            errors.append([i, "position_token_in_target"])
        for raw in ims:
            p = Path(raw)
            if not p.is_absolute():
                p = image_root / p
            if not p.is_file():
                errors.append([i, "missing_image"])
            images.append(raw)
        targets.append(canonical(target))
    if len(images) != len(set(images)):
        errors.append([-1, "duplicate_image_within_split"])
    if len(targets) != len(set(targets)):
        errors.append([-1, "duplicate_target_within_split"])
    return rows, set(images), set(targets), errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--test", required=True, type=Path)
    p.add_argument("--image-root", required=True, type=Path)
    args = p.parse_args()
    train, train_images, train_targets, train_errors = inspect(args.train, args.image_root)
    test, test_images, test_targets, test_errors = inspect(args.test, args.image_root)
    errors = train_errors + test_errors
    if train_images & test_images:
        errors.append([-1, "image_leakage_train_to_test"])
    if train_targets & test_targets:
        errors.append([-1, "exact_target_leakage_train_to_test"])
    result = {
        "train_samples": len(train), "test_samples": len(test),
        "train_unique_images": len(train_images), "test_unique_images": len(test_images),
        "train_unique_targets": len(train_targets), "test_unique_targets": len(test_targets),
        "all_checks_passed": not errors, "errors": errors[:30],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
