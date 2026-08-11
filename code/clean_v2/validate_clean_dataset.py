#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

POSITION = re.compile(r"(?:(?:左|中|右)(?:上|中|下)|上|下)图")
DIAG = re.compile(r"病理诊断\s*[：:]\s*(.+)", re.S)


def target(record):
    return next((str(x.get("content", "")) for x in record.get("messages", []) if x.get("role") == "assistant"), "")


def compact(text):
    return re.sub(r"\s+", "", text).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--image-root", required=True)
    args = ap.parse_args()
    train = json.load(open(args.train, encoding="utf-8"))
    test = json.load(open(args.test, encoding="utf-8"))
    root = Path(args.image_root)
    assert len(train) == 5090
    assert len(test) == 276
    for split, rows in (("train", train), ("test", test)):
        seen = set()
        for i, row in enumerate(rows):
            text = target(row)
            assert not POSITION.search(text), (split, i, POSITION.findall(text), text[:200])
            match = DIAG.search(text)
            assert match and match.group(1).strip(), (split, i, "missing diagnosis")
            assert compact(match.group(1).splitlines()[0]) not in {"未找到", "无法诊断", "不详", "无"}
            images = row.get("images") or []
            assert images and len(images) == len(set(images)), (split, i, "bad images")
            assert sum(1 for x in images if not (root / x).is_file()) == 0, (split, i, "missing image")
            key = compact(text)
            assert key not in seen, (split, i, "duplicate target")
            seen.add(key)
    train_targets = {compact(target(x)) for x in train}
    test_targets = {compact(target(x)) for x in test}
    assert not (train_targets & test_targets), "exact cleaned target leakage remains"
    print(json.dumps({"train": len(train), "test": len(test), "all_checks_passed": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
