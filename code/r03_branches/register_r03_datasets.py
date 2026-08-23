#!/usr/bin/env python3
"""Idempotently register R03-01/02 datasets in LLaMA-Factory dataset_info.json."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


ENTRIES = {
    "r03_01_diagnosis_nv1": {
        "file_name": "r03_01_02_artifacts/r03_01_train.json",
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
    },
    "r03_02_evidence_to_diagnosis": {
        "file_name": "r03_01_02_artifacts/r03_02_train_reference_evidence.json",
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-info", required=True, type=Path)
    args = parser.parse_args()
    info = json.loads(args.dataset_info.read_text(encoding="utf-8"))
    changed = False
    for key, value in ENTRIES.items():
        if key in info and info[key] != value:
            raise SystemExit(f"conflicting existing registration: {key}")
        if key not in info:
            info[key] = value
            changed = True
    if changed:
        backup = args.dataset_info.with_name(args.dataset_info.name + ".before_r03_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        shutil.copy2(args.dataset_info, backup)
        temp = args.dataset_info.with_suffix(args.dataset_info.suffix + ".tmp_r03")
        temp.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(args.dataset_info)
        print(json.dumps({"changed": True, "backup": str(backup), "registered": sorted(ENTRIES)}, ensure_ascii=False))
    else:
        print(json.dumps({"changed": False, "registered": sorted(ENTRIES)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
