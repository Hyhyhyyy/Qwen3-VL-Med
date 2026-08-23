#!/usr/bin/env python3
"""Build deterministic training-side cases for aggregation model smoke tests."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def assistant(record: dict) -> str:
    return next(str(x.get("content", "")) for x in record.get("messages", []) if x.get("role") == "assistant")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", type=int, default=32)
    args = parser.parse_args()
    train = json.loads(args.train.read_text(encoding="utf-8"))
    by_image = {str((x.get("images") or [""])[0]): x for x in train}
    audit = [json.loads(x) for x in args.audit.read_text(encoding="utf-8").splitlines() if x.strip()]
    groups: dict[int, list[dict]] = defaultdict(list)
    for meta in audit:
        if meta.get("split") != "train" or meta.get("image") not in by_image:
            continue
        groups[int(meta["clean_case_index"])].append({"meta": meta, "reference_view": assistant(by_image[meta["image"]])})
    records = []
    for case_id in sorted(groups):
        views = groups[case_id]
        if len(views) < 2:
            continue
        diagnoses = {str(x["meta"]["case_level_reference_diagnosis_not_used_as_target"]).strip() for x in views}
        sources = {int(x["meta"]["source_index"]) for x in views}
        if len(diagnoses) != 1 or len(sources) != 1:
            continue
        evidence = "\n".join(f"证据{i}：{x['reference_view']}" for i, x in enumerate(views, 1))
        prompt = (
            "你是一名严谨的肝脏病理医师。以下是同一病例不同视野的逐图形态证据。"
            "本任务是病例级诊断分类，不是形态复述。请推断最可能的疾病或病理过程；"
            "仅输出一行“病理诊断：疾病/病理过程（必要时含分期）”。"
            "禁止输出汇管区、小叶、肝细胞等形态描述段落；证据不足时输出“病理诊断：不确定”。"
            "不得使用文件名、位置词或未提供的事实。\n" + evidence
        )
        records.append({
            "case_aggregation_index": len(records), "clean_case_index": case_id,
            "source_index": next(iter(sources)), "view_indices": [], "images": [],
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "病理诊断：" + next(iter(diagnoses))},
            ],
        })
        if len(records) >= args.cases:
            break
    if len(records) != args.cases:
        raise SystemExit(f"requested {args.cases} cases, built {len(records)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "cases": len(records), "source": "train_only"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
