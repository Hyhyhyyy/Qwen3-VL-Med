#!/usr/bin/env python3
"""Build leakage-safe case aggregation inputs from R03 view predictions."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    predictions = {int(x["case_index"]): x for x in read_jsonl(args.predictions)}
    audit = [x for x in read_jsonl(args.audit) if x.get("split") == "test"]
    groups: dict[int, list[dict]] = defaultdict(list)
    for meta in audit:
        idx = int(meta["one_to_one_index"])
        row = predictions.get(idx)
        if row is None or row.get("error") or not str(row.get("prediction") or "").strip():
            raise SystemExit(f"missing or invalid view prediction: {idx}")
        if list(row.get("images") or []) != [meta["image"]]:
            raise SystemExit(f"image/audit mismatch: {idx}")
        groups[int(meta["clean_case_index"])].append({"meta": meta, "prediction": row["prediction"]})

    records = []
    for ordinal, clean_case_index in enumerate(sorted(groups)):
        views = sorted(groups[clean_case_index], key=lambda x: int(x["meta"]["one_to_one_index"]))
        references = {str(x["meta"]["case_level_reference_diagnosis_not_used_as_target"]).strip() for x in views}
        source_indices = {int(x["meta"]["source_index"]) for x in views}
        if len(references) != 1 or len(source_indices) != 1:
            raise SystemExit(f"inconsistent audit metadata for case {clean_case_index}")
        evidence = "\n".join(f"证据{n}：{x['prediction']}" for n, x in enumerate(views, 1))
        prompt = (
            "你是一名严谨的肝脏病理医师。以下是同一病例不同视野的逐图形态证据。"
            "本任务是病例级诊断分类，不是形态复述。请推断最可能的疾病或病理过程；"
            "仅输出一行“病理诊断：疾病/病理过程（必要时含分期）”。"
            "禁止输出汇管区、小叶、肝细胞等形态描述段落；证据不足时输出“病理诊断：不确定”。"
            "不得使用文件名、位置词或未提供的事实。\n" + evidence
        )
        reference = next(iter(references))
        records.append({
            "case_aggregation_index": ordinal,
            "clean_case_index": clean_case_index,
            "source_index": next(iter(source_indices)),
            "view_indices": [int(x["meta"]["one_to_one_index"]) for x in views],
            "images": [],
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "病理诊断：" + reference},
            ],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"case_count": len(records), "view_count": len(audit)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
