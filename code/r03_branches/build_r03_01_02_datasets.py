#!/usr/bin/env python3
"""Build diagnosis-only nV1 (R03-01) and view-evidence association (R03-02)."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


DIAG_RE = re.compile(r"病理诊断\s*[：:]\s*(.+?)\s*$", re.S)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def assistant(record: dict) -> str:
    return next(str(x.get("content", "")) for x in record.get("messages", []) if x.get("role") == "assistant")


def diagnosis(text: str) -> str:
    match = DIAG_RE.search(text.replace("\r\n", "\n"))
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, rows) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def build_direct(records: list[dict], split: str) -> tuple[list[dict], list[dict]]:
    output, errors = [], []
    for i, row in enumerate(records):
        images = list(row.get("images") or [])
        diag = diagnosis(assistant(row))
        if not images or not diag:
            errors.append({"split": split, "index": i, "image_count": len(images), "diagnosis": diag})
            continue
        prompt = "".join("<image>" for _ in images) + (
            "你是一名严谨的肝脏病理医师。请综合同一病例的全部病理图像，只输出一行"
            "“病理诊断：疾病/病理过程（必要时含分级分期）”。不要输出图像分析过程；"
            "证据不足时输出“病理诊断：不确定”。"
        )
        output.append({
            "images": images,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "病理诊断：" + diag},
            ],
        })
    return output, errors


def build_association(one_rows: list[dict], audit_rows: list[dict], split: str):
    by_image = {str((x.get("images") or [""])[0]): x for x in one_rows}
    groups: dict[int, list[dict]] = defaultdict(list)
    for meta in audit_rows:
        image = str(meta.get("image", ""))
        if meta.get("split") == split and image in by_image:
            groups[int(meta["clean_case_index"])].append({"meta": meta, "view": assistant(by_image[image])})
    output, audit_out, errors = [], [], []
    for case_id in sorted(groups):
        views = groups[case_id]
        diagnoses = {str(x["meta"].get("case_level_reference_diagnosis_not_used_as_target", "")).strip() for x in views}
        sources = {int(x["meta"]["source_index"]) for x in views}
        if len(diagnoses) != 1 or len(sources) != 1 or not next(iter(diagnoses), ""):
            errors.append({"split": split, "clean_case_index": case_id, "diagnoses": sorted(diagnoses), "sources": sorted(sources)})
            continue
        views = sorted(views, key=lambda x: str(x["meta"]["image"]))
        evidence = "\n".join(f"证据{i}：{x['view']}" for i, x in enumerate(views, 1))
        prompt = (
            "你是一名严谨的肝脏病理医师。以下是同一病例不同视野的逐图形态证据。"
            "本任务是病理诊断，不是形态复述。仅输出一行“病理诊断：疾病/病理过程（必要时含分级分期）”。"
            "禁止输出形态描述段落；证据不足时输出“病理诊断：不确定”。\n" + evidence
        )
        diag = next(iter(diagnoses))
        output.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "病理诊断：" + diag},
            ],
        })
        audit_out.append({
            "split": split, "association_index": len(output) - 1, "clean_case_index": case_id,
            "source_index": next(iter(sources)), "view_count": len(views),
            "images": [x["meta"]["image"] for x in views],
        })
    return output, audit_out, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-train", required=True, type=Path)
    parser.add_argument("--clean-test", required=True, type=Path)
    parser.add_argument("--one-train", required=True, type=Path)
    parser.add_argument("--one-test", required=True, type=Path)
    parser.add_argument("--one-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_train, clean_test = load(args.clean_train), load(args.clean_test)
    one_train, one_test, audit = load(args.one_train), load(args.one_test), jsonl(args.one_audit)
    direct_train, dte = build_direct(clean_train, "train")
    direct_test, dse = build_direct(clean_test, "test")
    assoc_train, assoc_train_audit, ate = build_association(one_train, audit, "train")
    assoc_test_ref, assoc_test_audit, ase = build_association(one_test, audit, "test")
    if dte or dse or ate or ase:
        raise SystemExit(json.dumps({"direct_train": dte[:5], "direct_test": dse[:5], "assoc_train": ate[:5], "assoc_test": ase[:5]}, ensure_ascii=False))
    train_images = {x for row in direct_train for x in row["images"]}
    test_images = {x for row in direct_test for x in row["images"]}
    if train_images & test_images:
        raise SystemExit("direct branch image leakage")
    assoc_train_images = {x for row in assoc_train_audit for x in row["images"]}
    assoc_test_images = {x for row in assoc_test_audit for x in row["images"]}
    if assoc_train_images & assoc_test_images:
        raise SystemExit("association branch image leakage")
    files = {
        "r03_01_train.json": direct_train,
        "r03_01_test.json": direct_test,
        "r03_02_train_reference_evidence.json": assoc_train,
        "r03_02_test_reference_evidence.json": assoc_test_ref,
        "r03_02_train_audit.json": assoc_train_audit,
        "r03_02_test_audit.json": assoc_test_audit,
    }
    for name, rows in files.items():
        write_json(args.output_dir / name, rows)
    manifest = {
        "schema_version": "1.0", "warning": "R03-02 reference-evidence test is a structural/oracle-evidence ceiling only; final test must use R03 single-view predictions.",
        "counts": {name: len(rows) for name, rows in files.items()},
        "r03_01": {"task": "multiple pathology images to diagnosis only", "target_prefix": "病理诊断："},
        "r03_02": {"task": "multiple single-view morphology texts to diagnosis", "final_test_input": "R03 single-view predictions grouped by immutable audit map"},
        "direct_train_image_count": len(train_images), "direct_test_image_count": len(test_images),
        "association_train_image_count": len(assoc_train_images), "association_test_image_count": len(assoc_test_images),
        "association_train_views_per_case": dict(Counter(x["view_count"] for x in assoc_train_audit)),
        "association_test_views_per_case": dict(Counter(x["view_count"] for x in assoc_test_audit)),
        "sources": {str(path): sha256(path) for path in [args.clean_train, args.clean_test, args.one_train, args.one_test, args.one_audit]},
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
