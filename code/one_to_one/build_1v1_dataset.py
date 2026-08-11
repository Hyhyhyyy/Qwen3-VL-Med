#!/usr/bin/env python3
"""Build a traceable one-image-to-one-observation dataset from expert view sections.

The case-level diagnosis is deliberately not copied into every image target.  Such
copying would turn a multi-image (bag) label into noisy per-instance supervision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


POSITIONS = ("左上图", "右上图", "左中图", "右中图", "左下图", "右下图", "上图", "下图")
POS_ALT = "|".join(map(re.escape, POSITIONS))
HEADER_RE = re.compile(rf"(?m)^\s*({POS_ALT})\s*[：:]?\s*")
ANY_POS_RE = re.compile(POS_ALT)
DIAG_RE = re.compile(r"病理诊断\s*[：:]\s*(.*)", re.S)
DIAG_START_RE = re.compile(r"(?m)^\s*病理诊断\s*[：:]")
PUNCT_RE = re.compile(r"[\s，。；;：:、\\（）()【】\[\]<>《》“”\"'—-]+")

PROMPT = (
    "<image>你是一名严谨的肝脏病理医师。请只依据这一张病理切片图像，输出“图像分析”。"
    "客观描述本图实际可见的组织结构及形态学改变，包括适用时的肝小叶、汇管区、肝细胞、"
    "炎症、坏死、脂肪变、胆管和纤维化等；不要补写本图未显示的其他视野信息，不要依据文件名"
    "或位置词猜测，也不要仅凭单张局部图像强行给出病例级最终诊断。"
)


def normalize(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def assistant_text(record: dict) -> str:
    return next((str(x.get("content", "")) for x in record.get("messages", [])
                 if x.get("role") == "assistant"), "")


def diagnosis(text: str) -> str:
    m = DIAG_RE.search(normalize(text))
    return m.group(1).splitlines()[0].strip() if m else ""


def split_view_sections(text: str) -> tuple[dict[str, str], list[str]]:
    """Return unique line-header sections and audit reasons."""
    text = normalize(text)
    body = DIAG_START_RE.split(text, maxsplit=1)[0].strip()
    matches = list(HEADER_RE.finditer(body))
    if not matches:
        return {}, ["no_line_start_position_sections"]
    sections: dict[str, str] = {}
    reasons = []
    for i, match in enumerate(matches):
        label = match.group(1)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = normalize(body[match.end():end])
        content = ANY_POS_RE.sub("", content)
        content = normalize(content)
        if label in sections:
            reasons.append("duplicate_position_section")
        elif content:
            sections[label] = content
        else:
            reasons.append("empty_position_section")
    return sections, sorted(set(reasons))


def image_position(path: str) -> str | None:
    name = Path(path).name
    hits = [p for p in POSITIONS if p in name]
    # Prefer specific six-grid labels over generic 上图/下图 substrings.
    hits = [p for p in hits if p not in ("上图", "下图")] or hits
    return hits[0] if len(hits) == 1 else None


def chinese_chars(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def canonical(text: str) -> str:
    return PUNCT_RE.sub("", normalize(text)).lower()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def build_split(raw: list[dict], kept_indices: list[int], split: str,
                image_root: Path, min_chars: int):
    output, audit, excluded = [], [], []
    seen_images, seen_targets = set(), set()
    for clean_case_index, source_index in enumerate(kept_indices):
        record = raw[source_index]
        images = list(record.get("images") or [])
        target = assistant_text(record)
        sections, parse_reasons = split_view_sections(target)
        mapping = [(image, image_position(image)) for image in images]
        reasons = list(parse_reasons)
        labels = [label for _, label in mapping]
        if any(label is None for label in labels):
            reasons.append("image_filename_position_unresolved")
        if len(set(x for x in labels if x)) != len(labels):
            reasons.append("duplicate_image_position")
        if set(x for x in labels if x) != set(sections):
            reasons.append("image_section_labels_not_exact_match")
        missing = []
        for image in images:
            path = Path(image)
            if not path.is_absolute():
                path = image_root / path
            if not path.is_file():
                missing.append(image)
        if missing:
            reasons.append("missing_image_file")
        if reasons:
            excluded.append({
                "split": split, "clean_case_index": clean_case_index,
                "source_index": source_index, "reasons": sorted(set(reasons)),
                "image_count": len(images), "section_labels": list(sections),
                "image_labels": labels,
            })
            continue

        case_samples = []
        case_audit = []
        for image, label in mapping:
            analysis = normalize(sections[label])
            local_reasons = []
            if chinese_chars(analysis) < min_chars:
                local_reasons.append("single_view_analysis_too_short")
            if image in seen_images:
                local_reasons.append("duplicate_image_across_samples")
            answer = f"图像分析：{analysis}"
            key = canonical(answer)
            if key in seen_targets:
                local_reasons.append("duplicate_single_view_target")
            if ANY_POS_RE.search(answer):
                local_reasons.append("position_token_remains_in_target")
            if local_reasons:
                excluded.append({
                    "split": split, "clean_case_index": clean_case_index,
                    "source_index": source_index, "image": image, "position": label,
                    "reasons": local_reasons,
                })
                continue
            sample = {
                "images": [image],
                "messages": [
                    {"role": "user", "content": PROMPT},
                    {"role": "assistant", "content": answer},
                ],
            }
            case_samples.append(sample)
            case_audit.append({
                "split": split, "one_to_one_index": len(output) + len(case_samples) - 1,
                "clean_case_index": clean_case_index, "source_index": source_index,
                "image": image, "source_position": label,
                "case_level_reference_diagnosis_not_used_as_target": diagnosis(target),
                "analysis_chars": chinese_chars(analysis),
            })
            seen_images.add(image)
            seen_targets.add(key)
        output.extend(case_samples)
        audit.extend(case_audit)
    return output, audit, excluded


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-train", required=True, type=Path)
    p.add_argument("--raw-test", required=True, type=Path)
    p.add_argument("--kept-map", required=True, type=Path)
    p.add_argument("--image-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--min-analysis-chars", type=int, default=12)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_raw = json.loads(args.raw_train.read_text(encoding="utf-8"))
    test_raw = json.loads(args.raw_test.read_text(encoding="utf-8"))
    index_map = json.loads(args.kept_map.read_text(encoding="utf-8"))

    train, train_audit, train_excluded = build_split(
        train_raw, index_map["train_clean_index_to_source_index"], "train",
        args.image_root, args.min_analysis_chars)
    test, test_audit, test_excluded = build_split(
        test_raw, index_map["test_clean_index_to_source_index"], "test",
        args.image_root, args.min_analysis_chars)

    # Exact target leakage is removed from train, while preserving case-level split.
    test_targets = {canonical(assistant_text(x)) for x in test}
    filtered_train, leakage = [], []
    for i, sample in enumerate(train):
        if canonical(assistant_text(sample)) in test_targets:
            leakage.append({"split": "train", "one_to_one_index_before_filter": i,
                            "reasons": ["exact_single_view_target_leaks_into_test"],
                            "image": sample["images"][0]})
        else:
            filtered_train.append(sample)
    train = filtered_train
    train_excluded.extend(leakage)

    (args.output_dir / "wsi_train_1v1_v1.json").write_text(
        json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "wsi_test_1v1_v1.json").write_text(
        json.dumps(test, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "one_to_one_audit.jsonl").open("w", encoding="utf-8") as f:
        for row in train_audit + test_audit:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "excluded_1v1.jsonl").open("w", encoding="utf-8") as f:
        for row in train_excluded + test_excluded:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    train_reasons, test_reasons = Counter(), Counter()
    for row in train_excluded:
        train_reasons.update(row["reasons"])
    for row in test_excluded:
        test_reasons.update(row["reasons"])
    manifest = {
        "version": "1v1-v1",
        "task": "one pathology image to one expert view-specific morphology description",
        "prompt": PROMPT,
        "diagnosis_policy": {
            "copy_case_diagnosis_to_each_image": False,
            "reason": "A case-level multi-image diagnosis is a bag label, not a reliable label for every local view.",
            "case_level_generation": "Generate each view description, then aggregate all descriptions at case level for final diagnosis.",
        },
        "quality_rules": {
            "source_cases_must_be_in_clean_v2": True,
            "require_exact_unique_position_header_to_image_filename_mapping": True,
            "minimum_single_view_analysis_chinese_characters": args.min_analysis_chars,
            "remove_all_position_tokens_from_targets": True,
            "deduplicate_images_and_targets": True,
            "remove_exact_train_target_leakage_into_test": True,
            "preserve_case_level_train_test_split": True,
        },
        "sources": {
            "raw_train": str(args.raw_train), "raw_train_sha256": sha256(args.raw_train),
            "raw_test": str(args.raw_test), "raw_test_sha256": sha256(args.raw_test),
            "clean_v2_kept_map": str(args.kept_map), "clean_v2_kept_map_sha256": sha256(args.kept_map),
        },
        "counts": {
            "train_1v1_samples": len(train), "test_1v1_samples": len(test),
            "train_source_views_before_leakage_filter": len(train_audit),
            "train_excluded_items_or_cases": len(train_excluded),
            "test_excluded_items_or_cases": len(test_excluded),
        },
        "train_exclusion_reason_counts": dict(train_reasons.most_common()),
        "test_exclusion_reason_counts": dict(test_reasons.most_common()),
    }
    (args.output_dir / "manifest_1v1.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    examples = {"train": train[:3], "test": test[:3]}
    (args.output_dir / "examples_1v1.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    print(json.dumps({"train_reasons": manifest["train_exclusion_reason_counts"],
                      "test_reasons": manifest["test_exclusion_reason_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
