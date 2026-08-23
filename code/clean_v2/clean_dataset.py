#!/usr/bin/env python3
"""Create traceable clean-v2 train/test JSON without mutating source data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


POSITION = r"(?:(?:左|中|右)(?:上|中|下)|上|下)图"
POSITION_RE = re.compile(POSITION)
POSITION_HEADER_RE = re.compile(rf"(?m)^\s*{POSITION}\s*[：:]?\s*")
POSITION_HEADER_CAPTURE_RE = re.compile(rf"(?m)^\s*({POSITION})\s*[：:]?\s*")
DIAG_RE = re.compile(r"病理诊断\s*[：:]\s*(.*)", re.S)
PUNCT_RE = re.compile(r"[\s，,。；;：:、/\\（）()【】\[\]<>《》·—_-]+")

PATHOLOGY_CUES = (
    "肝穿", "汇管区", "小叶", "肝细胞", "胆管", "炎", "坏死", "脂变", "脂肪",
    "纤维", "淤胆", "胆汁", "肉芽肿", "浆细胞", "淋巴细胞", "肝窦", "门静脉",
    "内皮", "铜", "铁", "CK7", "网状支架", "胶原", "染色", "结构", "浸润",
)


def normalize_space(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_position_headers(text: str) -> str:
    text = normalize_space(text)
    text = POSITION_HEADER_RE.sub("", text)
    # Remove inline visual-position references as well. They are annotation
    # metadata rather than pathology facts and inflate lexical metrics.
    text = POSITION_RE.sub("", text)
    text = re.sub(r"[（(]\s*[）)]", "", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def compact(text: str) -> str:
    return PUNCT_RE.sub("", normalize_space(text)).lower()


def assistant_text(record: dict) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def user_text(record: dict) -> str:
    for message in record.get("messages", []):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def diagnosis(text: str) -> str:
    match = DIAG_RE.search(normalize_space(text))
    if not match:
        return ""
    return match.group(1).splitlines()[0].strip()


def analysis_body(text: str) -> str:
    match = re.search(r"病理诊断\s*[：:]", normalize_space(text))
    return text[: match.start()].strip() if match else text.strip()


def chinese_char_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def source_signature(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def update_assistant(record: dict, cleaned_target: str) -> dict:
    result = copy.deepcopy(record)
    for message in result.get("messages", []):
        if message.get("role") == "assistant":
            message["content"] = cleaned_target
            return result
    return result


def base_reasons(record: dict, target: str, cleaned: str, image_root: Path, min_analysis_chars: int) -> list[str]:
    reasons = []
    images = list(record.get("images") or [])
    prompt = user_text(record)
    line_headers = POSITION_HEADER_CAPTURE_RE.findall(target)
    line_header_counts = Counter(line_headers)
    diag = diagnosis(cleaned)
    body = analysis_body(cleaned)
    body_compact = compact(body)
    cue_count = sum(cue.lower() in body.lower() for cue in PATHOLOGY_CUES)

    if not target.strip():
        reasons.append("missing_assistant_target")
    if not images:
        reasons.append("no_images")
    if len(images) != len(set(images)):
        reasons.append("duplicate_image_path_within_case")
    if prompt.count("<image>") != len(images):
        reasons.append("image_placeholder_count_mismatch")
    missing = []
    for raw in images:
        path = Path(raw)
        if not path.is_absolute():
            path = image_root / path
        if not path.is_file():
            missing.append(raw)
    if missing:
        reasons.append("missing_image_file")
    if any(count > 1 for count in line_header_counts.values()):
        reasons.append("repeated_position_header")
    if len(line_headers) > len(images):
        reasons.append("position_header_count_exceeds_images")
    if not DIAG_RE.search(cleaned):
        reasons.append("missing_diagnosis_section")
    elif not diag:
        reasons.append("empty_diagnosis")
    if compact(diag) in {"未找到", "无法诊断", "不详", "无"}:
        reasons.append("diagnosis_unavailable_placeholder")
    if chinese_char_count(body_compact) < min_analysis_chars:
        reasons.append("analysis_too_short")
    if cue_count < 2:
        reasons.append("insufficient_pathology_content")
    if chinese_char_count(compact(cleaned)) < min_analysis_chars + 8:
        reasons.append("target_too_short")
    return sorted(set(reasons))


def prepare(records: list[dict], split: str, image_root: Path, min_analysis_chars: int):
    prepared = []
    excluded = []
    diagnostics = Counter()
    position_tokens = Counter()
    for index, record in enumerate(records):
        target = assistant_text(record)
        position_tokens.update(POSITION_RE.findall(target))
        cleaned = strip_position_headers(target)
        reasons = base_reasons(record, target, cleaned, image_root, min_analysis_chars)
        item = {
            "source_index": index,
            "record": update_assistant(record, cleaned),
            "cleaned_target": cleaned,
            "canonical_target": compact(cleaned),
            "diagnosis": diagnosis(cleaned),
            "removed_position_header_count": len(POSITION_RE.findall(target)),
        }
        if reasons:
            excluded.append({
                "split": split,
                "source_index": index,
                "reasons": reasons,
                "images": record.get("images", []),
                "diagnosis": item["diagnosis"],
                "target_preview": normalize_space(target)[:500],
            })
        else:
            prepared.append(item)
            diagnostics[item["diagnosis"]] += 1
    return prepared, excluded, diagnostics, position_tokens


def deduplicate(items: list[dict], split: str):
    kept = []
    excluded = []
    seen = {}
    for item in items:
        key = item["canonical_target"]
        if key in seen:
            excluded.append({
                "split": split,
                "source_index": item["source_index"],
                "reasons": ["duplicate_cleaned_target"],
                "duplicate_of_source_index": seen[key],
                "images": item["record"].get("images", []),
                "diagnosis": item["diagnosis"],
                "target_preview": item["cleaned_target"][:500],
            })
        else:
            seen[key] = item["source_index"]
            kept.append(item)
    return kept, excluded


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-analysis-chars", type=int, default=20)
    args = parser.parse_args()

    train_path, test_path = Path(args.train_json), Path(args.test_json)
    image_root, output_dir = Path(args.image_root), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_raw = json.load(open(train_path, "r", encoding="utf-8"))
    test_raw = json.load(open(test_path, "r", encoding="utf-8"))

    train, train_excluded, train_diag, train_positions = prepare(
        train_raw, "train", image_root, args.min_analysis_chars
    )
    test, test_excluded, test_diag, test_positions = prepare(
        test_raw, "test", image_root, args.min_analysis_chars
    )
    train, train_dup = deduplicate(train, "train")
    test, test_dup = deduplicate(test, "test")
    train_excluded.extend(train_dup)
    test_excluded.extend(test_dup)

    # Prevent exact cleaned-label leakage into the evaluation set.
    clean_test_targets = {item["canonical_target"] for item in test}
    final_train = []
    leakage_excluded = []
    for item in train:
        if item["canonical_target"] in clean_test_targets:
            leakage_excluded.append({
                "split": "train",
                "source_index": item["source_index"],
                "reasons": ["exact_cleaned_target_leaks_into_test"],
                "images": item["record"].get("images", []),
                "diagnosis": item["diagnosis"],
                "target_preview": item["cleaned_target"][:500],
            })
        else:
            final_train.append(item)
    train_excluded.extend(leakage_excluded)

    train_records = [item["record"] for item in final_train]
    test_records = [item["record"] for item in test]
    train_out = output_dir / "wsi_train_clean_v2.json"
    test_out = output_dir / "wsi_test_clean_v2.json"
    write_json(train_out, train_records)
    write_json(test_out, test_records)
    index_map = {
        "train_clean_index_to_source_index": [item["source_index"] for item in final_train],
        "test_clean_index_to_source_index": [item["source_index"] for item in test],
    }
    write_json(output_dir / "kept_index_map.json", index_map)

    example_candidates = [item for item in final_train if item["removed_position_header_count"] > 0][:5]
    examples = []
    for item in example_candidates:
        source_index = item["source_index"]
        examples.append({
            "source_index": source_index,
            "images": train_raw[source_index].get("images", []),
            "before": assistant_text(train_raw[source_index]),
            "after": item["cleaned_target"],
        })
    write_json(output_dir / "before_after_examples.json", examples)

    excluded_path = output_dir / "excluded_records.jsonl"
    with excluded_path.open("w", encoding="utf-8") as f:
        for item in sorted(train_excluded + test_excluded, key=lambda x: (x["split"], x["source_index"])):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    reason_counts = Counter()
    for item in train_excluded:
        reason_counts.update(item["reasons"])
    test_reason_counts = Counter()
    for item in test_excluded:
        test_reason_counts.update(item["reasons"])
    removed_headers_train = sum(item["removed_position_header_count"] for item in final_train)
    removed_headers_test = sum(item["removed_position_header_count"] for item in test)
    manifest = {
        "version": "clean-v2",
        "rules": {
            "strip_position_headers_from_all_kept_targets": True,
            "minimum_analysis_chinese_characters": args.min_analysis_chars,
            "minimum_pathology_cue_count": 2,
            "exclude_diagnosis_only_or_unavailable": True,
            "exclude_missing_or_duplicate_images": True,
            "exclude_image_placeholder_mismatch": True,
            "exclude_repeated_or_excess_position_headers": True,
            "deduplicate_exact_cleaned_targets_keep_first": True,
            "exclude_exact_cleaned_target_leakage_into_test": True,
        },
        "source": {
            "train_json": str(train_path),
            "test_json": str(test_path),
            "train_sha256": source_signature(train_path),
            "test_sha256": source_signature(test_path),
        },
        "counts": {
            "source_train": len(train_raw),
            "kept_train": len(train_records),
            "excluded_train": len(train_raw) - len(train_records),
            "source_test": len(test_raw),
            "kept_test": len(test_records),
            "excluded_test": len(test_raw) - len(test_records),
            "position_headers_removed_from_kept_train": removed_headers_train,
            "position_headers_removed_from_kept_test": removed_headers_test,
            "exact_target_leakage_removed_from_train": len(leakage_excluded),
        },
        "train_exclusion_reason_counts": dict(reason_counts.most_common()),
        "test_exclusion_reason_counts": dict(test_reason_counts.most_common()),
        "source_train_position_token_counts": dict(train_positions.most_common()),
        "source_test_position_token_counts": dict(test_positions.most_common()),
        "kept_train_top_diagnoses_before_dedup": train_diag.most_common(30),
        "kept_test_top_diagnoses": test_diag.most_common(30),
        "outputs": {
            "train": str(train_out),
            "test": str(test_out),
            "excluded": str(excluded_path),
            "kept_index_map": str(output_dir / "kept_index_map.json"),
            "before_after_examples": str(output_dir / "before_after_examples.json"),
        },
    }
    write_json(output_dir / "cleaning_manifest.json", manifest)

    report = [
        "# Qwen3-VL clean-v2 数据清洗报告",
        "",
        f"- 原训练集：{len(train_raw)}",
        f"- 保留训练集：{len(train_records)}",
        f"- 排除训练集：{len(train_raw) - len(train_records)}",
        f"- 原测试集：{len(test_raw)}",
        f"- 高质量测试集：{len(test_records)}",
        f"- 排除测试集：{len(test_raw) - len(test_records)}",
        f"- 从保留训练答案移除位置标题：{removed_headers_train}",
        f"- 从保留测试答案移除位置标题：{removed_headers_test}",
        f"- 从训练集移除的精确测试答案泄漏：{len(leakage_excluded)}",
        "",
        "## 训练排除原因（原因可重叠）",
        "",
    ]
    report.extend(f"- {reason}: {count}" for reason, count in reason_counts.most_common())
    report.extend(["", "## 测试排除原因（原因可重叠）", ""])
    report.extend(f"- {reason}: {count}" for reason, count in test_reason_counts.most_common())
    report.extend([
        "",
        "## 说明",
        "",
        "位置标题只从保留答案中删除，不删除其后的医学描述。每条排除记录均记录原始索引和全部原因。",
        "自动规则不能替代病理专家标注；第二轮对比应同时报告过滤规模和排除清单。",
        "",
    ])
    (output_dir / "cleaning_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
