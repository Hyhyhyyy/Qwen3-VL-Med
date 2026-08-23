#!/usr/bin/env python3
"""Build a leakage-controlled Oracle/generated-evidence training mixture.

The script is deliberately data-agnostic. Clinical records stay outside Git;
only the expected field contract is public. Each training case contributes one
Oracle evidence record and one record assembled from archived upstream model
predictions. Test records are rejected by default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"expected a JSON list: {path}")
        return value
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def message_pair(record: dict[str, Any]) -> tuple[str, str]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("record has no ShareGPT messages list")
    user = next((str(x.get("content", "")) for x in messages if x.get("role") == "user"), "")
    assistant = next((str(x.get("content", "")) for x in messages if x.get("role") == "assistant"), "")
    if not user or not assistant:
        raise ValueError("record must contain non-empty user and assistant messages")
    return user, assistant


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(args: argparse.Namespace) -> dict[str, Any]:
    predictions_list = read_records(args.predictions)
    predictions: dict[int, dict[str, Any]] = {}
    for row in predictions_list:
        index = int(row[args.prediction_index_key])
        if index in predictions:
            raise ValueError(f"duplicate prediction index: {index}")
        predictions[index] = row

    cases = [row for row in read_records(args.case_manifest) if row.get(args.split_key) == args.train_split]
    if any(row.get(args.split_key) != args.train_split for row in cases):
        raise AssertionError("non-training case passed the split filter")
    cases.sort(key=lambda row: int(row[args.oracle_index_key]))
    oracle = read_records(args.oracle_train)
    if len(cases) != len(oracle):
        raise ValueError(f"case/oracle count mismatch: {len(cases)} != {len(oracle)}")
    if args.expected_cases is not None and len(cases) != args.expected_cases:
        raise ValueError(f"expected {args.expected_cases} cases, got {len(cases)}")

    mixed: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    used_predictions: set[int] = set()
    for position, (case, oracle_record) in enumerate(zip(cases, oracle)):
        oracle_index = int(case[args.oracle_index_key])
        if oracle_index != position:
            raise ValueError(f"non-contiguous Oracle linkage at position {position}: {oracle_index}")
        oracle_user, oracle_answer = message_pair(oracle_record)
        sample_indices = [int(value) for value in case[args.evidence_indices_key]]
        image_paths = [str(value) for value in case.get(args.image_paths_key, [])]
        if image_paths and len(image_paths) != len(sample_indices):
            raise ValueError(f"image/prediction linkage mismatch: {case[args.case_id_key]}")

        evidence_rows: list[str] = []
        for offset, sample_index in enumerate(sample_indices):
            prediction = predictions.get(sample_index)
            if prediction is None or prediction.get("error"):
                raise ValueError(f"missing or failed upstream prediction: {sample_index}")
            if image_paths:
                prediction_images = list(prediction.get("images") or [])
                if prediction_images != [image_paths[offset]]:
                    raise ValueError(f"upstream image mismatch: {sample_index}")
            text = str(prediction.get(args.prediction_text_key) or "").strip()
            if not text:
                raise ValueError(f"empty upstream prediction: {sample_index}")
            evidence_rows.append(text)
            used_predictions.add(sample_index)

        generated_user = args.prompt_prefix + "\n".join(
            f"{args.evidence_label}{number}: {text}"
            for number, text in enumerate(evidence_rows, start=1)
        )
        mixed.extend([
            oracle_record,
            {
                "messages": [
                    {"role": "user", "content": generated_user},
                    {"role": "assistant", "content": oracle_answer},
                ]
            },
        ])
        audits.append({
            "case_id": str(case[args.case_id_key]),
            "oracle_index": oracle_index,
            "upstream_prediction_indices": sample_indices,
            "view_count": len(sample_indices),
            "oracle_input_chars": len(oracle_user),
            "generated_input_chars": len(generated_user),
        })

    expected_predictions = {
        int(index)
        for case in cases
        for index in case[args.evidence_indices_key]
    }
    if used_predictions != expected_predictions:
        missing = len(expected_predictions - used_predictions)
        extra = len(used_predictions - expected_predictions)
        raise ValueError(f"upstream coverage mismatch: missing={missing} extra={extra}")

    random.Random(args.seed).shuffle(mixed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    output_path.write_text(json.dumps(mixed, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_path = args.output_dir / f"{Path(args.output_name).stem}.audit.json"
    audit_path.write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": "robust-evidence-mix-v1",
        "train_split": args.train_split,
        "unique_train_cases": len(cases),
        "oracle_records": len(cases),
        "generated_evidence_records": len(cases),
        "total_records": len(mixed),
        "upstream_predictions_used": len(used_predictions),
        "mixture": {"oracle": 0.5, "generated_evidence": 0.5},
        "shuffle_seed": args.seed,
        "non_train_records_used": 0,
        "output_sha256": file_sha256(output_path),
    }
    manifest_path = args.output_dir / f"{Path(args.output_name).stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--case-manifest", required=True, type=Path)
    parser.add_argument("--oracle-train", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-name", default="robust_evidence_mix.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-cases", type=int)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--split-key", default="split")
    parser.add_argument("--case-id-key", default="case_uid")
    parser.add_argument("--oracle-index-key", default="oracle_sample_index")
    parser.add_argument("--evidence-indices-key", default="evidence_sample_indices")
    parser.add_argument("--image-paths-key", default="image_paths")
    parser.add_argument("--prediction-index-key", default="case_index")
    parser.add_argument("--prediction-text-key", default="prediction")
    parser.add_argument("--prompt-prefix", default="Evidence from the same case:\n")
    parser.add_argument("--evidence-label", default="Evidence ")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
