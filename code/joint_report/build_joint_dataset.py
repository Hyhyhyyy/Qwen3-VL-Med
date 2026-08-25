#!/usr/bin/env python3
"""Build a case-level joint diagnosis and per-image evidence target.

The script operates only on caller-supplied private files. It writes the derived
dataset and an audit manifest outside this repository; no clinical content is
embedded in the source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROMPT = (
    "Review every image from the same case and return exactly one JSON object. "
    "Provide overall_diagnosis first, then one ordered per_image_findings item "
    "for each image_id, followed by case-level and cross-image evidence. State "
    "uncertainty when the available evidence is insufficient."
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assistant_text(sample: dict) -> str:
    for message in reversed(sample["messages"]):
        if message["role"] == "assistant":
            return str(message["content"]).strip()
    raise ValueError("assistant message missing")


def split_report(report: str, diagnosis_prefix: str) -> tuple[str, str]:
    lines = [line.strip() for line in report.replace("\r\n", "\n").split("\n") if line.strip()]
    positions = [index for index, line in enumerate(lines) if line.startswith(diagnosis_prefix)]
    if positions != [len(lines) - 1]:
        raise ValueError("report must contain exactly one final diagnosis line")
    narrative = "\n".join(lines[:-1])
    diagnosis = lines[-1][len(diagnosis_prefix) :].strip()
    if not narrative or not diagnosis:
        raise ValueError("empty narrative or diagnosis")
    return narrative, diagnosis


def build_split(
    case_path: Path,
    image_path: Path,
    output_path: Path,
    diagnosis_prefix: str,
) -> dict:
    cases = load_json(case_path)
    image_rows = load_json(image_path)
    by_image: dict[str, str] = {}
    for row in image_rows:
        images = row.get("images", [])
        if len(images) != 1:
            raise ValueError("each image-level row must contain exactly one image")
        image = images[0]
        if image in by_image:
            raise ValueError("duplicate image key")
        by_image[image] = assistant_text(row)

    outputs = []
    image_count = 0
    for case in cases:
        images = case["images"]
        narrative, diagnosis = split_report(assistant_text(case), diagnosis_prefix)
        findings = []
        for position, image in enumerate(images, start=1):
            if image not in by_image:
                raise ValueError("missing aligned image-level evidence")
            findings.append(
                {
                    "image_id": f"image_{position}",
                    "quality": None,
                    "morphology": [by_image[image]],
                    "negative_evidence": [],
                    "uncertainty": None,
                }
            )
        image_count += len(images)
        target = {
            "overall_diagnosis": {
                "diagnosis": diagnosis,
                "grade_stage": None,
                "differential": [],
                "uncertainty": None,
            },
            "per_image_findings": findings,
            "case_level_findings": [],
            "integrated_evidence": {
                "supporting_image_ids": [f"image_{i}" for i in range(1, len(images) + 1)],
                "cross_image_summary": narrative,
                "missing_information": [],
            },
        }
        outputs.append(
            {
                "images": images,
                "messages": [
                    {"role": "user", "content": "<image>" * len(images) + PROMPT},
                    {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
                ],
            }
        )

    if len(by_image) != image_count:
        raise ValueError("case/image alignment is not one-to-one")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "case_count": len(cases),
        "image_count": image_count,
        "case_input_sha256": sha256(case_path),
        "image_input_sha256": sha256(image_path),
        "output_sha256": sha256(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-train", type=Path, required=True)
    parser.add_argument("--case-test", type=Path, required=True)
    parser.add_argument("--image-train", type=Path, required=True)
    parser.add_argument("--image-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--diagnosis-prefix", default="Diagnosis:")
    args = parser.parse_args()

    reports = {}
    for split in ("train", "test"):
        reports[split] = build_split(
            getattr(args, f"case_{split}"),
            getattr(args, f"image_{split}"),
            args.output_dir / f"joint_{split}.json",
            args.diagnosis_prefix,
        )

    train_images = {image for row in load_json(args.output_dir / "joint_train.json") for image in row["images"]}
    test_images = {image for row in load_json(args.output_dir / "joint_test.json") for image in row["images"]}
    if train_images & test_images:
        raise ValueError("train/test image leakage detected")

    audit = {"target_mode": "single_model_joint_multitask", "splits": reports, "image_overlap": 0}
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
