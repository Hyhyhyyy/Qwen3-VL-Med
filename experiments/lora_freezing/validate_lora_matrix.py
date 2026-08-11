#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
FILES = [
    "lora_01_r04_all_components.yaml",
    "lora_02_r05_freeze_vision.yaml",
    "lora_03_r06_freeze_projector.yaml",
    "lora_04_r07_freeze_language.yaml",
]
CONTROLLED = [
    "model_name_or_path", "template", "stage", "do_train", "finetuning_type",
    "lora_target", "lora_rank", "lora_alpha", "lora_dropout", "use_rslora",
    "use_dora", "dataset_dir", "dataset", "cutoff_len", "image_max_pixels",
    "image_min_pixels", "per_device_train_batch_size", "gradient_accumulation_steps",
    "learning_rate", "num_train_epochs", "lr_scheduler_type", "warmup_ratio",
    "bf16", "gradient_checkpointing", "flash_attn", "save_strategy", "seed",
]


def differing(a: dict, b: dict) -> set[str]:
    return {key for key in set(a) | set(b) if a.get(key) != b.get(key)}


def main() -> None:
    configs = {name: yaml.safe_load((ROOT / name).read_text(encoding="utf-8")) for name in FILES}
    baseline = configs[FILES[0]]
    errors: list[str] = []

    for name, cfg in configs.items():
        for key in CONTROLLED:
            if cfg.get(key) != baseline.get(key):
                errors.append(f"unexpected controlled mismatch {name}: {key}")
        if cfg.get("output_dir") == baseline.get("output_dir") and name != FILES[0]:
            errors.append(f"duplicate output_dir: {name}")
        if cfg.get("save_strategy") != "no":
            errors.append(f"intermediate checkpoints enabled: {name}")

    expected = {
        FILES[1]: {"freeze_vision_tower", "output_dir"},
        FILES[2]: {"freeze_multi_modal_projector", "additional_target", "output_dir"},
        FILES[3]: {"freeze_language_model", "output_dir"},
    }
    for name, expected_keys in expected.items():
        actual = differing(baseline, configs[name])
        if actual != expected_keys:
            errors.append(f"invalid single-factor contrast {name}: {sorted(actual)}")

    if not (
        baseline["freeze_vision_tower"] is False
        and baseline["freeze_multi_modal_projector"] is False
        and baseline["freeze_language_model"] is False
        and baseline["additional_target"] == "visual.merger"
    ):
        errors.append("R04 must adapt vision, projector and language")
    if configs[FILES[1]]["freeze_vision_tower"] is not True:
        errors.append("R05 must freeze vision")
    if configs[FILES[2]]["freeze_multi_modal_projector"] is not True:
        errors.append("R06 must freeze projector")
    if configs[FILES[2]].get("additional_target") is not None:
        errors.append("R06 must not save/train projector as an additional target")
    if configs[FILES[3]]["freeze_language_model"] is not True:
        errors.append("R07 must freeze language")

    result = {"files": FILES, "all_checks_passed": not errors, "errors": errors}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
