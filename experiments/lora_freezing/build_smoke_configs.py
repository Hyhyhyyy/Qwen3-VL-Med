#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
OUT = Path("/tmp/four_lora_smoke_configs")
FILES = [
    "lora_01_r04_all_components.yaml",
    "lora_02_r05_freeze_vision.yaml",
    "lora_03_r06_freeze_projector.yaml",
    "lora_04_r07_freeze_language.yaml",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for source_name in FILES:
        config = yaml.safe_load((ROOT / source_name).read_text(encoding="utf-8"))
        run_tag = source_name.split("_")[2].lower()
        config.update(
            output_dir=f"./outputs/smoke_{run_tag}_10steps",
            max_samples=64,
            max_steps=10,
            num_train_epochs=1.0,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            preprocessing_num_workers=2,
            preprocessing_batch_size=4,
            logging_steps=1,
        )
        target = OUT / source_name.replace(".yaml", "_smoke.yaml")
        target.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(target)


if __name__ == "__main__":
    main()
