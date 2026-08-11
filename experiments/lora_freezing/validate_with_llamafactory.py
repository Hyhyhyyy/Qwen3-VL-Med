#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import yaml
from llamafactory.hparams.parser import _parse_train_args


ROOT = Path(__file__).resolve().parent
FILES = sorted(ROOT.glob("lora_0[1-4]_*.yaml"))


def main():
    results = []
    for path in FILES:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        model_args, data_args, training_args, finetuning_args, _ = _parse_train_args(raw)
        results.append({
            "file": path.name,
            "model": model_args.model_name_or_path,
            "dataset": data_args.dataset,
            "output_dir": training_args.output_dir,
            "finetuning_type": finetuning_args.finetuning_type,
            "lora_target": finetuning_args.lora_target,
            "lora_rank": finetuning_args.lora_rank,
            "lora_alpha": finetuning_args.lora_alpha,
            "lora_dropout": finetuning_args.lora_dropout,
            "additional_target": finetuning_args.additional_target,
            "freeze_vision_tower": finetuning_args.freeze_vision_tower,
            "freeze_multi_modal_projector": finetuning_args.freeze_multi_modal_projector,
            "freeze_language_model": finetuning_args.freeze_language_model,
            "per_device_batch": training_args.per_device_train_batch_size,
            "gradient_accumulation": training_args.gradient_accumulation_steps,
            "effective_four_gpu_batch": (
                training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * 4
            ),
            "save_strategy": str(training_args.save_strategy),
        })
    print(json.dumps({"llamafactory_dataclass_parser_passed": len(results) == 4, "configs": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
