#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from safetensors import safe_open


EXPECTED = {
    "R04": {"vision_lora": True, "projector": True, "language_lora": True},
    "R05": {"vision_lora": False, "projector": True, "language_lora": True},
    "R06": {"vision_lora": True, "projector": False, "language_lora": True},
    "R07": {"vision_lora": True, "projector": True, "language_lora": False},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, choices=sorted(EXPECTED))
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    adapter_path = Path(args.adapter_dir) / "adapter_model.safetensors"
    if not adapter_path.is_file():
        raise FileNotFoundError(adapter_path)

    with safe_open(adapter_path, framework="np", device="cpu") as handle:
        keys = list(handle.keys())
        parameter_count = sum(math.prod(handle.get_slice(key).get_shape()) for key in keys)

    observed = {
        "vision_lora": any("visual.blocks" in key and "lora_" in key for key in keys),
        "projector": any("visual.merger" in key for key in keys),
        "language_lora": any("language_model" in key and "lora_" in key for key in keys),
    }
    expected = EXPECTED[args.run_id]
    errors = [name for name, value in expected.items() if observed[name] is not value]
    result = {
        "run_id": args.run_id,
        "adapter": str(adapter_path),
        "key_count": len(keys),
        "saved_parameter_count": parameter_count,
        "expected": expected,
        "observed": observed,
        "checks_passed": not errors,
        "mismatches": errors,
        "representative_keys": keys[:20],
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
