#!/usr/bin/env python3
"""Generate one deterministic prediction per test case on a rank-specific shard."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_completed(path: Path) -> set[int]:
    completed: set[int] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if not row.get("error") and row.get("prediction") is not None:
                    completed.add(int(row["case_index"]))
            except Exception:
                continue
    return completed


def resolve_images(record: dict, image_root: str) -> list[str]:
    paths = []
    for raw in record.get("images", []):
        p = Path(raw)
        if not p.is_absolute():
            p = Path(image_root) / p
        paths.append(str(p.resolve()))
    return paths


def extract_prompt_and_reference(record: dict) -> tuple[str, str]:
    prompt = ""
    reference = ""
    for message in record.get("messages", []):
        if message.get("role") == "user" and not prompt:
            prompt = str(message.get("content", ""))
        elif message.get("role") == "assistant" and not reference:
            reference = str(message.get("content", ""))
    return prompt, reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("bootstrap_seed", 20260810)) + args.rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    with open(cfg["test_json"], "r", encoding="utf-8") as f:
        records = json.load(f)
    if args.limit is not None:
        records = records[: args.limit]

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    part_path = output_dir / f"predictions.part{args.rank}.jsonl"
    completed = load_completed(part_path)

    processor_source = cfg["model_dir"]
    if not (Path(processor_source) / "processor_config.json").exists():
        processor_source = cfg["base_processor_dir"]

    processor = AutoProcessor.from_pretrained(
        processor_source,
        trust_remote_code=True,
        min_pixels=int(cfg.get("min_pixels", 262144)),
        max_pixels=int(cfg.get("max_pixels", 262144)),
        local_files_only=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["model_dir"],
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    adapter_dir = cfg.get("adapter_dir")
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model = model.to("cuda:0")
    model.eval()

    max_new_tokens = int(cfg.get("max_new_tokens", 1024))
    do_sample = bool(cfg.get("do_sample", False))
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
    }
    if do_sample:
        generation_kwargs.update(
            temperature=float(cfg.get("temperature", 0.7)),
            top_p=float(cfg.get("top_p", 0.9)),
        )

    shard_indices = [i for i in range(len(records)) if i % args.world_size == args.rank]
    print(f"rank={args.rank} cases={len(shard_indices)} already_completed={len(completed)}", flush=True)

    with part_path.open("a", encoding="utf-8") as out:
        for ordinal, idx in enumerate(shard_indices, start=1):
            if idx in completed:
                continue
            record = records[idx]
            prompt, reference = extract_prompt_and_reference(record)
            image_paths = resolve_images(record, cfg["image_root"])
            started = time.perf_counter()
            row = {
                "case_index": idx,
                "rank": args.rank,
                "images": record.get("images", []),
                "resolved_images": image_paths,
                "prompt": prompt,
                "reference": reference,
                "prediction": None,
                "error": None,
            }
            images: list[Image.Image] = []
            try:
                missing = [p for p in image_paths if not Path(p).is_file()]
                if missing:
                    raise FileNotFoundError(f"missing image(s): {missing}")
                images = [Image.open(p).convert("RGB") for p in image_paths]
                # The dataset stores literal <image> placeholders. Hugging Face's
                # Qwen3-VL chat template only emits visual tokens for structured
                # content items, so rebuild the user message explicitly.
                prompt_without_placeholders = prompt.replace("<image>", "").strip()
                structured_content = [{"type": "image"} for _ in images]
                structured_content.append({"type": "text", "text": prompt_without_placeholders})
                conversation = [{"role": "user", "content": structured_content}]
                rendered = processor.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                processor_kwargs = {
                    "text": [rendered],
                    "padding": True,
                    "return_tensors": "pt",
                }
                # Qwen3-VL indexes image-size metadata when the images keyword is
                # present. Text-only evidence-to-diagnosis records therefore
                # must omit the keyword instead of passing an empty list.
                if images:
                    processor_kwargs["images"] = images
                inputs = processor(**processor_kwargs)
                inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}
                input_tokens = int(inputs["input_ids"].shape[-1])
                torch.cuda.reset_peak_memory_stats(0)
                with torch.inference_mode():
                    generated = model.generate(**inputs, **generation_kwargs)
                continuation = generated[:, input_tokens:]
                prediction = processor.batch_decode(
                    continuation,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                output_tokens = int(continuation.shape[-1])
                row.update(
                    prediction=prediction,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    hit_max_new_tokens=output_tokens >= max_new_tokens,
                    peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(0)),
                )
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc(limit=8)
            finally:
                for image in images:
                    image.close()
                row["latency_sec"] = time.perf_counter() - started
                row["generated_at_unix"] = time.time()
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                status = "OK" if not row.get("error") else "ERROR"
                print(
                    f"rank={args.rank} {ordinal}/{len(shard_indices)} case={idx} "
                    f"status={status} latency={row['latency_sec']:.2f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
