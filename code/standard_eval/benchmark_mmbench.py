#!/usr/bin/env python3
"""Evaluate a local Qwen3-VL checkpoint on MMBench DEV EN v1.1.

The downloaded VLMEvalKit TSV contains circular option permutations.  We report
vanilla accuracy on all rows, vanilla accuracy on the original (circ0) rows,
and strict circular accuracy (a base question counts only when every option
permutation is answered correctly).  Exact letter extraction avoids an API
judge and is deterministic.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import statistics
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def extract_choice(text: str, choices: list[str]) -> str | None:
    tail = text.split("</think>")[-1].strip().upper()
    patterns = [
        r"^\s*[\(\[]?([A-Z])[\)\].,:;\s]",
        r"(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)\s*[\(\[]?([A-Z])",
        r"\b([A-Z])\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, tail):
            letter = match.group(1)
            if letter in choices:
                return letter
    return None


def decode_image(encoded: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def resolve_image(
    encoded_or_index: str,
    image_lookup: dict[int, str],
    image_cache: dict[int, Image.Image],
    source_index: int,
) -> Image.Image:
    """MMBench circular rows store the original row index instead of Base64."""
    value = encoded_or_index.strip()
    cache_key = int(value) if value.isdigit() else source_index % 1_000_000
    if cache_key not in image_cache:
        image_cache[cache_key] = decode_image(image_lookup[cache_key] if value.isdigit() else value)
    return image_cache[cache_key].copy()


def summary(rows: list[dict]) -> dict:
    valid = [row for row in rows if row.get("error") is None]
    parsed = [row for row in valid if row.get("prediction_choice") is not None]
    original = [row for row in parsed if int(row["index"]) <= 1_000_000]
    groups: dict[int, list[dict]] = {}
    for row in parsed:
        groups.setdefault(int(row["index"]) % 1_000_000, []).append(row)
    circular_hits = [all(item["hit"] for item in group) for group in groups.values()]
    by_category = {}
    for category in sorted({str(row["category"]) for row in parsed}):
        subset = [row for row in parsed if str(row["category"]) == category]
        by_category[category] = round(statistics.fmean(float(row["hit"]) for row in subset), 6)
    return {
        "benchmark": "MMBench_DEV_EN_V11",
        "protocol": "VLMEvalKit official TSV; deterministic generation; exact option-letter extraction; official Qwen demo pixel bounds",
        "n_rows": len(rows),
        "n_valid": len(valid),
        "n_parsed": len(parsed),
        "parse_rate": round(len(parsed) / len(valid), 6) if valid else 0.0,
        "vanilla_all_accuracy": round(statistics.fmean(float(row["hit"]) for row in parsed), 6) if parsed else None,
        "vanilla_circ0_accuracy": round(statistics.fmean(float(row["hit"]) for row in original), 6) if original else None,
        "strict_circular_accuracy": round(statistics.fmean(circular_hits), 6) if circular_hits else None,
        "n_circular_groups": len(groups),
        "n_errors": len(rows) - len(valid),
        "by_category": by_category,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    frame = pd.read_csv(args.dataset, sep="\t")
    image_lookup = {
        int(item["index"]): str(item["image"])
        for _, item in frame.iterrows()
        if not str(item["image"]).strip().isdigit()
    }
    image_cache: dict[int, Image.Image] = {}
    if args.limit is not None:
        frame = frame.iloc[: args.limit]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "predictions.jsonl"
    completed = {}
    if detail_path.exists():
        for line in detail_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[int(row["row_number"])] = row

    processor_source = cfg["model_dir"]
    if not (Path(processor_source) / "processor_config.json").exists():
        processor_source = cfg["base_processor_dir"]
    processor = AutoProcessor.from_pretrained(
        processor_source,
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        local_files_only=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["model_dir"],
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    if cfg.get("adapter_dir"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg["adapter_dir"], is_trainable=False)
    model = model.to("cuda:0").eval()
    processor.tokenizer.padding_side = "left"

    mode = "a" if detail_path.exists() else "w"
    with detail_path.open(mode, encoding="utf-8") as output_file:
        pending = [(row_number, item) for row_number, (_, item) in enumerate(frame.iterrows())
                   if not (row_number in completed and completed[row_number].get("error") is None)]
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            started = time.perf_counter()
            results = []
            images = []
            rendered_texts = []
            choice_sets = []
            try:
                for row_number, item in batch:
                    result = {
                        "row_number": row_number,
                        "index": int(item["index"]),
                        "answer": str(item["answer"]).strip().upper(),
                        "category": str(item["category"]),
                        "l2_category": str(item["l2-category"]),
                        "prediction_choice": None,
                        "prediction_text": None,
                        "hit": False,
                        "error": None,
                    }
                    results.append(result)
                    choices = [letter for letter in "ABCDE" if letter in frame.columns and pd.notna(item.get(letter))]
                    choice_sets.append(choices)
                    prompt_parts = []
                    if pd.notna(item.get("hint")):
                        prompt_parts.append(f"Hint: {item['hint']}")
                    prompt_parts.append(f"Question: {item['question']}")
                    prompt_parts.append("Options:")
                    prompt_parts.extend(f"{letter}. {item[letter]}" for letter in choices)
                    prompt_parts.append("Select the correct answer. Answer with only the option letter.")
                    prompt = "\n".join(prompt_parts)
                    images.append(resolve_image(
                        str(item["image"]), image_lookup, image_cache, int(item["index"])
                    ))
                    conversation = [{"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ]}]
                    rendered_texts.append(processor.apply_chat_template(
                        conversation, tokenize=False, add_generation_prompt=True
                    ))

                inputs = processor(text=rendered_texts, images=images, padding=True, return_tensors="pt")
                inputs = {key: value.to("cuda:0") if hasattr(value, "to") else value for key, value in inputs.items()}
                input_width = int(inputs["input_ids"].shape[-1])
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                texts = processor.batch_decode(
                    generated[:, input_width:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                elapsed_each = (time.perf_counter() - started) / len(batch)
                for result, text_value, choices in zip(results, texts, choice_sets):
                    text_value = text_value.strip()
                    choice = extract_choice(text_value, choices)
                    result.update(
                        prediction_choice=choice,
                        prediction_text=text_value,
                        hit=choice == result["answer"],
                        peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(0)),
                        latency_sec=elapsed_each,
                    )
            except Exception as exc:  # noqa: BLE001
                elapsed_each = (time.perf_counter() - started) / max(1, len(batch))
                for result in results:
                    result["error"] = f"{type(exc).__name__}: {exc}"
                    result["latency_sec"] = elapsed_each
            finally:
                for image in images:
                    image.close()
                for result in results:
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    print(
                        f"{result['row_number'] + 1}/{len(frame)} row={result['row_number']} "
                        f"{'OK' if result['error'] is None else result['error']} "
                        f"choice={result['prediction_choice']} hit={int(result['hit'])} "
                        f"latency={result['latency_sec']:.2f}s",
                        flush=True,
                    )
                output_file.flush()

    for cached_image in image_cache.values():
        cached_image.close()

    latest = {}
    for line in detail_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[int(row["row_number"])] = row
    rows = [latest[i] for i in range(len(frame))]
    result_summary = summary(rows)
    (output_dir / "benchmark.json").write_text(json.dumps(result_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
