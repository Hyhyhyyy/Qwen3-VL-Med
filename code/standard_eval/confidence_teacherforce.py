#!/usr/bin/env python3
"""Compute generation-token confidence without regenerating archived reports.

For deterministic greedy decoding, a single teacher-forced forward pass over an
archived prediction yields the same next-token distributions as autoregressive
generation for that exact prefix.  This is substantially faster and preserves
the historical evaluation split for each run.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

METRIC_DIR = Path(__file__).resolve().parents[1] / "metric_eval"
sys.path.insert(0, str(METRIC_DIR))
from compute_metrics import canonical_diagnosis  # noqa: E402


def find_last_subsequence(sequence: list[int], pattern: list[int]) -> int:
    if not pattern or len(pattern) > len(sequence):
        return -1
    first = pattern[0]
    for start in range(len(sequence) - len(pattern), -1, -1):
        if sequence[start] == first and sequence[start : start + len(pattern)] == pattern:
            return start
    return -1


def resolve_images(row: dict, image_root: str) -> list[Path]:
    resolved = []
    for raw in row.get("images", []):
        path = Path(raw)
        if not path.is_absolute():
            path = Path(image_root) / path
        resolved.append(path.resolve())
    return resolved


def build_user_content(prompt: str, image_count: int) -> list[dict]:
    clean_prompt = prompt.replace("<image>", "").strip()
    return ([{"type": "image"} for _ in range(image_count)]
            + [{"type": "text", "text": clean_prompt}])


def calibration_summary(rows: list[dict], bins: int) -> dict:
    valid = [r for r in rows if r.get("error") is None]
    confs = [float(r["mean_token_confidence"]) for r in valid]
    corrects = [float(r["diagnosis_exact"]) for r in valid]
    reliability = []
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        selected = [i for i, c in enumerate(confs)
                    if (lo <= c < hi) or (b == bins - 1 and lo <= c <= hi)]
        if not selected:
            continue
        bin_conf = statistics.fmean(confs[i] for i in selected)
        bin_acc = statistics.fmean(corrects[i] for i in selected)
        fraction = len(selected) / len(valid)
        ece += fraction * abs(bin_conf - bin_acc)
        reliability.append({
            "bin_low": lo,
            "bin_high": hi,
            "confidence": round(bin_conf, 6),
            "accuracy": round(bin_acc, 6),
            "count": len(selected),
        })
    entropy = []
    for confidence in confs:
        p = min(max(confidence, 1e-6), 1 - 1e-6)
        entropy.append(-(p * math.log(p) + (1 - p) * math.log(1 - p)) / math.log(2))
    return {
        "available": bool(valid),
        "protocol": "teacher-forced archived greedy prediction; confidence=mean max-softmax over generated tokens; correctness=normalized diagnosis exact match",
        "ece_bins": bins,
        "ece": round(ece, 6),
        "mean_confidence": round(statistics.fmean(confs), 6),
        "mean_entropy_proxy": round(statistics.fmean(entropy), 6),
        "diagnosis_exact_accuracy": round(statistics.fmean(corrects), 6),
        "n_cases": len(valid),
        "n_errors": len(rows) - len(valid),
        "reliability": reliability,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    source_rows = [json.loads(line) for line in Path(args.predictions).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        source_rows = source_rows[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "confidence_cases.jsonl"
    completed = {}
    if detail_path.exists():
        for line in detail_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[int(row["case_index"])] = row

    if args.summary_only:
        selected = [completed[int(row.get("case_index", i))] for i, row in enumerate(source_rows)]
        summary = calibration_summary(selected, args.bins)
        (output_dir / "calibration.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

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
    if cfg.get("adapter_dir"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg["adapter_dir"], is_trainable=False)
    model = model.to("cuda:0").eval()

    mode = "a" if detail_path.exists() else "w"
    with detail_path.open(mode, encoding="utf-8") as output:
        for ordinal, source in enumerate(source_rows, start=1):
            case_index = int(source.get("case_index", ordinal - 1))
            if case_index in completed and completed[case_index].get("error") is None:
                print(f"{ordinal}/{len(source_rows)} case={case_index} SKIP", flush=True)
                continue
            started = time.perf_counter()
            result = {"case_index": case_index, "error": None}
            images = []
            try:
                prediction = str(source["prediction"]).strip()
                reference = str(source["reference"]).strip()
                prompt = str(source["prompt"])
                image_paths = resolve_images(source, cfg["image_root"])
                missing = [str(p) for p in image_paths if not p.is_file()]
                if missing:
                    raise FileNotFoundError(f"missing image(s): {missing}")
                images = [Image.open(path).convert("RGB") for path in image_paths]
                conversation = [
                    {"role": "user", "content": build_user_content(prompt, len(images))},
                    {"role": "assistant", "content": [{"type": "text", "text": prediction}]},
                ]
                rendered = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
                inputs = processor(text=[rendered], images=images, padding=True, return_tensors="pt")
                input_ids = inputs["input_ids"][0].tolist()
                prediction_ids = processor.tokenizer.encode(prediction, add_special_tokens=False)
                target_start = find_last_subsequence(input_ids, prediction_ids)
                if target_start <= 0:
                    raise ValueError("prediction token sequence not found in rendered conversation")
                inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}
                torch.cuda.reset_peak_memory_stats(0)
                with torch.inference_mode():
                    logits = model(**inputs, use_cache=False).logits[0, target_start - 1 : target_start + len(prediction_ids) - 1].float()
                    max_logit = logits.max(dim=-1).values
                    confidence = torch.exp(max_logit - torch.logsumexp(logits, dim=-1))
                result.update(
                    mean_token_confidence=float(confidence.mean().item()),
                    min_token_confidence=float(confidence.min().item()),
                    target_tokens=len(prediction_ids),
                    diagnosis_exact=float(canonical_diagnosis(reference) == canonical_diagnosis(prediction)),
                    peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(0)),
                )
                del logits, confidence, inputs
                torch.cuda.empty_cache()
            except Exception as exc:  # noqa: BLE001
                result["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                for image in images:
                    image.close()
                result["latency_sec"] = time.perf_counter() - started
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                print(f"{ordinal}/{len(source_rows)} case={case_index} {'OK' if result['error'] is None else result['error']} latency={result['latency_sec']:.2f}s", flush=True)

    latest = {}
    for line in detail_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[int(row["case_index"])] = row
    selected = [latest[int(r.get("case_index", i))] for i, r in enumerate(source_rows)]
    summary = calibration_summary(selected, args.bins)
    (output_dir / "calibration.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
