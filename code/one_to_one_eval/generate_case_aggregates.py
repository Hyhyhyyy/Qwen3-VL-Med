#!/usr/bin/env python3
"""Generate one text-only case diagnosis from aggregated R03 view evidence."""
from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def completed_indices(path: Path) -> set[int]:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if not row.get("error") and str(row.get("prediction") or "").strip():
                    done.add(int(row["case_index"]))
            except Exception:
                pass
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--world-size", required=True, type=int)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(cfg.get("bootstrap_seed", 20260810)) + args.rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    records = json.loads(Path(cfg["test_json"]).read_text(encoding="utf-8"))
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    part = output_dir / f"predictions.part{args.rank}.jsonl"
    done = completed_indices(part)
    processor_source = cfg["model_dir"] if (Path(cfg["model_dir"]) / "processor_config.json").exists() else cfg["base_processor_dir"]
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["model_dir"], torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True, local_files_only=True,
    )
    if cfg.get("adapter_dir"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg["adapter_dir"], is_trainable=False)
    model = model.to("cuda:0")
    model.eval()
    indices = [i for i in range(len(records)) if i % args.world_size == args.rank]
    with part.open("a", encoding="utf-8") as handle:
        for ordinal, idx in enumerate(indices, 1):
            if idx in done:
                continue
            record = records[idx]
            prompt = next(str(x.get("content", "")) for x in record["messages"] if x.get("role") == "user")
            reference = next(str(x.get("content", "")) for x in record["messages"] if x.get("role") == "assistant")
            started = time.perf_counter()
            row = {
                "case_index": idx, "rank": args.rank, "images": [], "prompt": prompt,
                "reference": reference, "prediction": None, "error": None,
                "clean_case_index": record["clean_case_index"], "source_index": record["source_index"],
                "view_indices": record["view_indices"],
            }
            try:
                rendered = processor.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                )
                inputs = processor(text=[rendered], padding=True, return_tensors="pt")
                inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}
                input_tokens = int(inputs["input_ids"].shape[-1])
                torch.cuda.reset_peak_memory_stats(0)
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs, max_new_tokens=int(cfg.get("max_new_tokens", 256)),
                        do_sample=False, use_cache=True,
                    )
                continuation = generated[:, input_tokens:]
                prediction = processor.batch_decode(
                    continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False,
                )[0].strip()
                row.update(
                    prediction=prediction,
                    input_tokens=input_tokens,
                    output_tokens=int(continuation.shape[-1]),
                    hit_max_new_tokens=int(continuation.shape[-1]) >= int(cfg.get("max_new_tokens", 256)),
                    peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(0)),
                )
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc(limit=8)
            row["latency_sec"] = time.perf_counter() - started
            row["generated_at_unix"] = time.time()
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"rank={args.rank} {ordinal}/{len(indices)} case={idx} status={'OK' if not row['error'] else 'ERROR'}", flush=True)


if __name__ == "__main__":
    main()
