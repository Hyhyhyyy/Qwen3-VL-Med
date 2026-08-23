#!/usr/bin/env python3
"""Target-diagnosis-token visual attribution for Qwen3-VL.

The script captures the final decoder layer's attention probabilities and their
gradient under teacher forcing.  It reports bounded visual shares so values are
comparable across runs:

* attention_visual_mass: mean attention probability assigned to image tokens.
* gradient_visual_share: fraction of absolute attention-gradient on image keys.
* grad_attention_visual_share: fraction of |gradient * attention| on image keys.

Only the final text-decoder layer uses eager attention; earlier layers remain
SDPA, which keeps a 4B model within a 40 GB A100 for batch size one.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import statistics
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def find_last_subsequence(sequence: list[int], pattern: list[int]) -> int:
    if not pattern or len(pattern) > len(sequence):
        return -1
    for start in range(len(sequence) - len(pattern), -1, -1):
        if sequence[start : start + len(pattern)] == pattern:
            return start
    return -1


def diagnosis_span(text: str) -> str:
    matches = list(re.finditer(r"病理诊断\s*[:：]\s*", text))
    if not matches:
        raise ValueError("prediction does not contain a 病理诊断 label")
    return text[matches[-1].end() :].strip()


def resolve_images(row: dict, image_root: str) -> list[Path]:
    paths = []
    for raw in row.get("images", []):
        path = Path(raw)
        if not path.is_absolute():
            path = Path(image_root) / path
        paths.append(path.resolve())
    return paths


def sample_indices(n: int, count: int) -> list[int]:
    if count >= n:
        return list(range(n))
    if count == 1:
        return [n // 2]
    return sorted({round(i * (n - 1) / (count - 1)) for i in range(count)})


def summarize(rows: list[dict]) -> dict:
    valid = [row for row in rows if row.get("error") is None]
    metrics = [
        "attention_visual_mass",
        "gradient_visual_share",
        "grad_attention_visual_share",
    ]
    summary = {
        "available": bool(valid),
        "protocol": "teacher-forced predicted diagnosis tokens; final decoder layer; heads and target tokens macro-averaged; batch=1",
        "n_cases": len(valid),
        "n_errors": len(rows) - len(valid),
    }
    for metric in metrics:
        values = [float(row[metric]) for row in valid]
        summary[metric] = round(statistics.fmean(values), 6)
        summary[f"{metric}_sd"] = round(statistics.stdev(values), 6) if len(values) > 1 else 0.0
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    all_rows = [json.loads(line) for line in Path(args.predictions).read_text(encoding="utf-8").splitlines() if line.strip()]
    indices = sample_indices(len(all_rows), args.sample_count)
    if args.limit is not None:
        indices = indices[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "interpretability_cases.jsonl"
    completed = {}
    if detail_path.exists():
        for line in detail_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[int(row["source_row_index"])] = row

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
    model.requires_grad_(False)

    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    final_attention = core.model.language_model.layers[-1].self_attn
    final_attention.config = copy.deepcopy(final_attention.config)
    final_attention.config._attn_implementation = "eager"
    captured: dict[str, torch.Tensor] = {}

    def capture_attention(_module, _inputs, output):
        weights = output[1]
        if weights is None:
            raise RuntimeError("eager attention did not return attention weights")
        weights.retain_grad()
        captured["weights"] = weights

    hook = final_attention.register_forward_hook(capture_attention)
    mode = "a" if detail_path.exists() else "w"
    try:
        with detail_path.open(mode, encoding="utf-8") as output_file:
            for ordinal, source_index in enumerate(indices, start=1):
                if source_index in completed and completed[source_index].get("error") is None:
                    print(f"{ordinal}/{len(indices)} source={source_index} SKIP", flush=True)
                    continue
                source = all_rows[source_index]
                case_index = int(source.get("case_index", source_index))
                result = {"source_row_index": source_index, "case_index": case_index, "error": None}
                images = []
                started = time.perf_counter()
                try:
                    prediction = str(source["prediction"]).strip()
                    target_text = diagnosis_span(prediction)
                    image_paths = resolve_images(source, cfg["image_root"])
                    missing = [str(path) for path in image_paths if not path.is_file()]
                    if missing:
                        raise FileNotFoundError(f"missing image(s): {missing}")
                    images = [Image.open(path).convert("RGB") for path in image_paths]
                    clean_prompt = str(source["prompt"]).replace("<image>", "").strip()
                    user_content = ([{"type": "image"} for _ in images]
                                    + [{"type": "text", "text": clean_prompt}])
                    conversation = [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": [{"type": "text", "text": prediction}]},
                    ]
                    rendered = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
                    inputs = processor(text=[rendered], images=images, padding=True, return_tensors="pt")
                    input_ids_cpu = inputs["input_ids"][0]
                    target_ids = processor.tokenizer.encode(target_text, add_special_tokens=False)
                    target_start = find_last_subsequence(input_ids_cpu.tolist(), target_ids)
                    if target_start <= 0:
                        raise ValueError("diagnosis token sequence not found in rendered conversation")
                    query_positions = torch.arange(target_start - 1, target_start + len(target_ids) - 1, device="cuda:0")
                    target_tensor = torch.tensor(target_ids, device="cuda:0")
                    visual_mask = input_ids_cpu.eq(core.config.image_token_id).to("cuda:0")
                    if not visual_mask.any():
                        raise ValueError("no image placeholder tokens found")

                    inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}
                    inputs["pixel_values"].requires_grad_(True)
                    captured.clear()
                    torch.cuda.reset_peak_memory_stats(0)
                    outputs = model(**inputs, use_cache=False, logits_to_keep=query_positions)
                    log_probs = torch.log_softmax(outputs.logits[0].float(), dim=-1)
                    score = log_probs.gather(1, target_tensor[:, None]).mean()
                    score.backward()

                    attention = captured["weights"][0].float()
                    gradient = captured["weights"].grad[0].float().abs()
                    selected_attention = attention[:, query_positions, :]
                    selected_gradient = gradient[:, query_positions, :]
                    key_positions = torch.arange(attention.shape[-1], device=attention.device)
                    causal = key_positions[None, :] <= query_positions[:, None]
                    valid = causal[None, :, :]
                    visual = visual_mask[None, None, :]

                    attention_mass = selected_attention.masked_fill(~visual, 0).sum(-1).mean()
                    gradient_valid = selected_gradient.masked_fill(~valid, 0)
                    gradient_visual = gradient_valid.masked_fill(~visual, 0)
                    gradient_share = gradient_visual.sum(-1) / gradient_valid.sum(-1).clamp_min(1e-12)
                    grad_attention = (selected_gradient * selected_attention).masked_fill(~valid, 0)
                    grad_attention_visual = grad_attention.masked_fill(~visual, 0)
                    grad_attention_share = grad_attention_visual.sum(-1) / grad_attention.sum(-1).clamp_min(1e-12)

                    image_token_counts = (inputs["image_grid_thw"].prod(-1)
                                          // core.model.visual.spatial_merge_size**2).tolist()
                    flat_visual_attn = selected_attention[:, :, visual_mask].mean(dim=(0, 1))
                    flat_visual_ga = grad_attention_visual[:, :, visual_mask].mean(dim=(0, 1))
                    per_image_attention = []
                    per_image_grad_attention = []
                    offset = 0
                    for count in image_token_counts:
                        per_image_attention.append(float(flat_visual_attn[offset : offset + count].sum().item()))
                        per_image_grad_attention.append(float(flat_visual_ga[offset : offset + count].sum().item()))
                        offset += count
                    ga_total = sum(per_image_grad_attention)
                    if ga_total > 0:
                        per_image_grad_attention = [value / ga_total for value in per_image_grad_attention]

                    result.update(
                        target_text=target_text,
                        target_tokens=len(target_ids),
                        visual_tokens=int(visual_mask.sum().item()),
                        attention_visual_mass=float(attention_mass.item()),
                        gradient_visual_share=float(gradient_share.mean().item()),
                        grad_attention_visual_share=float(grad_attention_share.mean().item()),
                        per_image_attention_mass=per_image_attention,
                        per_image_grad_attention_share=per_image_grad_attention,
                        target_mean_log_probability=float(score.item()),
                        peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(0)),
                    )
                    del outputs, log_probs, score, attention, gradient, selected_attention, selected_gradient, inputs
                    captured.clear()
                    torch.cuda.empty_cache()
                except Exception as exc:  # noqa: BLE001
                    result["error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    for image in images:
                        image.close()
                    result["latency_sec"] = time.perf_counter() - started
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_file.flush()
                    print(f"{ordinal}/{len(indices)} source={source_index} {'OK' if result['error'] is None else result['error']} latency={result['latency_sec']:.2f}s", flush=True)
    finally:
        hook.remove()

    latest = {}
    for line in detail_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[int(row["source_row_index"])] = row
    selected_rows = [latest[index] for index in indices]
    summary = summarize(selected_rows)
    (output_dir / "interpretability.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
