#!/usr/bin/env python3
"""Archive full token-to-visual attribution arrays and publication plots.

This module is intentionally independent of model loading. A Qwen3-VL
attribution extractor passes the tensors and image-token layout to
``archive_case``. Clinical images and generated archives must remain in a
controlled output directory outside this public repository.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


SCHEMA_VERSION = "phase4-interpretability-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_uint8(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.all():
        raise ValueError("heatmap contains NaN or Inf")
    low = float(array.min(initial=0.0))
    high = float(array.max(initial=0.0))
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.rint((array - low) * (255.0 / (high - low))).astype(np.uint8)


def colorize(values: np.ndarray) -> Image.Image:
    level = normalized_uint8(values)
    # Compact blue -> cyan -> yellow -> red map without a plotting dependency.
    x = level.astype(np.float32) / 255.0
    red = np.clip(2.0 * x, 0.0, 1.0)
    green = np.clip(2.0 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - x), 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(np.rint(rgb * 255).astype(np.uint8), mode="RGB")


def save_heatmap(values: np.ndarray, path: Path, scale: int = 8) -> None:
    image = colorize(values)
    width = max(image.width * scale, image.width)
    height = max(image.height * scale, image.height)
    image.resize((width, height), resample=Image.Resampling.NEAREST).save(path)


def build_visual_mapping(images: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    image_indices: list[int] = []
    grid_tyx: list[tuple[int, int, int]] = []
    boxes: list[tuple[float, float, float, float]] = []
    sequence_positions: list[int] = []
    expected_start = 0
    for image_index, record in enumerate(images):
        start = int(record["vision_token_start"])
        end = int(record["vision_token_end_exclusive"])
        grid_t = int(record.get("token_grid_t", 1))
        grid_h = int(record["token_grid_h"])
        grid_w = int(record["token_grid_w"])
        width = int(record["original_width"])
        height = int(record["original_height"])
        if start != expected_start:
            raise ValueError(f"visual token ranges must be contiguous: {start} != {expected_start}")
        if end - start != grid_t * grid_h * grid_w:
            raise ValueError("visual token range does not match token grid")
        for local_index in range(end - start):
            t, remainder = divmod(local_index, grid_h * grid_w)
            y, x = divmod(remainder, grid_w)
            image_indices.append(image_index)
            grid_tyx.append((t, y, x))
            boxes.append((
                x * width / grid_w,
                y * height / grid_h,
                (x + 1) * width / grid_w,
                (y + 1) * height / grid_h,
            ))
            sequence_positions.append(int(record.get("sequence_start", start)) + local_index)
        expected_start = end
    return {
        "visual_token_image_index": np.asarray(image_indices, dtype=np.int16),
        "visual_token_grid_tyx": np.asarray(grid_tyx, dtype=np.int16),
        "visual_token_bbox_original_xyxy": np.asarray(boxes, dtype=np.float32),
        "visual_token_sequence_positions": np.asarray(sequence_positions, dtype=np.int32),
    }


def validate_arrays(attention_heads: np.ndarray, abs_gradient: np.ndarray) -> tuple[int, int, int]:
    if attention_heads.ndim != 3:
        raise ValueError("attention_heads must have shape [target, head, visual]")
    target, heads, visual = attention_heads.shape
    if abs_gradient.shape != (target, visual):
        raise ValueError("abs_gradient must have shape [target, visual]")
    if target < 1 or heads < 1 or visual < 1:
        raise ValueError("attribution arrays cannot have empty dimensions")
    if not np.isfinite(attention_heads).all() or not np.isfinite(abs_gradient).all():
        raise ValueError("attribution arrays contain NaN or Inf")
    if (attention_heads < 0).any() or (abs_gradient < 0).any():
        raise ValueError("attention and absolute gradient must be non-negative")
    return target, heads, visual


def archive_case(
    output_root: Path,
    *,
    run_id: str,
    case_uid: str,
    comparison_group: str,
    target_span_id: str,
    target_text: str,
    target_token_ids: np.ndarray,
    target_token_texts: list[str],
    attention_heads: np.ndarray,
    abs_gradient: np.ndarray,
    images: list[dict[str, Any]],
    target_char_offsets: np.ndarray | None = None,
    attention_link_abs_gradient_heads: np.ndarray | None = None,
    decoder_layer: int = 35,
    aggregation: str = "mean_heads",
) -> dict[str, Any]:
    attention_heads = np.asarray(attention_heads, dtype=np.float32)
    abs_gradient = np.asarray(abs_gradient, dtype=np.float32)
    target, heads, visual = validate_arrays(attention_heads, abs_gradient)
    target_token_ids = np.asarray(target_token_ids, dtype=np.int32)
    if target_token_ids.shape != (target,) or len(target_token_texts) != target:
        raise ValueError("target token metadata does not match target dimension")
    mapping = build_visual_mapping(images)
    if mapping["visual_token_image_index"].shape != (visual,):
        raise ValueError("image mapping does not cover the visual dimension")
    if attention_link_abs_gradient_heads is None:
        attention_link_abs_gradient_heads = np.zeros_like(attention_heads)
    attention_link_abs_gradient_heads = np.asarray(attention_link_abs_gradient_heads, dtype=np.float32)
    if attention_link_abs_gradient_heads.shape != attention_heads.shape:
        raise ValueError("attention-link gradient must match attention_heads")

    attention = attention_heads.mean(axis=1)
    grad_x_attention_heads = abs_gradient[:, None, :] * attention_heads
    grad_x_attention = abs_gradient * attention
    arrays: dict[str, np.ndarray] = {
        "attention_heads": attention_heads.astype(np.float16),
        "attention_link_abs_gradient_heads": attention_link_abs_gradient_heads.astype(np.float16),
        "grad_x_attention_heads": grad_x_attention_heads.astype(np.float16),
        "attention": attention.astype(np.float16),
        "abs_gradient": abs_gradient.astype(np.float16),
        "grad_x_attention": grad_x_attention.astype(np.float16),
        "target_token_ids": target_token_ids,
        "target_char_offsets": (
            np.asarray(target_char_offsets, dtype=np.int32)
            if target_char_offsets is not None
            else np.full((target, 2), -1, dtype=np.int32)
        ),
        **mapping,
    }

    case_dir = Path(output_root) / run_id / case_uid / target_span_id
    case_dir.mkdir(parents=True, exist_ok=True)
    npz_path = case_dir / "attribution.float16.npz"
    np.savez_compressed(npz_path, **arrays)

    save_heatmap(attention, case_dir / "target_by_visual_attention.png")
    save_heatmap(abs_gradient, case_dir / "target_by_visual_abs_gradient.png")
    save_heatmap(grad_x_attention, case_dir / "target_by_visual_grad_x_attention.png")
    for image_index, record in enumerate(images):
        start = int(record["vision_token_start"])
        end = int(record["vision_token_end_exclusive"])
        grid_t = int(record.get("token_grid_t", 1))
        grid_h = int(record["token_grid_h"])
        grid_w = int(record["token_grid_w"])
        spatial = grad_x_attention[:, start:end].mean(axis=0).reshape(grid_t, grid_h, grid_w).mean(axis=0)
        save_heatmap(spatial, case_dir / f"image_{image_index:02d}_grad_x_attention.png", scale=24)
        for target_index in range(target_token_ids.shape[0]):
            token_prefix = f"target_{target_index:03d}_image_{image_index:02d}"
            token_attention = attention[target_index, start:end].reshape(grid_t, grid_h, grid_w).mean(axis=0)
            token_gradient = abs_gradient[target_index, start:end].reshape(grid_t, grid_h, grid_w).mean(axis=0)
            token_product = grad_x_attention[target_index, start:end].reshape(grid_t, grid_h, grid_w).mean(axis=0)
            save_heatmap(token_attention, case_dir / f"{token_prefix}_attention.png", scale=24)
            save_heatmap(token_gradient, case_dir / f"{token_prefix}_abs_gradient.png", scale=24)
            save_heatmap(token_product, case_dir / f"{token_prefix}_grad_x_attention.png", scale=24)
    head_matrix = attention_heads.mean(axis=0)
    save_heatmap(head_matrix, case_dir / "all_heads_by_visual_attention.png", scale=4)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "case_uid": case_uid,
        "comparison_group": comparison_group,
        "target_span_id": target_span_id,
        "target_text": target_text,
        "target_token_texts": target_token_texts,
        "target_token_ids": target_token_ids.tolist(),
        "decoder_layer": decoder_layer,
        "attention_heads": list(range(heads)),
        "attention_implementation": "eager",
        "matrix_dtype": "float16",
        "aggregation": aggregation,
        "npz_path": npz_path.name,
        "npz_sha256": sha256(npz_path),
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()},
        "images": images,
        "qc": {
            "has_nan": False,
            "has_inf": False,
            "shape_match": True,
            "token_mapping_complete": True,
            "spatial_mapping_complete": True,
        },
    }
    manifest_path = case_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


__all__ = ["SCHEMA_VERSION", "archive_case", "build_visual_mapping", "validate_arrays"]
