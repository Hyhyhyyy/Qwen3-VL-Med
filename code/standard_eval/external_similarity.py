#!/usr/bin/env python3
"""External embedding cosine similarity between prediction and reference.

Two optional encoders:
  - GPT-3-ADA (OpenAI): needs OPENAI_API_KEY + `openai` package + network.
  - BioBERT / Chinese encoder (local HF): needs the model downloadable or a
    local path (env BIOBERT_MODEL, default bert-base-chinese).

If an encoder is unavailable, the corresponding field is left as `available:False`
with a reason, and sheet_export will emit "需补：…" for that column. This keeps
the pipeline runnable offline; you opt into external similarity by providing
credentials / a local model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def load_pairs(out_dir: Path):
    pred_path = out_dir / "predictions.jsonl"
    rows = [json.loads(l) for l in pred_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    pairs = []
    for r in rows:
        pred = (r.get("prediction") or "").strip()
        ref = (r.get("reference") or "").strip()
        if pred and ref:
            pairs.append((pred, ref))
    return pairs


def run_openai(pairs, model="text-embedding-3-small"):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def embed(texts):
        resp = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]

    sims = []
    batch = 32
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        preds = [p[0] for p in chunk]
        refs = [p[1] for p in chunk]
        pe = embed(preds)
        re = embed(refs)
        sims.extend(cosine(a, b) for a, b in zip(pe, re))
    return sum(sims) / len(sims)


def run_local(pairs, model_name):
    from transformers import AutoModel, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=False, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(model_name, local_files_only=False, trust_remote_code=True)
    mdl = mdl.to("cuda:0" if torch.cuda.is_available() else "cpu").eval()

    def embed(texts):
        out = []
        for t in texts:
            inp = tok(t, return_tensors="pt", truncation=True, max_length=512,
                      padding=True).to(mdl.device)
            with torch.inference_mode():
                h = mdl(**inp).last_hidden_state[:, 0, :].float().cpu().tolist()[0]
            out.append(h)
        return out

    sims = []
    batch = 16
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        pe = embed([p[0] for p in chunk])
        re = embed([p[1] for p in chunk])
        sims.extend(cosine(a, b) for a, b in zip(pe, re))
    return sum(sims) / len(sims)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.load(open(args.config, "r", encoding="utf-8"))
    out_dir = Path(cfg["output_dir"])
    if not (out_dir / "predictions.jsonl").is_file():
        raise SystemExit(f"缺少 {out_dir / 'predictions.jsonl'}")
    pairs = load_pairs(out_dir)
    if not pairs:
        raise SystemExit("predictions.jsonl 没有可用的 (prediction, reference) 对")

    result = {
        "gpt3_ada_similarity": None, "gpt3_ada_available": False, "gpt3_ada_reason": None,
        "biobert_similarity": None, "biobert_available": False, "biobert_reason": None,
        "n_pairs": len(pairs),
    }

    # GPT-3-ADA
    if os.environ.get("OPENAI_API_KEY"):
        try:
            result["gpt3_ada_similarity"] = round(run_openai(pairs), 6)
            result["gpt3_ada_available"] = True
        except Exception as exc:  # noqa: BLE001
            result["gpt3_ada_reason"] = f"运行失败：{type(exc).__name__}: {exc}"
    else:
        result["gpt3_ada_reason"] = "未设置 OPENAI_API_KEY（可选；不设则跳过，指标标「需补」）"

    # BioBERT / local encoder
    biobert_model = os.environ.get("BIOBERT_MODEL", "bert-base-chinese")
    try:
        result["biobert_similarity"] = round(run_local(pairs, biobert_model), 6)
        result["biobert_available"] = True
        result["biobert_model"] = biobert_model
    except Exception as exc:  # noqa: BLE001
        result["biobert_reason"] = f"运行失败（可能需下载或 CUDA 不足）：{type(exc).__name__}: {exc}"

    (out_dir / "external_similarity.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
