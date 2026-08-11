#!/usr/bin/env python3
"""Compute reproducible lexical, diagnostic and pathology-fact metrics.

Clinical hallucination/omission values produced here are reference-based proxies.
They do not establish visual truth and must not be presented as expert-validated harm rates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

try:
    import jieba
except Exception:
    jieba = None

try:
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
    from nltk.translate.chrf_score import corpus_chrf, sentence_chrf
except Exception:
    corpus_bleu = sentence_bleu = corpus_chrf = sentence_chrf = None

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_recall_fscore_support,
    )
except Exception:
    accuracy_score = balanced_accuracy_score = cohen_kappa_score = None
    confusion_matrix = f1_score = matthews_corrcoef = precision_recall_fscore_support = None


DIAG_RE = re.compile(r"病理诊断\s*[：:]\s*(.+)", re.S)
IMAGE_LABEL_RE = re.compile(r"(?:左上|右上|左下|右下|上|下)图")
ALL_POSITION_TOKEN_RE = re.compile(r"(?:(?:左|中|右)(?:上|中|下)|上|下)图")
STRIP_POSITION_TOKENS_FOR_SCORING = False
NEGATIONS = ("未见", "无明显", "无", "未", "不伴", "阴性", "未找到", "不能确定", "不支持")
UNCERTAINTY = ("考虑", "倾向", "可能", "疑似", "建议结合", "不能排除", "尚不能")
SEVERITIES = ("极重度", "重度", "中重度", "中度", "轻中度", "轻度", "少量", "少数", "偶见", "明显")
KEY_ENTITIES = {
    "interface_hepatitis",
    "lobular_inflammation",
    "portal_inflammation",
    "spotty_necrosis",
    "steatosis",
    "fibrosis",
    "bile_duct_injury",
    "ductular_reaction",
    "granuloma",
    "ballooning",
    "cholestasis",
}

# Longest expressions should appear first inside each group.
ENTITY_TERMS: dict[str, tuple[str, ...]] = {
    "portal_tract": ("中小汇管区", "汇管区"),
    "lobular_architecture": ("小叶结构",),
    "portal_inflammation": ("汇管区炎性细胞浸润", "汇管区炎症", "炎性细胞浸润"),
    "interface_hepatitis": ("界面性肝炎", "界面炎"),
    "lobular_inflammation": ("小叶性肝炎", "小叶内炎症", "小叶炎"),
    "spotty_necrosis": ("点灶状坏死", "灶状坏死", "点状坏死"),
    "bridging_necrosis": ("桥接坏死", "桥接性坏死"),
    "steatosis": ("大泡性脂变", "小泡性脂变", "脂肪性变", "脂肪变", "脂变"),
    "macrovesicular_steatosis": ("大泡性脂变", "大泡脂变"),
    "microvesicular_steatosis": ("小泡性脂变", "小泡脂变"),
    "ballooning": ("气球样变", "气球样改变"),
    "bile_duct": ("小胆管", "胆管"),
    "bile_duct_injury": ("胆管上皮损伤", "胆管损伤", "攻击小胆管", "胆管上皮排列不整"),
    "ductular_reaction": ("细胆管反应", "胆管反应"),
    "granuloma": ("上皮样肉芽肿", "肉芽肿"),
    "fibrosis": ("纤维组织增生", "纤维化", "纤维间隔"),
    "cholestasis": ("胆汁淤积", "淤胆"),
    "plasma_cells": ("浆细胞浸润", "浆细胞"),
    "lymphocyte_aggregate": ("淋巴细胞聚集灶", "淋巴细胞聚集"),
    "sinusoidal_inflammation": ("肝窦内炎症", "窦内炎症", "窦内轻度单个核细胞浸润"),
    "hepatocyte_rosette": ("肝细胞玫瑰花环", "玫瑰花环"),
    "mallory_bodies": ("Mallory-Denk小体", "Mallory小体", "马洛里小体"),
    "iron_deposition": ("含铁血黄素", "铁沉积"),
    "copper_deposition": ("铜沉积", "铜相关蛋白"),
}

DIAG_SYNONYMS = {
    "pbc": "原发性胆汁性胆管炎",
    "原发性胆汁性肝硬化": "原发性胆汁性胆管炎",
    "nash": "非酒精性脂肪性肝炎",
    "nafld": "非酒精性脂肪性肝病",
    "aiH": "自身免疫性肝炎",
    "aih": "自身免疫性肝炎",
    "psc": "原发性硬化性胆管炎",
}

DIAG_CONCEPTS: dict[str, tuple[str, ...]] = {
    "unanswerable_placeholder": ("未找到",),
    "primary_biliary_cholangitis": ("原发性胆汁性胆管炎", "原发性胆汁性肝硬化", "pbc"),
    "autoimmune_hepatitis": ("自身免疫性肝炎", "自身免疫样肝炎", "aih", "di-alh"),
    "steatohepatitis": ("非酒精性脂肪性肝炎", "代谢相关性脂肪性肝炎", "代谢相关（非酒精性）脂肪性肝炎", "nash", "mash"),
    "fatty_liver_or_steatosis": ("脂肪肝", "脂肪性肝病", "肝细胞脂变", "单纯性脂肪", "脂肪性肝炎"),
    "drug_induced_liver_injury": ("药物性肝损伤", "药物诱导", "药物或化学物性肝损伤", "药物/化学物性肝损伤", "dili"),
    "primary_sclerosing_cholangitis": ("原发性硬化性胆管炎", "小胆管型硬化性胆管炎", "硬化性胆管炎", "psc"),
    "hepatitis_b": ("乙型肝炎", "乙型病毒性肝炎", "乙型病毒型肝炎"),
    "transplant_rejection": ("排斥", "排异"),
    "portal_sinusoidal_vascular_disease": ("门脉肝窦血管病", "非肝硬化性门脉高压", "肝门脉硬化症", "窦前性门脉高压", "psvd"),
    "cirrhosis": ("肝硬化",),
    "fibrosis": ("肝纤维化", "纤维化"),
    "cholestatic_hepatitis": ("淤胆性肝炎", "淤胆型肝炎", "胆汁淤积性肝炎"),
    "cholestasis": ("胆汁淤积", "淤胆"),
    "bile_duct_loss": ("胆管消失综合征", "胆管缺失", "胆管减少"),
    "wilson_disease": ("肝豆状核变性", "wilson"),
    "hepatic_outflow_obstruction": ("布加综合征", "budd-chiari", "肝静脉回流障碍", "淤血性肝"),
    "lobular_hepatitis": ("小叶性肝炎", "小叶性肝损伤", "小叶性肝病"),
    "acute_hepatitis": ("急性肝炎", "急性重型肝炎"),
    "granulomatous_or_sarcoidosis": ("肉芽肿性肝炎", "结节病"),
    "alcohol_related_liver_disease": ("酒精性肝",),
    "iron_overload": ("铁过载", "铁沉积", "含铁血黄素"),
    "biliary_obstruction": ("大胆管梗阻", "大胆管不全梗阻", "胆道并发症"),
    "congenital_hepatic_fibrosis": ("先天性肝纤维化",),
    "pfic": ("pfic",),
    "alagille_syndrome": ("阿拉吉利综合征",),
    "amyloidosis": ("淀粉样变",),
    "vascular_neoplasm": ("上皮样血管内皮瘤",),
}


def normalize_text(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    # Preserve raw generations in predictions.jsonl, but exclude hidden-thought
    # wrappers from report-quality scoring.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    if STRIP_POSITION_TOKENS_FOR_SCORING:
        text = ALL_POSITION_TOKEN_RE.sub("", text)
        text = re.sub(r"[（(]\s*[）)]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compact(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(text)).lower()


def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    if jieba is not None:
        tokens = [x.strip() for x in jieba.lcut(text) if x.strip()]
    else:
        tokens = re.findall(r"[\u4e00-\u9fff]|[A-Za-z]+|\d+(?:\.\d+)?|[^\s]", text)
    return tokens


def char_tokens(text: str) -> list[str]:
    return [c for c in compact(text) if not c.isspace()]


def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1)))


def prf(overlap: int, predicted: int, reference: int) -> tuple[float, float, float]:
    precision = overlap / predicted if predicted else (1.0 if reference == 0 else 0.0)
    recall = overlap / reference if reference else (1.0 if predicted == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def set_prf(reference: set, predicted: set) -> tuple[float, float, float]:
    return prf(len(reference & predicted), len(predicted), len(reference))


def rouge_n(reference: list[str], predicted: list[str], n: int) -> tuple[float, float, float]:
    ref_counts, pred_counts = ngrams(reference, n), ngrams(predicted, n)
    overlap = sum((ref_counts & pred_counts).values())
    return prf(overlap, sum(pred_counts.values()), sum(ref_counts.values()))


def lcs_length(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if x == y else max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(reference: list[str], predicted: list[str]) -> tuple[float, float, float]:
    return prf(lcs_length(reference, predicted), len(predicted), len(reference))


def meteor_exact(reference: list[str], predicted: list[str]) -> float:
    """Chinese exact-token METEOR variant without English WordNet synonym expansion."""
    if not reference and not predicted:
        return 1.0
    ref_counts, pred_counts = Counter(reference), Counter(predicted)
    matches = sum((ref_counts & pred_counts).values())
    if not matches:
        return 0.0
    p, r = matches / max(1, len(predicted)), matches / max(1, len(reference))
    f_mean = (10 * p * r) / (r + 9 * p) if r + 9 * p else 0.0
    # Conservative chunk approximation based on contiguous exact matches.
    positions: dict[str, list[int]] = defaultdict(list)
    for i, token in enumerate(reference):
        positions[token].append(i)
    used: set[int] = set()
    aligned: list[int] = []
    for token in predicted:
        candidates = [i for i in positions.get(token, []) if i not in used]
        if candidates:
            chosen = candidates[0]
            used.add(chosen)
            aligned.append(chosen)
    chunks = 0
    last = None
    for pos in aligned:
        if last is None or pos != last + 1:
            chunks += 1
        last = pos
    penalty = 0.5 * (chunks / matches) ** 3
    return f_mean * (1 - penalty)


def extract_diagnosis(text: str) -> str:
    match = DIAG_RE.search(normalize_text(text))
    if not match:
        return ""
    value = match.group(1).splitlines()[0].strip()
    value = re.sub(r"[。；;].*$", "", value)
    return value.strip()


def canonical_diagnosis(text: str) -> str:
    value = extract_diagnosis(text) if "病理诊断" in text else normalize_text(text)
    value = value.lower()
    value = re.sub(r"[（(][^）)]*(?:期|级|ama|评分)[^）)]*[）)]", "", value, flags=re.I)
    value = re.sub(r"(?:分期|stage)?\s*[ivxⅠⅡⅢⅣ一二三四0-4]+\s*期", "", value, flags=re.I)
    value = re.sub(r"^(?:考虑|倾向|符合|诊断为|提示)\s*", "", value)
    for source, target in DIAG_SYNONYMS.items():
        value = value.replace(source.lower(), target)
    value = re.sub(r"[\s，,。；;：:、]+", "", value)
    return value or "__MISSING__"


def diagnosis_terms(text: str) -> set[str]:
    value = canonical_diagnosis(text)
    if value == "__MISSING__":
        return set()
    parts = re.split(r"(?:并|伴|合并|及|/|\+)", value)
    return {x for x in parts if x}


def diagnosis_concepts(text: str) -> set[str]:
    value = compact(extract_diagnosis(text))
    if not value:
        return set()
    found = set()
    for concept, terms in DIAG_CONCEPTS.items():
        if any(term.lower() in value for term in terms):
            found.add(concept)
    return found


def local_context(text: str, start: int, end: int, radius: int = 14) -> str:
    left_boundary = max(text.rfind("\n", 0, start), text.rfind("。", 0, start), text.rfind("；", 0, start))
    right_candidates = [x for x in (text.find("\n", end), text.find("。", end), text.find("；", end)) if x >= 0]
    right_boundary = min(right_candidates) if right_candidates else len(text)
    return text[max(left_boundary + 1, start - radius) : min(right_boundary, end + radius)]


def extract_entities(text: str) -> dict[str, dict]:
    text = normalize_text(text)
    result: dict[str, dict] = {}
    for entity, terms in ENTITY_TERMS.items():
        best = None
        for term in terms:
            match = re.search(re.escape(term), text, re.I)
            if match and (best is None or len(term) > len(best[0])):
                best = (term, match.start(), match.end())
        if best is None:
            continue
        term, start, end = best
        ctx = local_context(text, start, end)
        before = text[max(0, start - 10) : start]
        status = "absent" if any(n in before or n + term in ctx for n in NEGATIONS) else "present"
        severity = next((s for s in SEVERITIES if s in ctx), None)
        uncertain = any(x in ctx for x in UNCERTAINTY)
        result[entity] = {
            "status": status,
            "severity": severity,
            "uncertain": uncertain,
            "term": term,
            "context": ctx,
        }
    return result


def fact_set(text: str) -> set[str]:
    entities = extract_entities(text)
    facts = {f"entity:{key}:{value['status']}" for key, value in entities.items()}
    facts.update(f"diagnosis:{x}" for x in diagnosis_terms(text))
    return facts


def entity_set(text: str) -> set[str]:
    return set(extract_entities(text))


def relation_proxy_set(text: str) -> set[str]:
    result = set()
    for entity, attrs in extract_entities(text).items():
        result.add(f"{entity}|status={attrs['status']}")
        if attrs["severity"]:
            result.add(f"{entity}|severity={attrs['severity']}")
        if attrs["uncertain"]:
            result.add(f"{entity}|uncertain=true")
    return result


def key_fact_set(text: str) -> set[str]:
    entities = extract_entities(text)
    facts = {
        f"entity:{key}:{value['status']}"
        for key, value in entities.items()
        if key in KEY_ENTITIES
    }
    facts.update(f"diagnosis:{x}" for x in diagnosis_terms(text))
    return facts


def contradiction_stats(reference: str, prediction: str) -> tuple[int, int, int]:
    ref, pred = extract_entities(reference), extract_entities(prediction)
    common = set(ref) & set(pred)
    contradictions = sum(ref[k]["status"] != pred[k]["status"] for k in common)
    negation_flips = contradictions
    denominator = len(set(ref) | set(pred))
    return contradictions, negation_flips, denominator


def severity_stats(reference: str, prediction: str) -> tuple[int, int]:
    ref, pred = extract_entities(reference), extract_entities(prediction)
    comparable = [k for k in set(ref) & set(pred) if ref[k]["severity"] and pred[k]["severity"]]
    correct = sum(ref[k]["severity"] == pred[k]["severity"] for k in comparable)
    return correct, len(comparable)


NUMBER_RE = re.compile(r"(?<![A-Za-z\d])([0-9]+(?:\.[0-9]+)?)\s*(%|个|期|级)?")


def numeric_claims(text: str) -> list[dict]:
    text = normalize_text(text)
    claims = []
    for match in NUMBER_RE.finditer(text):
        value = float(match.group(1))
        unit = match.group(2) or "number"
        ctx = local_context(text, match.start(), match.end(), radius=18)
        entity = None
        for key, terms in ENTITY_TERMS.items():
            if any(term in ctx for term in terms):
                entity = key
                break
        if entity is None and unit == "期":
            entity = "stage"
        claims.append({"entity": entity or "unscoped", "value": value, "unit": unit, "context": ctx})
    return claims


ROMAN_STAGE = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "Ⅰ": 1, "Ⅱ": 2, "Ⅲ": 3, "Ⅳ": 4, "一": 1, "二": 2, "三": 3, "四": 4}


def extract_stage(text: str) -> int | None:
    match = re.search(r"([ivxⅠⅡⅢⅣ一二三四1-4]+)\s*期", normalize_text(text), re.I)
    if not match:
        return None
    raw = match.group(1)
    if raw.isdigit():
        return int(raw)
    return ROMAN_STAGE.get(raw.lower(), ROMAN_STAGE.get(raw))


def compare_numbers(reference: str, prediction: str) -> dict:
    ref, pred = numeric_claims(reference), numeric_claims(prediction)
    used: set[int] = set()
    errors = []
    exact = 0
    within5 = 0
    within10 = 0
    for r in ref:
        candidates = [
            (i, p)
            for i, p in enumerate(pred)
            if i not in used and p["entity"] == r["entity"] and p["unit"] == r["unit"]
        ]
        if not candidates:
            continue
        i, p = min(candidates, key=lambda x: abs(x[1]["value"] - r["value"]))
        used.add(i)
        error = abs(p["value"] - r["value"])
        errors.append(error)
        exact += error == 0
        within5 += error <= 5
        within10 += error <= 10
    return {
        "reference_count": len(ref),
        "prediction_count": len(pred),
        "matched_count": len(errors),
        "exact_count": exact,
        "within5_count": within5,
        "within10_count": within10,
        "absolute_errors": errors,
        "unsupported_count": max(0, len(pred) - len(used)),
        "omitted_count": max(0, len(ref) - len(errors)),
    }


def image_labels(text: str) -> set[str]:
    return set(IMAGE_LABEL_RE.findall(normalize_text(text)))


def repetition_rate(tokens: list[str], n: int = 3) -> float:
    grams = list(ngrams(tokens, n).elements())
    if not grams:
        return 0.0
    return 1 - len(set(grams)) / len(grams)


def safe_mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return statistics.fmean(vals) if vals else None


def bootstrap_ci(values: list[float], samples: int, seed: int) -> dict | None:
    values = [float(x) for x in values if x is not None and not math.isnan(float(x))]
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    estimates = [statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    estimates.sort()
    lo = estimates[max(0, int(0.025 * samples) - 1)]
    hi = estimates[min(samples - 1, int(0.975 * samples))]
    return {"mean": statistics.fmean(values), "ci95_low": lo, "ci95_high": hi, "n": n}


def bootstrap_callable(n: int, fn: Callable[[list[int]], float], samples: int, seed: int) -> dict:
    rng = random.Random(seed)
    point = float(fn(list(range(n))))
    vals = [float(fn([rng.randrange(n) for _ in range(n)])) for _ in range(samples)]
    vals.sort()
    return {
        "value": point,
        "ci95_low": vals[max(0, int(0.025 * samples) - 1)],
        "ci95_high": vals[min(samples - 1, int(0.975 * samples))],
        "n": n,
    }


def cider_lite(references: list[list[str]], predictions: list[list[str]]) -> float:
    """Single-reference CIDEr-style TF-IDF cosine over 1-4 grams, scaled by 10."""
    n_docs = len(references)
    if not n_docs:
        return 0.0
    dfs = {n: Counter() for n in range(1, 5)}
    for tokens in references:
        for n in range(1, 5):
            dfs[n].update(set(ngrams(tokens, n)))
    scores = []
    for ref, pred in zip(references, predictions):
        per_n = []
        for n in range(1, 5):
            rc, pc = ngrams(ref, n), ngrams(pred, n)
            keys = set(rc) | set(pc)
            rv, pv = {}, {}
            for key in keys:
                idf = math.log((n_docs + 1) / (dfs[n][key] + 1)) + 1
                rv[key] = rc[key] * idf
                pv[key] = pc[key] * idf
            dot = sum(rv[k] * pv[k] for k in keys)
            rn = math.sqrt(sum(x * x for x in rv.values()))
            pn = math.sqrt(sum(x * x for x in pv.values()))
            per_n.append(dot / (rn * pn) if rn and pn else 0.0)
        scores.append(10 * statistics.fmean(per_n))
    return statistics.fmean(scores)


def classification_metrics(y_true: list[str], y_pred: list[str], samples: int, seed: int) -> tuple[dict, list[list]]:
    labels = sorted(set(y_true) | set(y_pred))
    if f1_score is None:
        exact = safe_mean([a == b for a, b in zip(y_true, y_pred)]) or 0.0
        return {"accuracy": exact, "sklearn_metrics_available": False}, []

    def macro(indices: list[int]) -> float:
        return float(f1_score([y_true[i] for i in indices], [y_pred[i] for i in indices], average="macro", zero_division=0))

    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    per_p, per_r, per_f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f_macro),
        "micro_precision": float(p_micro),
        "micro_recall": float(r_micro),
        "micro_f1": float(f_micro),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_f1_ci95": bootstrap_callable(len(y_true), macro, samples, seed),
        "per_diagnosis": {
            label: {
                "precision": float(per_p[i]),
                "recall": float(per_r[i]),
                "f1": float(per_f[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
    }
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return result, [["reference\\prediction", *labels], *[[labels[i], *row] for i, row in enumerate(matrix)]]


def main() -> None:
    global STRIP_POSITION_TOKENS_FOR_SCORING
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.load(open(args.config, "r", encoding="utf-8"))
    STRIP_POSITION_TOKENS_FOR_SCORING = bool(cfg.get("strip_position_tokens_for_scoring", False))
    out_dir = Path(cfg["output_dir"])
    prediction_path = out_dir / "predictions.jsonl"
    rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("no predictions found")

    references = [normalize_text(x["reference"]) for x in rows]
    predictions = [normalize_text(x["prediction"]) for x in rows]
    ref_tokens = [tokenize(x) for x in references]
    pred_tokens = [tokenize(x) for x in predictions]
    ref_chars = [char_tokens(x) for x in references]
    pred_chars = [char_tokens(x) for x in predictions]
    bootstrap_samples = int(cfg.get("bootstrap_samples", 1000))
    seed = int(cfg.get("bootstrap_seed", 20260810))

    per_case = []
    for row, ref, pred, rt, pt, rc, pc in zip(rows, references, predictions, ref_tokens, pred_tokens, ref_chars, pred_chars):
        r1 = rouge_n(rt, pt, 1)
        r2 = rouge_n(rt, pt, 2)
        rl = rouge_l(rt, pt)
        crl = rouge_l(rc, pc)
        ref_facts, pred_facts = fact_set(ref), fact_set(pred)
        fact_p, fact_r, fact_f1 = set_prf(ref_facts, pred_facts)
        ref_entities, pred_entities = entity_set(ref), entity_set(pred)
        entity_p, entity_r, entity_f1 = set_prf(ref_entities, pred_entities)
        ref_relations, pred_relations = relation_proxy_set(ref), relation_proxy_set(pred)
        relation_p, relation_r, relation_f1 = set_prf(ref_relations, pred_relations)
        ref_concepts, pred_concepts = diagnosis_concepts(ref), diagnosis_concepts(pred)
        concept_p, concept_r, concept_f1 = set_prf(ref_concepts, pred_concepts)
        ref_key, pred_key = key_fact_set(ref), key_fact_set(pred)
        key_p, key_r, key_f1 = set_prf(ref_key, pred_key)
        contradictions, negation_flips, contradiction_den = contradiction_stats(ref, pred)
        sev_correct, sev_count = severity_stats(ref, pred)
        nums = compare_numbers(ref, pred)
        ref_labels, pred_labels = image_labels(ref), image_labels(pred)
        image_coverage = len(ref_labels & pred_labels) / len(ref_labels) if ref_labels else None
        hallucination = 1 - fact_p
        discordance = contradictions / contradiction_den if contradiction_den else 0.0
        crqs_style = statistics.fmean([fact_r, key_r, 1 - hallucination, 1 - discordance])
        if sentence_bleu is not None:
            bleu4 = sentence_bleu([rt], pt, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)
        else:
            bleu4 = None
        if sentence_chrf is not None:
            chrf = float(sentence_chrf(ref, pred))
        else:
            chrf = None
        per_case.append({
            "case_index": int(row["case_index"]),
            "reference_diagnosis": canonical_diagnosis(ref),
            "prediction_diagnosis": canonical_diagnosis(pred),
            "diagnosis_exact": float(canonical_diagnosis(ref) == canonical_diagnosis(pred)),
            "reference_diagnosis_concepts": "|".join(sorted(ref_concepts)),
            "prediction_diagnosis_concepts": "|".join(sorted(pred_concepts)),
            "diagnosis_concept_precision": concept_p,
            "diagnosis_concept_recall": concept_r,
            "diagnosis_concept_f1": concept_f1,
            "bleu4": bleu4,
            "rouge1_precision": r1[0], "rouge1_recall": r1[1], "rouge1_f1": r1[2],
            "rouge2_precision": r2[0], "rouge2_recall": r2[1], "rouge2_f1": r2[2],
            "rougeL_precision": rl[0], "rougeL_recall": rl[1], "rougeL_f1": rl[2],
            "char_rougeL_f1": crl[2],
            "meteor_exact": meteor_exact(rt, pt),
            "chrf": chrf,
            "clinical_fact_precision": fact_p,
            "clinical_fact_recall": fact_r,
            "clinical_fact_f1": fact_f1,
            "hare_style_entity_precision_proxy": entity_p,
            "hare_style_entity_recall_proxy": entity_r,
            "hare_style_entity_f1_proxy": entity_f1,
            "hare_style_relation_precision_proxy": relation_p,
            "hare_style_relation_recall_proxy": relation_r,
            "hare_style_relation_f1_proxy": relation_f1,
            "key_fact_precision": key_p,
            "key_fact_recall": key_r,
            "key_fact_f1": key_f1,
            "reference_fact_count": len(ref_facts),
            "prediction_fact_count": len(pred_facts),
            "hallucinated_fact_rate_proxy": hallucination,
            "omission_rate_proxy": 1 - fact_r,
            "clinical_discordance_rate_proxy": discordance,
            "negation_flip_count": negation_flips,
            "severity_correct": sev_correct,
            "severity_comparable": sev_count,
            "numeric_reference_count": nums["reference_count"],
            "numeric_prediction_count": nums["prediction_count"],
            "numeric_matched_count": nums["matched_count"],
            "numeric_exact_count": nums["exact_count"],
            "numeric_within5_count": nums["within5_count"],
            "numeric_within10_count": nums["within10_count"],
            "numeric_unsupported_count_proxy": nums["unsupported_count"],
            "numeric_omitted_count": nums["omitted_count"],
            "numeric_mae": safe_mean(nums["absolute_errors"]),
            "reference_stage": extract_stage(ref),
            "prediction_stage": extract_stage(pred),
            "diagnosis_section_present": float(bool(DIAG_RE.search(pred))),
            "image_label_coverage": image_coverage,
            "reference_length_chars": len(compact(ref)),
            "prediction_length_chars": len(compact(pred)),
            "length_ratio": len(compact(pred)) / max(1, len(compact(ref))),
            "trigram_repetition_rate": repetition_rate(pt),
            "hit_max_new_tokens": float(bool(row.get("hit_max_new_tokens"))),
            "latency_sec": float(row.get("latency_sec", 0.0)),
            "input_tokens": int(row.get("input_tokens", 0)),
            "output_tokens": int(row.get("output_tokens", 0)),
            "peak_gpu_memory_bytes": int(row.get("peak_gpu_memory_bytes", 0)),
            "crqs_style_proxy": crqs_style,
        })

    def mean_field(name: str) -> float | None:
        return safe_mean([x.get(name) for x in per_case])

    lexical = {
        "mean_sentence_bleu4": mean_field("bleu4"),
        "mean_rouge1_precision": mean_field("rouge1_precision"),
        "mean_rouge1_recall": mean_field("rouge1_recall"),
        "mean_rouge1_f1": mean_field("rouge1_f1"),
        "mean_rouge2_precision": mean_field("rouge2_precision"),
        "mean_rouge2_recall": mean_field("rouge2_recall"),
        "mean_rouge2_f1": mean_field("rouge2_f1"),
        "mean_rougeL_precision": mean_field("rougeL_precision"),
        "mean_rougeL_recall": mean_field("rougeL_recall"),
        "mean_rougeL_f1": mean_field("rougeL_f1"),
        "mean_char_rougeL_f1": mean_field("char_rougeL_f1"),
        "mean_meteor_exact_chinese": mean_field("meteor_exact"),
        "mean_chrf": mean_field("chrf"),
        "cider_lite": cider_lite(ref_tokens, pred_tokens),
    }
    if corpus_bleu is not None:
        refs_for_bleu = [[x] for x in ref_tokens]
        smooth = SmoothingFunction().method1
        for n in range(1, 5):
            weights = tuple([1 / n] * n + [0] * (4 - n))
            lexical[f"corpus_bleu{n}"] = float(corpus_bleu(refs_for_bleu, pred_tokens, weights=weights, smoothing_function=smooth))
    if corpus_chrf is not None:
        lexical["corpus_chrf"] = float(corpus_chrf([[x] for x in references], predictions))

    y_true = [x["reference_diagnosis"] for x in per_case]
    y_pred = [x["prediction_diagnosis"] for x in per_case]
    diagnostic, confusion = classification_metrics(y_true, y_pred, bootstrap_samples, seed)
    diagnostic["free_text_label_macro_f1"] = diagnostic.pop("macro_f1", None)
    diagnostic["free_text_label_macro_precision"] = diagnostic.pop("macro_precision", None)
    diagnostic["free_text_label_macro_recall"] = diagnostic.pop("macro_recall", None)
    diagnostic["free_text_label_micro_f1"] = diagnostic.pop("micro_f1", None)
    diagnostic["free_text_label_micro_precision"] = diagnostic.pop("micro_precision", None)
    diagnostic["free_text_label_micro_recall"] = diagnostic.pop("micro_recall", None)
    if "macro_f1_ci95" in diagnostic:
        diagnostic["free_text_label_macro_f1_ci95"] = diagnostic.pop("macro_f1_ci95")

    true_concepts = [diagnosis_concepts(x) for x in references]
    pred_concepts = [diagnosis_concepts(x) for x in predictions]
    concept_tp = sum(len(a & b) for a, b in zip(true_concepts, pred_concepts))
    concept_fp = sum(len(b - a) for a, b in zip(true_concepts, pred_concepts))
    concept_fn = sum(len(a - b) for a, b in zip(true_concepts, pred_concepts))
    concept_micro_p, concept_micro_r, concept_micro_f1 = prf(
        concept_tp, concept_tp + concept_fp, concept_tp + concept_fn
    )
    per_concept = {}
    for concept in DIAG_CONCEPTS:
        tp = sum(concept in a and concept in b for a, b in zip(true_concepts, pred_concepts))
        fp = sum(concept not in a and concept in b for a, b in zip(true_concepts, pred_concepts))
        fn = sum(concept in a and concept not in b for a, b in zip(true_concepts, pred_concepts))
        p, r, f = prf(tp, tp + fp, tp + fn)
        support = sum(concept in a for a in true_concepts)
        per_concept[concept] = {"precision": p, "recall": r, "f1": f, "support": support}
    supported_concept_f1 = [v["f1"] for v in per_concept.values() if v["support"] > 0]
    diagnostic.update({
        "concept_micro_precision": concept_micro_p,
        "concept_micro_recall": concept_micro_r,
        "concept_micro_f1": concept_micro_f1,
        "concept_macro_f1_supported": safe_mean(supported_concept_f1),
        "mean_case_concept_f1": mean_field("diagnosis_concept_f1"),
        "per_concept": per_concept,
    })
    placeholder_indices = [i for i, concepts in enumerate(true_concepts) if "unanswerable_placeholder" in concepts]
    nonplaceholder_indices = [i for i in range(len(per_case)) if i not in set(placeholder_indices)]
    diagnostic["placeholder_case_count"] = len(placeholder_indices)
    diagnostic["placeholder_exact_accuracy"] = safe_mean([
        "unanswerable_placeholder" in pred_concepts[i] for i in placeholder_indices
    ])
    diagnostic["nonplaceholder_free_text_exact_accuracy"] = safe_mean([
        per_case[i]["diagnosis_exact"] for i in nonplaceholder_indices
    ])

    total_num_matched = sum(x["numeric_matched_count"] for x in per_case)
    comparable_stages = [x for x in per_case if x["reference_stage"] is not None and x["prediction_stage"] is not None]
    stage_exact = safe_mean([x["reference_stage"] == x["prediction_stage"] for x in comparable_stages])
    stage_within1 = safe_mean([abs(x["reference_stage"] - x["prediction_stage"]) <= 1 for x in comparable_stages])

    clinical = {
        "clinical_fact_precision": mean_field("clinical_fact_precision"),
        "clinical_fact_recall": mean_field("clinical_fact_recall"),
        "clinical_fact_f1": mean_field("clinical_fact_f1"),
        "hare_style_entity_precision_proxy": mean_field("hare_style_entity_precision_proxy"),
        "hare_style_entity_recall_proxy": mean_field("hare_style_entity_recall_proxy"),
        "hare_style_entity_f1_proxy": mean_field("hare_style_entity_f1_proxy"),
        "hare_style_relation_precision_proxy": mean_field("hare_style_relation_precision_proxy"),
        "hare_style_relation_recall_proxy": mean_field("hare_style_relation_recall_proxy"),
        "hare_style_relation_f1_proxy": mean_field("hare_style_relation_f1_proxy"),
        "key_fact_precision": mean_field("key_fact_precision"),
        "key_fact_recall": mean_field("key_fact_recall"),
        "key_fact_f1": mean_field("key_fact_f1"),
        "hallucinated_fact_rate_reference_proxy": mean_field("hallucinated_fact_rate_proxy"),
        "omission_rate_reference_proxy": mean_field("omission_rate_proxy"),
        "clinical_discordance_rate_reference_proxy": mean_field("clinical_discordance_rate_proxy"),
        "negation_flip_rate_per_case": safe_mean([float(x["negation_flip_count"] > 0) for x in per_case]),
        "severity_accuracy_on_comparable": (
            sum(x["severity_correct"] for x in per_case) / sum(x["severity_comparable"] for x in per_case)
            if sum(x["severity_comparable"] for x in per_case) else None
        ),
        "crqs_style_proxy": mean_field("crqs_style_proxy"),
        "crqs_style_components": {
            "clinical_fact_coverage": mean_field("clinical_fact_recall"),
            "key_information_recall": mean_field("key_fact_recall"),
            "one_minus_hallucination": 1 - (mean_field("hallucinated_fact_rate_proxy") or 0),
            "one_minus_discordance": 1 - (mean_field("clinical_discordance_rate_proxy") or 0),
        },
    }
    numeric = {
        "matched_claim_count": total_num_matched,
        "numeric_exact_accuracy": (
            sum(x["numeric_exact_count"] for x in per_case) / total_num_matched if total_num_matched else None
        ),
        "numeric_within5_accuracy": (
            sum(x["numeric_within5_count"] for x in per_case) / total_num_matched if total_num_matched else None
        ),
        "numeric_within10_accuracy": (
            sum(x["numeric_within10_count"] for x in per_case) / total_num_matched if total_num_matched else None
        ),
        "mean_case_numeric_mae": mean_field("numeric_mae"),
        "unsupported_numeric_claim_count_reference_proxy": sum(x["numeric_unsupported_count_proxy"] for x in per_case),
        "omitted_numeric_claim_count": sum(x["numeric_omitted_count"] for x in per_case),
        "stage_comparable_case_count": len(comparable_stages),
        "stage_exact_accuracy": stage_exact,
        "stage_within_one_accuracy": stage_within1,
    }
    structure = {
        "diagnosis_section_presence_rate": mean_field("diagnosis_section_present"),
        "mean_image_label_coverage": mean_field("image_label_coverage"),
        "mean_length_ratio": mean_field("length_ratio"),
        "mean_trigram_repetition_rate": mean_field("trigram_repetition_rate"),
        "hit_max_new_tokens_rate": mean_field("hit_max_new_tokens"),
    }
    efficiency = {
        "case_count": len(per_case),
        "mean_latency_sec": mean_field("latency_sec"),
        "median_latency_sec": float(statistics.median(x["latency_sec"] for x in per_case)),
        "p95_latency_sec": float(np.percentile([x["latency_sec"] for x in per_case], 95)),
        "mean_input_tokens": mean_field("input_tokens"),
        "mean_output_tokens": mean_field("output_tokens"),
        "total_output_tokens": sum(x["output_tokens"] for x in per_case),
        "output_tokens_per_second_aggregate": (
            sum(x["output_tokens"] for x in per_case) / sum(x["latency_sec"] for x in per_case)
        ),
        "max_peak_gpu_memory_bytes_reported_by_worker": max(x["peak_gpu_memory_bytes"] for x in per_case),
    }

    ci_fields = [
        "diagnosis_exact", "diagnosis_concept_f1", "rougeL_f1", "bleu4", "chrf", "clinical_fact_f1",
        "key_fact_recall", "hallucinated_fact_rate_proxy", "omission_rate_proxy",
        "clinical_discordance_rate_proxy", "crqs_style_proxy",
    ]
    confidence_intervals = {
        name: bootstrap_ci([x.get(name) for x in per_case], bootstrap_samples, seed + i)
        for i, name in enumerate(ci_fields)
    }

    availability = {
        "automatic_reference_based_metrics": {"available": True, "reason": None},
        "diagnostic_classification_metrics": {"available": True, "reason": None},
        "clinical_fact_rule_proxy": {"available": True, "reason": "rule-based Chinese liver pathology extraction"},
        "expert_validated_hallucination_rate": {"available": False, "reason": "requires image-grounded expert annotation"},
        "expert_clinical_correctness": {"available": False, "reason": "requires blinded pathologist review"},
        "roi_iou_dice_map": {"available": False, "reason": "test set has no ROI coordinates or segmentation masks"},
        "calibration_ece_brier_aurc": {"available": False, "reason": "free-text generation has no defined diagnosis probability"},
        "fairness_metrics": {"available": False, "reason": "no demographic/site/scanner metadata in current JSON"},
        "external_ood_metrics": {"available": False, "reason": "no external cohort configured"},
        "bert_score": {
            "available": False,
            "reason": "no validated Chinese medical BERTScore encoder configured; avoiding implicit network download",
        },
        "spice": {"available": False, "reason": "no validated Chinese pathology scene-graph parser"},
        "visual_ablation_metrics": {"available": False, "reason": "requires separate no-image/shuffle/drop-image generation run"},
    }

    metrics = {
        "metadata": {
            "task": "multi-image Chinese liver pathology report generation",
            "case_count": len(rows),
            "model_dir": cfg["model_dir"],
            "test_json": cfg["test_json"],
            "automatic_clinical_metric_warning": (
                "Hallucination, omission, CRQS-style and discordance values are reference-based rule proxies, "
                "not image-grounded expert-validated clinical endpoints."
            ),
        },
        "lexical": lexical,
        "diagnostic": diagnostic,
        "clinical_reference_proxies": clinical,
        "numeric_and_stage": numeric,
        "structure": structure,
        "efficiency": efficiency,
        "confidence_intervals": confidence_intervals,
        "metric_availability": availability,
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metric_availability.json").write_text(json.dumps(availability, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_dir / "per_case_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_case[0].keys()))
        writer.writeheader()
        writer.writerows(per_case)
    if confusion:
        with (out_dir / "diagnosis_confusion.csv").open("w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerows(confusion)

    report = build_markdown_report(metrics)
    (out_dir / "metrics_report.md").write_text(report, encoding="utf-8")
    print(report)


def fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def build_markdown_report(metrics: dict) -> str:
    lex = metrics["lexical"]
    diag = metrics["diagnostic"]
    cli = metrics["clinical_reference_proxies"]
    num = metrics["numeric_and_stage"]
    struct = metrics["structure"]
    eff = metrics["efficiency"]
    ci = metrics["confidence_intervals"]
    lines = [
        "# Qwen3-VL 中文肝病理报告生成 Metric 报告",
        "",
        f"- 测试病例：{metrics['metadata']['case_count']}",
        f"- 模型目录：`{metrics['metadata']['model_dir']}`",
        "- 重要说明：临床事实、幻觉、遗漏、冲突和 CRQS-style 为相对参考报告的规则代理指标，不等价于病理专家或图像证据验证。",
        "",
        "## 核心结果",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| Diagnosis Exact Match | {fmt(diag.get('accuracy'))} |",
        f"| Diagnosis Concept Micro-F1 | {fmt(diag.get('concept_micro_f1'))} |",
        f"| Diagnosis Concept Macro-F1（有支持概念） | {fmt(diag.get('concept_macro_f1_supported'))} |",
        f"| Diagnosis Free-text Label Macro-F1 | {fmt(diag.get('free_text_label_macro_f1'))} |",
        f"| Non-placeholder Free-text Exact Match | {fmt(diag.get('nonplaceholder_free_text_exact_accuracy'))} |",
        f"| Clinical Fact F1（代理） | {fmt(cli.get('clinical_fact_f1'))} |",
        f"| HARE-style Entity F1（代理） | {fmt(cli.get('hare_style_entity_f1_proxy'))} |",
        f"| HARE-style Relation F1（代理） | {fmt(cli.get('hare_style_relation_f1_proxy'))} |",
        f"| Key Fact Recall（代理） | {fmt(cli.get('key_fact_recall'))} |",
        f"| CRQS-style（代理） | {fmt(cli.get('crqs_style_proxy'))} |",
        f"| Hallucinated Fact Rate（参考代理） | {fmt(cli.get('hallucinated_fact_rate_reference_proxy'))} |",
        f"| Omission Rate（参考代理） | {fmt(cli.get('omission_rate_reference_proxy'))} |",
        f"| Clinical Discordance（参考代理） | {fmt(cli.get('clinical_discordance_rate_reference_proxy'))} |",
        f"| Negation Flip Case Rate | {fmt(cli.get('negation_flip_rate_per_case'))} |",
        "",
        "## 文本生成指标",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
    ]
    for key in ["corpus_bleu1", "corpus_bleu2", "corpus_bleu3", "corpus_bleu4", "mean_rouge1_f1", "mean_rouge2_f1", "mean_rougeL_f1", "mean_meteor_exact_chinese", "corpus_chrf", "cider_lite"]:
        lines.append(f"| {key} | {fmt(lex.get(key))} |")
    for key in ["bertscore_precision", "bertscore_recall", "bertscore_f1"]:
        if key in lex:
            lines.append(f"| {key} | {fmt(lex.get(key))} |")
    lines.extend([
        "",
        "## 数值、分期与结构",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| Numeric Exact Accuracy | {fmt(num.get('numeric_exact_accuracy'))} |",
        f"| Numeric ±5 Accuracy | {fmt(num.get('numeric_within5_accuracy'))} |",
        f"| Numeric ±10 Accuracy | {fmt(num.get('numeric_within10_accuracy'))} |",
        f"| Mean Case Numeric MAE | {fmt(num.get('mean_case_numeric_mae'))} |",
        f"| Stage Exact Accuracy | {fmt(num.get('stage_exact_accuracy'))} |",
        f"| Stage Within-one Accuracy | {fmt(num.get('stage_within_one_accuracy'))} |",
        f"| Diagnosis Section Presence | {fmt(struct.get('diagnosis_section_presence_rate'))} |",
        f"| Image-label Coverage | {fmt(struct.get('mean_image_label_coverage'))} |",
        f"| Hit Max-new-tokens Rate | {fmt(struct.get('hit_max_new_tokens_rate'))} |",
        "",
        "## 推理效率",
        "",
        f"- Mean latency: {fmt(eff.get('mean_latency_sec'))} s/case",
        f"- P95 latency: {fmt(eff.get('p95_latency_sec'))} s/case",
        f"- Aggregate output speed: {fmt(eff.get('output_tokens_per_second_aggregate'))} token/s/worker-sum",
        f"- Max reported peak GPU memory: {fmt(eff.get('max_peak_gpu_memory_bytes_reported_by_worker'))} bytes",
        "",
        "## 95% Bootstrap CI",
        "",
        "| 指标 | 均值 | 95% CI |",
        "|---|---:|---:|",
    ])
    for key, value in ci.items():
        if value:
            point = value.get("value", value.get("mean"))
            lines.append(f"| {key} | {fmt(point)} | [{fmt(value['ci95_low'])}, {fmt(value['ci95_high'])}] |")
    lines.extend([
        "",
        "## 当前无法自动计算",
        "",
    ])
    for key, value in metrics["metric_availability"].items():
        if not value["available"]:
            lines.append(f"- `{key}`：{value['reason']}")
    lines.extend([
        "",
        "## 文件",
        "",
        "- `metrics.json`：完整聚合结果。",
        "- `per_case_metrics.csv`：逐病例指标。",
        "- `diagnosis_confusion.csv`：诊断混淆矩阵。",
        "- `dataset_audit.json`：数据泄漏与重复审计。",
        "- `predictions.jsonl`：原始预测与参考报告。",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
