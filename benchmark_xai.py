"""Reliability benchmark for post-hoc explanations of MentalBERT (LIME, SHAP, IG).

On the same MentalBERT model and the same texts, compare faithfulness and
stability on one unit: the unique word TYPE. Interventions: delete, mask, and
sufficiency; plus a random-ranking baseline and occlusion. Results are stored
per (text, explainer, metric, k) for paired statistics in analyze_benchmark.py.

Usage:
    python benchmark_xai.py --models mentalbert --pool ../results/benchmark/pool.csv
    python benchmark_xai.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MASTER = ROOT.parent
DATA = MASTER / "data" / "reddit_mental_preprocessed.csv"
OUT_DIR = MASTER / "results" / "benchmark"

LABELS = ["ADHD", "Anxiety", "Bipolar", "BPD", "Depression", "Normal", "OCD", "PTSD"]
KMAX = 10
REPORT_KS = (1, 3, 5, 10)
MAX_WORDS = 110


# --------------------------------------------------------------------------- text units

def word_types(text: str) -> list[str]:
    return list(dict.fromkeys(w.lower() for w in text.split()))


def remove_types(text: str, types, mask_token: str | None = None) -> str:
    drop = {t.lower() for t in types}
    out = []
    for word in text.split():
        if word.lower() in drop:
            if mask_token:
                out.append(mask_token)
        else:
            out.append(word)
    return " ".join(out) if out else " "


def keep_only(text: str, types) -> str:
    keep = {t.lower() for t in types}
    out = [w for w in text.split() if w.lower() in keep]
    return " ".join(out) if out else " "


def stable_seed(*parts) -> int:
    digest = hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:8], 16)


# --------------------------------------------------------------------------- models

class BertModel:
    def __init__(self, name, hf_model, tokenizer, device, max_length: int = 256):
        self.name = name
        self.hf = hf_model
        self.tok = tokenizer
        self.device = device
        self.mask_token = tokenizer.mask_token
        self.family = "bert"
        self.max_length = max_length

    def proba(self, texts, bs: int = 32) -> np.ndarray:
        import torch

        texts = [t if str(t).strip() else " " for t in texts]
        order = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), len(LABELS)), dtype=float)
        for i in range(0, len(texts), bs):
            idx = order[i : i + bs]
            enc = self.tok(
                [texts[j] for j in idx],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.hf(**enc).logits
            out[idx] = torch.softmax(logits, dim=-1).cpu().numpy()
        return out


# --------------------------------------------------------------------------- explainers
# Every explainer returns {word_type: score for the predicted class}.

def explain_lime(model, text, pred, types, seed, budget) -> dict[str, float]:
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(class_names=LABELS, bow=True, random_state=seed)
    result = explainer.explain_instance(
        text,
        lambda batch: model.proba(list(batch)),
        labels=(pred,),
        num_features=min(len(types), 20),
        num_samples=budget,
    )
    scores = {t: 0.0 for t in types}
    for word, weight in result.as_list(label=pred):
        key = word.lower()
        if key in scores:
            scores[key] = float(weight)
    return scores


def explain_shap_perm(model, text, pred, types, seed, budget) -> dict[str, float]:
    """Permutation SHAP with word types as players, masked by deletion."""
    import shap

    n = len(types)

    def coalition(matrix):
        texts = []
        for row in np.asarray(matrix):
            texts.append(keep_only(text, [t for t, bit in zip(types, row) if float(bit) > 0.5]))
        return model.proba(texts)[:, pred]

    masker = shap.maskers.Independent(np.zeros((1, n)))
    explainer = shap.Explainer(coalition, masker, algorithm="permutation", seed=seed)
    values = explainer(np.ones((1, n)), max_evals=max(2 * n + 1, budget), silent=True)
    row = np.asarray(values.values[0], dtype=float).reshape(-1)
    return {t: float(v) for t, v in zip(types, row[:n])}


def explain_ig(model, text, pred, types, n_steps) -> dict[str, float]:
    from explain import aggregate_to_words, compute_token_attributions

    _, offsets, token_scores = compute_token_attributions(
        model.hf,
        model.tok,
        text,
        pred,
        model.device,
        n_steps=n_steps,
        max_length=getattr(model, "max_length", 256),
    )
    acc: dict[str, float] = defaultdict(float)
    for item in aggregate_to_words(text, offsets, token_scores):
        acc[item.word.lower()] += float(item.score)
    return {t: float(acc.get(t, 0.0)) for t in types}


def explain_occlusion(model, text, pred, types) -> dict[str, float]:
    p0 = float(model.proba([text])[0][pred])
    texts = [remove_types(text, [t]) for t in types]
    probs = model.proba(texts)[:, pred]
    return {t: p0 - float(p) for t, p in zip(types, probs)}


# --------------------------------------------------------------------------- metrics

def positive_ranking(scores: dict[str, float]) -> list[str]:
    items = [(t, s) for t, s in scores.items() if s > 0]
    items.sort(key=lambda pair: -pair[1])
    return [t for t, _ in items]


def evaluate_ranking(model, text, pred, p0, ranking) -> dict[str, float]:
    """Deletion / masking / sufficiency curves for k = 1..KMAX in one batch."""
    top = ranking[:KMAX]
    batch: list[str] = []
    # deletion curve for every k = 1..KMAX (AOPC), other interventions at REPORT_KS only
    for k in range(1, KMAX + 1):
        batch.append(remove_types(text, top[:k]))
    if model.mask_token:
        for k in REPORT_KS:
            batch.append(remove_types(text, top[:k], model.mask_token))
    for k in REPORT_KS:
        batch.append(keep_only(text, top[:k]))
    probs = model.proba(batch)
    pos = 0
    out: dict[str, float] = {}

    block = probs[pos : pos + KMAX]
    pos += KMAX
    comp = p0 - block[:, pred]
    flips = (block.argmax(axis=1) != pred).astype(float)
    for k in REPORT_KS:
        out[f"comp_del@{k}"] = float(comp[k - 1])
        out[f"flip_del@{k}"] = float(flips[k - 1])
    out["aopc_del"] = float(comp.mean())

    if model.mask_token:
        block = probs[pos : pos + len(REPORT_KS)]
        pos += len(REPORT_KS)
        comp_m = p0 - block[:, pred]
        for i, k in enumerate(REPORT_KS):
            out[f"comp_mask@{k}"] = float(comp_m[i])
        out["aopc4_mask"] = float(comp_m.mean())

    block = probs[pos : pos + len(REPORT_KS)]
    suff = p0 - block[:, pred]
    for i, k in enumerate(REPORT_KS):
        out[f"suff@{k}"] = float(suff[i])
    out["aopc4_suff"] = float(suff.mean())
    out["n_positive"] = float(len(ranking))
    return out


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def stability_metrics(ref_scores, new_scores, prefix: str) -> dict[str, float]:
    ref_rank = positive_ranking(ref_scores)
    new_rank = positive_ranking(new_scores)
    return {
        f"{prefix}_j5": jaccard(ref_rank[:5], new_rank[:5]),
        f"{prefix}_j10": jaccard(ref_rank[:10], new_rank[:10]),
    }


# --------------------------------------------------------------------------- per-text run

class Config:
    def __init__(self, args):
        self.budget_bert = args.budget_bert
        self.ig_steps = args.ig_steps
        self.ig_steps_alt = args.ig_steps_alt
        self.random_draws = args.random_draws
        self.stability = not args.no_stability
        self.stability_every = max(1, args.stability_every)


def build_explainers(model, cfg: Config):
    """name -> callable(text, pred, types, seed) -> scores."""
    budget = cfg.budget_bert
    return {
        "lime": lambda text, pred, types, seed: explain_lime(model, text, pred, types, seed, budget),
        "shap": lambda text, pred, types, seed: explain_shap_perm(model, text, pred, types, seed, budget),
        "ig": lambda text, pred, types, seed: explain_ig(model, text, pred, types, cfg.ig_steps),
    }


def run_one(model, text: str, pool_id: int, label: int, cfg: Config) -> tuple[dict, list[dict]]:
    types = word_types(text)
    p_vec = model.proba([text])[0]
    pred = int(np.argmax(p_vec))
    p0 = float(p_vec[pred])
    explainers = build_explainers(model, cfg)

    scores: dict[str, dict[str, float]] = {}
    seconds: dict[str, float] = {}
    for name, fn in explainers.items():
        started = time.perf_counter()
        scores[name] = fn(text, pred, types, stable_seed(pool_id, model.name, name, "base"))
        seconds[name] = time.perf_counter() - started
    started = time.perf_counter()
    scores["occlusion"] = explain_occlusion(model, text, pred, types)
    seconds["occlusion"] = time.perf_counter() - started

    rows: list[dict] = []

    def add(explainer: str, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            metric, _, k = key.partition("@")
            rows.append(
                {
                    "pool_id": pool_id,
                    "model": model.name,
                    "explainer": explainer,
                    "metric": metric,
                    "k": int(k) if k else 0,
                    "value": value,
                }
            )

    for name, sc in scores.items():
        metrics = evaluate_ranking(model, text, pred, p0, positive_ranking(sc))
        metrics["seconds"] = seconds[name]
        add(name, metrics)

    # random-ranking baseline: average over several draws
    draws = []
    for d in range(cfg.random_draws):
        rng = np.random.default_rng(stable_seed(pool_id, model.name, "random", d))
        ranking = list(rng.permutation(types))
        draws.append(evaluate_ranking(model, text, pred, p0, ranking))
    add("random", {key: float(np.mean([d[key] for d in draws])) for key in draws[0]})

    pred_same = None
    run_stability = cfg.stability and (model.family == "tfidf" or pool_id % cfg.stability_every == 0)
    if run_stability:
        # (a) re-run with another seed / IG step count
        for name, fn in explainers.items():
            if name == "ig":
                alt = explain_ig(model, text, pred, types, cfg.ig_steps_alt)
            else:
                alt = fn(text, pred, types, stable_seed(pool_id, model.name, name, "alt"))
            add(name, stability_metrics(scores[name], alt, "stab_rerun"))

        # (b) delete irrelevant words (outside every method's top-10), keep if prediction unchanged
        protected = set()
        for sc in scores.values():
            protected.update(positive_ranking(sc)[:KMAX])
        candidates = [t for t in types if t not in protected]
        rng = np.random.default_rng(stable_seed(pool_id, model.name, "perturb"))
        if len(candidates) >= 2:
            m = min(len(candidates), max(2, round(0.1 * len(types))))
            chosen = list(rng.choice(candidates, size=m, replace=False))
            ptext = remove_types(text, chosen)
            new_pred = int(np.argmax(model.proba([ptext])[0]))
            pred_same = int(new_pred == pred)
            if pred_same:
                ptypes = word_types(ptext)
                for name, fn in explainers.items():
                    new_scores = fn(ptext, pred, ptypes, stable_seed(pool_id, model.name, name, "base"))
                    add(name, stability_metrics(scores[name], new_scores, "stab_perturb"))

    top_all = {name: positive_ranking(sc)[:5] for name, sc in scores.items()}
    meta = {
        "pool_id": pool_id,
        "model": model.name,
        "label": label,
        "pred": pred,
        "prob": p0,
        "correct": int(pred == label),
        "n_words": len(text.split()),
        "n_types": len(types),
        "perturb_pred_same": pred_same if pred_same is not None else "",
        "top5": "|".join(f"{n}:{','.join(v)}" for n, v in top_all.items()),
    }
    return meta, rows


# --------------------------------------------------------------------------- data & models

def reproduce_split(data_path: Path):
    """Replicate the notebook split (RandomUnderSampler + 80/20, seed 42)."""
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(data_path)
    X = df["text"].fillna("").astype(str)
    y = df["label"]
    X_res, y_res = RandomUnderSampler(sampling_strategy="auto", random_state=42).fit_resample(
        X.values.reshape(-1, 1), y
    )
    balanced = pd.DataFrame({"text": X_res.flatten(), "label": y_res})
    train_df, test_df = train_test_split(balanced, test_size=0.2, random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_pool(test_df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    frame = test_df.copy()
    frame["text"] = frame["text"].map(lambda t: " ".join(str(t).split()[:MAX_WORDS]))
    frame = frame[frame["text"].str.split().str.len() >= 12]
    parts = [g.sample(n=min(per_class, len(g)), random_state=seed) for _, g in frame.groupby("label")]
    pool = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    pool["pool_id"] = range(len(pool))
    return pool


def load_mentalbert():
    import torch
    from transformers import AutoTokenizer

    folder = MASTER / "results" / "mentalbert" / "mentalbert"
    device = torch.device("cpu")
    hf = torch.load(folder / "mentalbert_full_model.pt", map_location=device, weights_only=False)
    hf.eval()
    tokenizer = AutoTokenizer.from_pretrained(folder)
    return BertModel("mentalbert", hf, tokenizer, device, max_length=256)


# --------------------------------------------------------------------------- main

def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if new:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="mentalbert")
    parser.add_argument("--per-class", type=int, default=15)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--budget-bert", type=int, default=200)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--ig-steps-alt", type=int, default=32)
    parser.add_argument("--random-draws", type=int, default=3)
    parser.add_argument("--no-stability", action="store_true")
    parser.add_argument(
        "--stability-every",
        type=int,
        default=3,
        help="transformers: run the (expensive) stability checks on every N-th pool text",
    )
    parser.add_argument("--smoke", action="store_true", help="2 texts per class=1, quick check")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--pool", default="", help="Existing pool.csv so texts match a prior run")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.smoke:
        args.per_class = 1
        out_dir = out_dir / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in args.models.split(",") if n.strip()]
    cfg = Config(args)

    train_df, test_df = reproduce_split(Path(args.data))
    if args.pool:
        pool = pd.read_csv(args.pool)
        if args.smoke:
            pool = pool.groupby("label", sort=False).head(1).reset_index(drop=True)
        print(f"[pool] {len(pool)} texts from {args.pool}", flush=True)
    else:
        pool = build_pool(test_df, args.per_class, args.seed)
        print(f"[pool] {len(pool)} texts from the notebook TEST split (seed={args.seed})", flush=True)
    pool.to_csv(out_dir / "pool.csv", index=False)

    models: dict[str, object] = {}
    unknown = [name for name in names if name != "mentalbert"]
    if unknown:
        raise SystemExit(f"unsupported models for this repo: {', '.join(unknown)}")
    if "mentalbert" in names:
        models["mentalbert"] = load_mentalbert()

    meta_path = out_dir / "meta.csv"
    rows_path = out_dir / "rows.csv"
    done = set()
    if meta_path.exists():
        existing = pd.read_csv(meta_path)
        done = set(zip(existing["pool_id"], existing["model"]))
        print(f"[resume] {len(done)} (text, model) pairs already done", flush=True)

    started = time.time()
    total = len(pool) * len(models)
    counter = 0
    for _, item in pool.iterrows():
        for name in names:
            if name not in models:
                continue
            counter += 1
            key = (int(item["pool_id"]), name)
            if key in done:
                continue
            t0 = time.perf_counter()
            meta, rows = run_one(
                models[name],
                item["text"],
                int(item["pool_id"]),
                int(item["label"]),
                cfg,
            )
            append_csv(rows_path, rows)
            append_csv(meta_path, [meta])
            print(
                f"[{counter}/{total}] pool={key[0]} {name:8s} label={LABELS[meta['label']]:10s} "
                f"pred={LABELS[meta['pred']]:10s} p={meta['prob']:.2f} "
                f"{time.perf_counter() - t0:5.1f}s (elapsed {time.time() - started:.0f}s)",
                flush=True,
            )
    print(f"[done] {rows_path}", flush=True)


if __name__ == "__main__":
    main()
