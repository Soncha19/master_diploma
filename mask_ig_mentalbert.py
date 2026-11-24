"""IG on MentalBERT with a [MASK] baseline instead of pad.

Same 120 texts and the same deletion AOPC as benchmark_xai.py.
The target is still the predicted-class logit. Ordinary pad-IG numbers
are read from the already saved benchmark, not recomputed.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from benchmark_xai import (
    LABELS,
    evaluate_ranking,
    jaccard,
    load_mentalbert,
    positive_ranking,
    word_types,
)
from explain import aggregate_to_words, compute_token_attributions

ROOT = Path(__file__).resolve().parent
MASTER = ROOT.parent
POOL = MASTER / "results" / "benchmark" / "pool.csv"
ROWS = MASTER / "results" / "benchmark_mentalbert" / "rows.csv"
META = MASTER / "results" / "benchmark_mentalbert" / "meta.csv"
OUT = MASTER / "results" / "benchmark_mentalbert" / "mask_ig.json"


def word_scores(model, text: str, pred: int, types: list[str], n_steps: int) -> dict[str, float]:
    _, offsets, token_scores = compute_token_attributions(
        model.hf,
        model.tok,
        text,
        pred,
        model.device,
        n_steps=n_steps,
        max_length=model.max_length,
        baseline="mask",
    )
    acc: dict[str, float] = defaultdict(float)
    for item in aggregate_to_words(text, offsets, token_scores):
        acc[item.word.lower()] += float(item.score)
    return {t: float(acc.get(t, 0.0)) for t in types}


def saved_mean(rows: pd.DataFrame, explainer: str, metric: str) -> pd.Series:
    part = rows[(rows.explainer == explainer) & (rows.metric == metric) & (rows.k == 0)]
    return part.groupby("pool_id")["value"].mean()


def paired(a: pd.Series, b: pd.Series) -> dict:
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    diff = (both["a"] - both["b"]).to_numpy()
    nonzero = diff[diff != 0]
    p = float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) >= 5 else float("nan")
    return {
        "n": int(len(both)),
        "mean_a": float(both["a"].mean()),
        "mean_b": float(both["b"].mean()),
        "mean_diff": float(diff.mean()),
        "p": p,
    }


def ig_top5(cell: str) -> list[str]:
    for part in str(cell).split("|"):
        if part.startswith("ig:"):
            return [word for word in part[3:].split(",") if word]
    return []


def main() -> None:
    pool = pd.read_csv(POOL)
    rows = pd.read_csv(ROWS)
    rows = rows[rows.model == "mentalbert"]
    meta = pd.read_csv(META).set_index("pool_id")
    model = load_mentalbert()
    mask_id = model.tok.mask_token_id
    if mask_id is None:
        raise SystemExit("MentalBERT tokenizer has no mask token")
    print(f"[baseline] mask_token_id={mask_id} token={model.tok.convert_ids_to_tokens(mask_id)}", flush=True)

    records = []
    started = time.time()
    for _, item in pool.iterrows():
        text = str(item["text"])
        pool_id = int(item["pool_id"])
        types = word_types(text)
        probs = model.proba([text])[0]
        pred = int(np.argmax(probs))
        p0 = float(probs[pred])
        t0 = time.perf_counter()
        scores = word_scores(model, text, pred, types, n_steps=16)
        seconds = time.perf_counter() - t0
        ranking = positive_ranking(scores)
        metrics = evaluate_ranking(model, text, pred, p0, ranking)
        jaccard_steps = None
        if pool_id % 3 == 0:
            alt = word_scores(model, text, pred, types, n_steps=32)
            jaccard_steps = jaccard(ranking[:5], positive_ranking(alt)[:5])
        ordinary = ig_top5(meta.loc[pool_id, "top5"]) if pool_id in meta.index else []
        records.append(
            {
                "pool_id": pool_id,
                "label": LABELS[int(item["label"])],
                "pred": LABELS[pred],
                "correct": int(pred == int(item["label"])),
                "aopc": metrics["aopc_del"],
                "jaccard_steps": jaccard_steps,
                "jaccard_vs_pad_ig": jaccard(ranking[:5], ordinary),
                "top5": ranking[:5],
                "seconds": seconds,
            }
        )
        print(
            f"[{len(records)}/{len(pool)}] {LABELS[pred]:12} "
            f"AOPC {metrics['aopc_del']:.3f} {seconds:4.1f}s "
            f"(elapsed {time.time() - started:.0f}s)",
            flush=True,
        )

    aopc = pd.Series({r["pool_id"]: r["aopc"] for r in records})
    ig_aopc = saved_mean(rows, "ig", "aopc_del")
    lime_aopc = saved_mean(rows, "lime", "aopc_del")
    random_aopc = saved_mean(rows, "random", "aopc_del")
    stab = [r["jaccard_steps"] for r in records if r["jaccard_steps"] is not None]
    overlap = [r["jaccard_vs_pad_ig"] for r in records]
    ig_j = rows[(rows.explainer == "ig") & (rows.metric == "stab_rerun_j5")]["value"]
    payload = {
        "baseline": "mask",
        "n": len(records),
        "aopc_mask": float(aopc.mean()),
        "aopc_ig": float(ig_aopc.mean()),
        "aopc_lime": float(lime_aopc.mean()),
        "aopc_random": float(random_aopc.mean()),
        "jaccard_steps_mask": float(np.mean(stab)),
        "jaccard_steps_ig": float(ig_j.mean()) if len(ig_j) else None,
        "n_stability": len(stab),
        "jaccard_vs_pad_ig": float(np.mean(overlap)),
        "n_identical_top5": int(sum(value == 1.0 for value in overlap)),
        "vs_ig": paired(aopc, ig_aopc),
        "vs_lime": paired(aopc, lime_aopc),
        "vs_random": paired(aopc, random_aopc),
        "median_seconds": float(np.median([r["seconds"] for r in records])),
        "records": records,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[result] mask AOPC {payload['aopc_mask']:.3f} "
        f"pad-IG {payload['aopc_ig']:.3f} LIME {payload['aopc_lime']:.3f} "
        f"random {payload['aopc_random']:.3f}",
        flush=True,
    )
    print(
        f"[stability] mask J@5 {payload['jaccard_steps_mask']:.3f} "
        f"pad-IG {payload['jaccard_steps_ig']:.3f} n={payload['n_stability']}",
        flush=True,
    )
    print(
        f"[overlap] J@5 vs pad-IG {payload['jaccard_vs_pad_ig']:.3f} "
        f"identical {payload['n_identical_top5']}/{payload['n']}",
        flush=True,
    )
    print(f"[vs ig] {payload['vs_ig']}", flush=True)
    print(f"[vs lime] {payload['vs_lime']}", flush=True)
    print(f"[wrote] {OUT}", flush=True)


if __name__ == "__main__":
    main()
