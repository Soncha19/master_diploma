"""Statistics for benchmark_xai.py output (paired, with random baseline).

Reads results/benchmark/{rows,meta}.csv and writes analysis.md / analysis.json.

For every model and metric it reports
  * mean with bootstrap 95% CI per explainer (including the random and occlusion references),
  * every method vs the random baseline (paired Wilcoxon, one-sided, win rate),
  * every pair of real methods (paired Wilcoxon, Holm-corrected within model x metric,
    bootstrap CI of the mean paired difference, matched-pairs rank-biserial effect size),
  * the same on a high-confidence / lower-confidence split (saturation check),
  * how well each method's faithfulness predicts model errors (AUC; few errors => wide CI).

Usage:
    python analyze_benchmark.py [--dir ../results/benchmark]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
DEFAULT_DIR = ROOT.parent / "results" / "benchmark"

# metric key -> (direction, description)
METRICS = {
    "aopc_del": ("higher", "AOPC deletion, k=1..10 (drop of P(class) when top words are deleted)"),
    "comp_del@5": ("higher", "Deletion drop at k=5"),
    "aopc4_mask": ("higher", "Mean drop when top words are replaced by the mask token (k=1,3,5,10)"),
    "aopc4_suff": ("lower", "Sufficiency: P(class) lost when ONLY the top words are kept (k=1,3,5,10)"),
    "flip_del@5": ("higher", "Prediction flips after deleting top-5 words"),
    "stab_rerun_j5": ("higher", "Top-5 Jaccard after re-running (new seed / more IG steps)"),
    "stab_perturb_j5": ("higher", "Top-5 Jaccard after deleting irrelevant words (prediction unchanged)"),
    "seconds": ("lower", "Seconds per explanation"),
}
REAL_METHODS = ("ig", "lime", "shap")
REFERENCES = ("occlusion", "random")
N_BOOT = 5000
RNG = np.random.default_rng(0)


def load(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(directory / "rows.csv")
    meta = pd.read_csv(directory / "meta.csv")
    rows["key"] = np.where(rows["k"] > 0, rows["metric"] + "@" + rows["k"].astype(str), rows["metric"])
    return rows, meta


def series(rows: pd.DataFrame, model: str, explainer: str, key: str) -> pd.Series:
    part = rows[(rows.model == model) & (rows.explainer == explainer) & (rows.key == key)]
    return part.set_index("pool_id")["value"].astype(float)


def boot_ci(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    idx = RNG.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon(diff: np.ndarray, alternative: str = "two-sided") -> tuple[float, float]:
    """Returns (p-value, matched-pairs rank-biserial correlation)."""
    diff = diff[~np.isnan(diff)]
    nonzero = diff[diff != 0]
    if len(nonzero) < 5:
        return float("nan"), float("nan")
    try:
        p = float(stats.wilcoxon(nonzero, alternative=alternative).pvalue)
    except ValueError:
        return float("nan"), float("nan")
    ranks = stats.rankdata(np.abs(nonzero))
    r_plus = ranks[nonzero > 0].sum()
    r_minus = ranks[nonzero < 0].sum()
    rbc = float((r_plus - r_minus) / (r_plus + r_minus))
    return p, rbc


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort([1.0 if np.isnan(p) else p for p in pvalues])
    adjusted = [float("nan")] * len(pvalues)
    running = 0.0
    m = len(pvalues)
    for rank, idx in enumerate(order):
        p = pvalues[idx]
        if np.isnan(p):
            continue
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[idx] = running
    return adjusted


def orient(values: np.ndarray, direction: str) -> np.ndarray:
    return values if direction == "higher" else -values


def describe(rows: pd.DataFrame, model: str, key: str, explainers: list[str]) -> list[dict]:
    out = []
    for name in explainers:
        s = series(rows, model, name, key).dropna()
        if s.empty:
            continue
        lo, hi = boot_ci(s.to_numpy())
        out.append({"explainer": name, "n": int(len(s)), "mean": float(s.mean()), "median": float(s.median()),
                    "ci_low": lo, "ci_high": hi})
    return out


def vs_random(rows: pd.DataFrame, model: str, key: str, direction: str, methods: list[str]) -> list[dict]:
    out = []
    rnd = series(rows, model, "random", key)
    for name in methods:
        s = series(rows, model, name, key)
        joined = pd.concat([s, rnd], axis=1, keys=["m", "r"]).dropna()
        if joined.empty:
            continue
        diff = orient((joined["m"] - joined["r"]).to_numpy(), direction)
        p, rbc = wilcoxon(diff, "greater")
        out.append({"explainer": name, "n": int(len(diff)), "mean_gain_over_random": float(diff.mean()),
                    "win_rate": float((diff > 0).sum() / max(1, (diff != 0).sum())), "p_greater": p, "rank_biserial": rbc})
    return out


def pairwise(rows: pd.DataFrame, model: str, key: str, direction: str, methods: list[str]) -> list[dict]:
    out = []
    for a, b in itertools.combinations(methods, 2):
        joined = pd.concat([series(rows, model, a, key), series(rows, model, b, key)], axis=1, keys=["a", "b"]).dropna()
        if joined.empty:
            continue
        diff = orient((joined["a"] - joined["b"]).to_numpy(), direction)  # >0 means a is better
        p, rbc = wilcoxon(diff)
        lo, hi = boot_ci(diff)
        out.append({"a": a, "b": b, "n": int(len(diff)), "mean_diff_a_minus_b_better": float(diff.mean()),
                    "ci_low": lo, "ci_high": hi, "p": p, "rank_biserial": rbc,
                    "a_win_rate": float((diff > 0).sum() / max(1, (diff != 0).sum()))})
    adjusted = holm([row["p"] for row in out])
    for row, p_adj in zip(out, adjusted):
        row["p_holm"] = p_adj
    return out


def auc(scores: np.ndarray, positives: np.ndarray) -> float:
    pos = scores[positives]
    neg = scores[~positives]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def error_detection(rows: pd.DataFrame, meta: pd.DataFrame, model: str, methods: list[str]) -> list[dict]:
    m = meta[meta.model == model].set_index("pool_id")
    wrong = (m["correct"] == 0)
    out = []
    signals = {"confidence (1-prob)": 1.0 - m["prob"]}
    for name in methods:
        s = series(rows, model, name, "aopc_del")
        signals[f"{name}: low deletion AOPC"] = -s
    for label, s in signals.items():
        joined = pd.concat([s, wrong], axis=1, keys=["s", "w"]).dropna()
        if joined["w"].sum() < 3:
            continue
        a = auc(joined["s"].to_numpy(), joined["w"].to_numpy(dtype=bool))
        idx = RNG.integers(0, len(joined), size=(1000, len(joined)))
        boots = []
        for ix in idx:
            sub = joined.iloc[ix]
            if sub["w"].sum() in (0, len(sub)):
                continue
            boots.append(auc(sub["s"].to_numpy(), sub["w"].to_numpy(dtype=bool)))
        lo, hi = (np.percentile(boots, 2.5), np.percentile(boots, 97.5)) if boots else (float("nan"),) * 2
        out.append({"signal": label, "n": int(len(joined)), "n_errors": int(joined["w"].sum()),
                    "auc": a, "ci_low": float(lo), "ci_high": float(hi)})
    return out


def fmt(x, digits=3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_no data_\n"
    header = "| " + " | ".join(frame.columns) + " |\n|" + "|".join("---" for _ in frame.columns) + "|\n"
    body = "".join("| " + " | ".join(str(v) for v in row) + " |\n" for row in frame.itertuples(index=False))
    return header + body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    args = parser.parse_args()
    directory = Path(args.dir)
    rows, meta = load(directory)

    result: dict = {}
    lines: list[str] = ["# Benchmark: LIME / SHAP / IG\n"]
    acc = meta.groupby("model").agg(n=("pool_id", "count"), accuracy=("correct", "mean"),
                                    mean_prob=("prob", "mean")).reset_index()
    lines.append("## Sample\n")
    lines.append(md_table(acc.round(3)))
    lines.append("\nAll values are paired per text. 'Random' = mean over random rankings of the same words.\n")

    for model in sorted(meta["model"].unique()):
        methods = [m for m in REAL_METHODS if not series(rows, model, m, "aopc_del").empty]
        everything = methods + [r for r in REFERENCES]
        lines.append(f"\n## Model: {model}\n")
        result[model] = {}
        for key, (direction, description) in METRICS.items():
            desc = describe(rows, model, key, everything)
            if not desc:
                continue
            lines.append(f"\n### {key} ({direction} is better) - {description}\n")
            frame = pd.DataFrame(desc)
            frame["95% CI"] = frame.apply(lambda r: f"[{fmt(r.ci_low)}, {fmt(r.ci_high)}]", axis=1)
            lines.append(md_table(frame[["explainer", "n", "mean", "median", "95% CI"]].round(3)))
            entry = {"description": description, "direction": direction, "descriptive": desc}
            if key not in ("seconds",):
                vr = vs_random(rows, model, key, direction, methods + ["occlusion"])
                if vr:
                    lines.append("\nvs random baseline (positive gain = better than random):\n")
                    vf = pd.DataFrame(vr)
                    vf["p_greater"] = vf["p_greater"].map(lambda v: fmt(v, 4))
                    lines.append(md_table(vf.round(3)))
                    entry["vs_random"] = vr
            pw = pairwise(rows, model, key, direction, methods)
            if pw:
                lines.append("\npaired comparison (positive = first method better; Holm-adjusted p):\n")
                pf = pd.DataFrame(pw)
                for col in ("p", "p_holm"):
                    pf[col] = pf[col].map(lambda v: fmt(v, 4))
                pf["CI"] = pf.apply(lambda r: f"[{fmt(r.ci_low)}, {fmt(r.ci_high)}]", axis=1)
                lines.append(md_table(pf[["a", "b", "n", "mean_diff_a_minus_b_better", "CI", "p", "p_holm",
                                          "rank_biserial", "a_win_rate"]].round(3)))
                entry["pairwise"] = pw
            result[model][key] = entry

        # saturation split
        m = meta[meta.model == model].set_index("pool_id")
        lines.append("\n### Split by model confidence (AOPC deletion, mean)\n")
        split_rows = []
        for label, mask in (("prob >= 0.95", m["prob"] >= 0.95), ("prob < 0.95", m["prob"] < 0.95)):
            ids = set(m.index[mask])
            for name in everything:
                s = series(rows, model, name, "aopc_del")
                s = s[s.index.isin(ids)]
                if len(s):
                    split_rows.append({"group": label, "explainer": name, "n": len(s), "mean": round(float(s.mean()), 3)})
        lines.append(md_table(pd.DataFrame(split_rows)))
        result[model]["confidence_split"] = split_rows

        ed = error_detection(rows, meta, model, methods)
        if ed:
            lines.append("\n### Do the signals detect model errors? (AUC, higher = detects errors)\n")
            lines.append(md_table(pd.DataFrame(ed).round(3)))
            result[model]["error_detection"] = ed

    (directory / "analysis.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    (directory / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
