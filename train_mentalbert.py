"""Fine-tune models that were not part of the bachelor thesis and score them.

The corpus split is the same protocol as the cited bachelor work
(RandomUnderSampler random_state=42, train_test_split test_size=0.2, random_state=42).
Bachelor checkpoints are not loaded and their classification scores are not computed.

Trains mental/mental-bert-base-uncased only. Metrics: macro-F1, macro precision, macro recall,
confusion matrix. Integrated Gradients run on a stratified slice of the test set
for the model with the higher macro-F1.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from benchmark_xai import LABELS, reproduce_split, DATA
from explain import aggregate_to_words, compute_token_attributions

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "results" / "mentalbert"

MODELS = {
    "mentalbert": "mental/mental-bert-base-uncased",
    "bert": "bert-base-uncased",
}


class EncodedTexts(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.enc = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        return {
            "input_ids": self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def classification_scores(y_true, y_pred) -> dict:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(LABELS))), zero_division=0
    )
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f),
        "per_class": [
            {
                "label": LABELS[i],
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(LABELS))
        ],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS)))).tolist(),
    }


def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in loader:
            gold.append(batch["labels"].numpy())
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**batch).logits
            preds.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(gold), np.concatenate(preds)


def train_one(name: str, hub_id: str, train_df, test_df, args, device) -> dict:
    folder = args.run_name or name
    dest = OUT / folder
    metrics_path = dest / "metrics.json"
    if metrics_path.exists() and not args.force:
        print(f"[skip] {name} already has metrics.json", flush=True)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(hub_id)
    model = AutoModelForSequenceClassification.from_pretrained(hub_id, num_labels=len(LABELS))
    model.to(device)

    train_ds = EncodedTexts(train_df["text"], train_df["label"].astype(int), tokenizer, args.max_length)
    test_ds = EncodedTexts(test_df["text"], test_df["label"].astype(int), tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    steps = max(1, len(train_loader))
    if args.probe:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        batch = next(iter(train_loader))
        batch = {k: v.to(device) for k, v in batch.items()}
        started = time.perf_counter()
        for _ in range(8):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
        elapsed = time.perf_counter() - started
        print(
            f"[probe] {name} 8 steps x batch {args.batch_size} = {elapsed:.1f}s "
            f"({elapsed / 8:.2f}s/step). Full epoch ~ {steps * elapsed / 8 / 60:.0f} min, "
            f"{args.epochs} epochs ~ {args.epochs * steps * elapsed / 8 / 3600:.1f} h",
            flush=True,
        )
        return {"probe": True, "sec_per_step": elapsed / 8, "steps_per_epoch": steps}

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = steps * args.epochs
    warmup = max(1, int(0.06 * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        remain = max(1, total_steps - warmup)
        return args.lr * max(0.0, (total_steps - step) / remain)

    best_f1 = -1.0
    best_state = None
    history = []
    step = 0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        t0 = time.perf_counter()
        for batch in train_loader:
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)
            step += 1
            optimizer.zero_grad(set_to_none=True)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            if step == 1 or step % 100 == 0:
                print(
                    f"[{name}] step {step}/{total_steps} loss {losses[-1]:.4f}",
                    flush=True,
                )
        y_true, y_pred = predict(model, test_loader, device)
        scores = classification_scores(y_true, y_pred)
        scores["epoch"] = epoch + 1
        scores["train_loss"] = float(np.mean(losses))
        scores["epoch_minutes"] = (time.perf_counter() - t0) / 60
        history.append({k: scores[k] for k in ("epoch", "train_loss", "macro_f1", "epoch_minutes")})
        print(
            f"[{name}] epoch {epoch + 1} loss {scores['train_loss']:.4f} "
            f"macro-F1 {scores['macro_f1']:.3f} P {scores['macro_precision']:.3f} "
            f"R {scores['macro_recall']:.3f} ({scores['epoch_minutes']:.1f} min)",
            flush=True,
        )
        if scores["macro_f1"] > best_f1:
            best_f1 = scores["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_scores = scores

    model.load_state_dict(best_state)
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest)
    tokenizer.save_pretrained(dest)
    pt_path = dest / f"{folder}_full_model.pt"
    torch.save(model, pt_path)
    print(f"[save] {pt_path}", flush=True)
    best_scores["hub_id"] = hub_id
    best_scores["history"] = history
    best_scores["n_train"] = int(len(train_df))
    best_scores["n_test"] = int(len(test_df))
    best_scores["max_length"] = args.max_length
    best_scores["epochs_ran"] = args.epochs
    best_scores["batch_size"] = args.batch_size
    best_scores["lr"] = args.lr
    metrics_path.write_text(json.dumps(best_scores, ensure_ascii=False, indent=2), encoding="utf-8")
    save_confusion(folder, best_scores)
    return best_scores


def save_confusion(name: str, scores: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.array(scores["confusion_matrix"], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=40, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Label")
    ax.set_title(f"{name}: macro-F1 {scores['macro_f1']:.3f}")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / f"cm_{name}.png", dpi=140)
    plt.close(fig)


def deletion_aopc(model, tokenizer, text: str, device, ranking: list[str], kmax: int = 5) -> float:
    model.eval()

    def prob(sample: str, target: int) -> float:
        enc = tokenizer(sample, return_tensors="pt", truncation=True, max_length=128)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits[0]
            return float(torch.softmax(logits, dim=-1)[target].item())

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits[0]
        dist = torch.softmax(logits, dim=-1)
        pred = int(dist.argmax().item())
        base = float(dist[pred].item())
    words = text.split()
    drops = []
    for k in range(1, kmax + 1):
        ban = set(ranking[:k])
        kept = [w for w in words if w not in ban]
        sample = " ".join(kept) if kept else words[0]
        drops.append(base - prob(sample, pred))
    return float(np.mean(drops))


def explain_best(name: str, test_df, args, device) -> None:
    dest = OUT / name
    tokenizer = AutoTokenizer.from_pretrained(dest)
    model = AutoModelForSequenceClassification.from_pretrained(dest)
    model.to(device)
    model.eval()
    frame = test_df.copy()
    frame["text"] = frame["text"].map(lambda t: " ".join(str(t).split()[:110]))
    frame = frame[frame["text"].str.split().str.len() >= 12]
    parts = [g.sample(n=min(args.ig_per_class, len(g)), random_state=11) for _, g in frame.groupby("label")]
    sample = __import__("pandas").concat(parts).reset_index(drop=True)
    rows = []
    for i, row in sample.iterrows():
        text = str(row["text"])
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits[0]
            pred = int(logits.argmax().item())
        tokens, offsets, scores = compute_token_attributions(
            model, tokenizer, text, pred, device, n_steps=args.ig_steps
        )
        words = aggregate_to_words(text, offsets, scores)
        ranking = [w.word for w in sorted(words, key=lambda item: item.score, reverse=True) if w.score > 0]
        rng = np.random.default_rng(1000 + i)
        types = list(dict.fromkeys(text.split()))
        random_ranking = list(rng.permutation(types))
        rows.append(
            {
                "label": LABELS[int(row["label"])],
                "pred": LABELS[pred],
                "top5": ranking[:5],
                "aopc_ig": deletion_aopc(model, tokenizer, text, device, ranking),
                "aopc_random": deletion_aopc(model, tokenizer, text, device, random_ranking),
            }
        )
        print(f"[ig] {i + 1}/{len(sample)} {rows[-1]['label']} -> {rows[-1]['pred']}", flush=True)
    ig_mean = float(np.mean([r["aopc_ig"] for r in rows]))
    rnd_mean = float(np.mean([r["aopc_random"] for r in rows]))
    payload = {"model": name, "n": len(rows), "aopc_ig": ig_mean, "aopc_random": rnd_mean, "rows": rows}
    (OUT / "ig_sample.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ig] AOPC IG {ig_mean:.3f} random {rnd_mean:.3f} n={len(rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="mentalbert")
    parser.add_argument("--run-name", default="", help="Separate output folder so existing weights are kept")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-ig", action="store_true")
    parser.add_argument("--ig-per-class", type=int, default=3)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--data", default=str(DATA))
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    train_df, test_df = reproduce_split(Path(args.data))
    print(f"[data] train {len(train_df)} test {len(test_df)}", flush=True)

    chosen = [name.strip() for name in args.models.split(",") if name.strip()]
    results = {}
    for name in chosen:
        results[name] = train_one(name, MODELS[name], train_df, test_df, args, device)
    if args.probe:
        return

    ranked = sorted(
        ((name, payload["macro_f1"]) for name, payload in results.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    summary = {
        "split": "RandomUnderSampler random_state=42; train_test_split test_size=0.2 random_state=42",
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "best": ranked[0][0],
        "models": {name: {k: results[name][k] for k in ("macro_f1", "macro_precision", "macro_recall", "hub_id")} for name in results},
    }
    summary_name = f"summary_{args.run_name}.json" if args.run_name else "summary.json"
    (OUT / summary_name).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[best]", summary["best"], summary["models"], flush=True)
    if not args.skip_ig:
        explain_best(summary["best"], test_df, args, device)


if __name__ == "__main__":
    main()
