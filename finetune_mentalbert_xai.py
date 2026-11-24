"""Continue MentalBERT from the saved checkpoint with a shared-word penalty.

Integrated Gradients on the training split builds a list of mood words that
several confused classes rely on. One extra epoch then keeps the gold-class
logit high after those words are replaced with [MASK]. Weights and metrics
go to a new folder. The original mentalbert_full_model.pt is not overwritten.

Classification metrics are macro-F1, macro precision, macro recall and the
confusion matrix on the same 6538-text test split.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from benchmark_xai import DATA, LABELS, reproduce_split
from explain import aggregate_to_words, compute_token_attributions
from train_mentalbert import classification_scores, save_confusion, set_seed

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "results" / "mentalbert"
SRC = OUT / "mentalbert" / "mentalbert_full_model.pt"
SRC_DIR = OUT / "mentalbert"
DEST = OUT / "mentalbert_xai_ft"
WORD_LISTS = DEST / "word_lists.json"

CONFUSED = ("Anxiety", "Bipolar", "BPD", "Depression", "OCD", "PTSD")
MINE_PER_CLASS = 80
IG_STEPS = 8
MAX_LENGTH = 128
BATCH = 8
EPOCHS = 1
LR = 1e-5
PENALTY = 0.5
SEED = 42


class EncodedPairs(Dataset):
    def __init__(self, input_ids, attention_mask, masked_ids, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.masked_ids = masked_ids
        self.labels = labels

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "masked_ids": self.masked_ids[idx],
            "labels": self.labels[idx],
        }


def load_model(device: torch.device):
    model = torch.load(SRC, map_location=device, weights_only=False)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(SRC_DIR)
    return model, tokenizer


def mine_word_lists(model, tokenizer, train_df, device) -> dict:
    if WORD_LISTS.exists():
        payload = json.loads(WORD_LISTS.read_text(encoding="utf-8"))
        print(f"[mine] reuse {WORD_LISTS} shared={len(payload['shared'])}", flush=True)
        return payload

    DEST.mkdir(parents=True, exist_ok=True)
    model.eval()
    totals: dict[str, dict[str, float]] = {name: defaultdict(float) for name in CONFUSED}
    counts: dict[str, dict[str, int]] = {name: defaultdict(int) for name in CONFUSED}
    seen = {name: 0 for name in CONFUSED}
    started = time.time()
    done = 0
    target = MINE_PER_CLASS * len(CONFUSED)
    for name in CONFUSED:
        label_id = LABELS.index(name)
        part = train_df[train_df["label"] == label_id]
        sample = part.sample(n=min(MINE_PER_CLASS, len(part)), random_state=SEED)
        for _, row in sample.iterrows():
            text = " ".join(str(row["text"]).split()[:110])
            types = list(dict.fromkeys(re.findall(r"\S+", text.lower())))
            _, offsets, scores = compute_token_attributions(
                model,
                tokenizer,
                text,
                label_id,
                device,
                n_steps=IG_STEPS,
                max_length=MAX_LENGTH,
                baseline="pad",
            )
            acc: dict[str, float] = defaultdict(float)
            for item in aggregate_to_words(text, offsets, scores):
                acc[item.word.lower()] += float(item.score)
            for word in types:
                value = acc.get(word, 0.0)
                if value > 0:
                    totals[name][word] += value
                    counts[name][word] += 1
            seen[name] += 1
            done += 1
            if done == 1 or done % 20 == 0 or done == target:
                print(f"[mine] {done}/{target} {name} elapsed {time.time() - started:.0f}s", flush=True)

    top: dict[str, list[str]] = {}
    membership: dict[str, set[str]] = defaultdict(set)
    for name in CONFUSED:
        means = []
        for word, total in totals[name].items():
            if counts[name][word] < 3 or len(word) < 4:
                continue
            means.append((total / counts[name][word], word))
        means.sort(reverse=True)
        words = [word for _, word in means[:25]]
        top[name] = words
        for word in words:
            membership[word].add(name)
    shared = sorted(word for word, owners in membership.items() if len(owners) >= 3)
    specific = {
        name: [word for word in words if word not in shared]
        for name, words in top.items()
    }
    payload = {
        "mine_per_class": MINE_PER_CLASS,
        "ig_steps": IG_STEPS,
        "target": "gold class logit",
        "shared": shared,
        "specific": specific,
        "top": top,
        "n_seen": seen,
    }
    WORD_LISTS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[mine] shared {len(shared)}: {', '.join(shared[:40])}", flush=True)
    return payload


def mask_row(text: str, input_ids: torch.Tensor, offsets, shared: set[str], mask_id: int, special_ids: set[int]) -> torch.Tensor:
    spans = [(match.start(), match.end()) for match in re.finditer(r"\S+", text.lower()) if match.group() in shared]
    masked = input_ids.clone()
    for index, (start, end) in enumerate(offsets):
        start, end = int(start), int(end)
        token_id = int(masked[index])
        if start == end or token_id in special_ids:
            continue
        if any(not (end <= left or start >= right) for left, right in spans):
            masked[index] = mask_id
    return masked


def words_for_label(label_id: int, shared: set[str], exempt_own: bool) -> set[str]:
    """Drop a shared word when it is the name of this text's own class."""
    if not exempt_own:
        return shared
    own = LABELS[int(label_id)].lower()
    return {word for word in shared if word != own}


def build_loaders(tokenizer, train_df, test_df, shared: set[str], exempt_own: bool = False):
    def encode(frame):
        texts = [str(text) for text in frame["text"]]
        label_ids = frame["label"].astype(int).tolist()
        encoded = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        mask_id = int(tokenizer.mask_token_id)
        special_ids = {
            int(token_id)
            for token_id in (
                tokenizer.pad_token_id,
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
                tokenizer.mask_token_id,
            )
            if token_id is not None
        }
        masked_rows = [
            mask_row(
                texts[i],
                encoded["input_ids"][i],
                encoded["offset_mapping"][i].tolist(),
                words_for_label(label_ids[i], shared, exempt_own),
                mask_id,
                special_ids,
            )
            for i in range(len(texts))
        ]
        labels = torch.tensor(label_ids, dtype=torch.long)
        return EncodedPairs(
            encoded["input_ids"],
            encoded["attention_mask"],
            torch.stack(masked_rows),
            labels,
        )

    print("[data] encoding train", flush=True)
    train_ds = encode(train_df)
    print("[data] encoding test", flush=True)
    test_ds = encode(test_df)
    changed = int((train_ds.input_ids != train_ds.masked_ids).any(dim=1).sum())
    print(f"[data] train texts with a masked shared word: {changed}/{len(train_ds)}", flush=True)
    return (
        DataLoader(train_ds, batch_size=BATCH, shuffle=True),
        DataLoader(test_ds, batch_size=BATCH, shuffle=False),
    )


def train(model, train_loader, test_loader, device) -> dict:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    steps = max(1, len(train_loader))
    warmup = max(1, int(0.06 * steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return LR * (step + 1) / warmup
        remain = max(1, steps - warmup)
        return LR * max(0.0, (steps - step) / remain)

    losses = []
    penalties = []
    started = time.perf_counter()
    for step, batch in enumerate(train_loader, start=1):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        optimizer.zero_grad(set_to_none=True)
        labels = batch["labels"].to(device)
        input_ids = batch["input_ids"].to(device)
        masked_ids = batch["masked_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        masked_logits = model(input_ids=masked_ids, attention_mask=attention_mask).logits
        gold_orig = outputs.logits.gather(1, labels[:, None]).squeeze(1).detach()
        gold_masked = masked_logits.gather(1, labels[:, None]).squeeze(1)
        penalty = (gold_orig - gold_masked).clamp(min=0).mean()
        loss = outputs.loss + PENALTY * penalty
        loss.backward()
        optimizer.step()
        losses.append(float(outputs.loss.item()))
        penalties.append(float(penalty.item()))
        if step == 1 or step % 100 == 0:
            print(
                f"[train] step {step}/{steps} ce {losses[-1]:.4f} penalty {penalties[-1]:.4f} "
                f"elapsed {(time.perf_counter() - started) / 60:.1f} min",
                flush=True,
            )

    model.eval()
    gold_parts, pred_parts = [], []
    with torch.no_grad():
        for batch in test_loader:
            gold_parts.append(batch["labels"].numpy())
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            pred_parts.append(logits.argmax(dim=-1).cpu().numpy())
    y_true = np.concatenate(gold_parts)
    y_pred = np.concatenate(pred_parts)
    scores = classification_scores(y_true, y_pred)
    scores["accuracy"] = float((y_true == y_pred).mean())
    scores["train_loss"] = float(np.mean(losses))
    scores["train_penalty"] = float(np.mean(penalties))
    scores["epoch_minutes"] = (time.perf_counter() - started) / 60
    scores["n_test"] = int(len(y_true))
    print(
        f"[train] macro-F1 {scores['macro_f1']:.3f} P {scores['macro_precision']:.3f} "
        f"R {scores['macro_recall']:.3f} acc {scores['accuracy']:.3f} "
        f"({scores['epoch_minutes']:.1f} min)",
        flush=True,
    )
    return scores


def compare_baseline(scores: dict) -> dict:
    baseline = json.loads((SRC_DIR / "metrics.json").read_text(encoding="utf-8"))
    base_cm = np.array(baseline["confusion_matrix"])
    new_cm = np.array(scores["confusion_matrix"])
    watched = []
    for src_name, dst_name in (
        ("Bipolar", "Depression"),
        ("BPD", "Depression"),
        ("Anxiety", "Depression"),
        ("PTSD", "Depression"),
        ("OCD", "Anxiety"),
    ):
        i, j = LABELS.index(src_name), LABELS.index(dst_name)
        watched.append(
            {
                "pair": f"{src_name}->{dst_name}",
                "baseline": int(base_cm[i, j]),
                "finetuned": int(new_cm[i, j]),
                "support": int(base_cm[i].sum()),
            }
        )
    per_class = []
    for item, old in zip(scores["per_class"], baseline["per_class"]):
        per_class.append(
            {
                "label": item["label"],
                "f1": item["f1"],
                "baseline_f1": old["f1"],
                "delta_f1": item["f1"] - old["f1"],
            }
        )
    return {
        "baseline_macro_f1": baseline["macro_f1"],
        "baseline_macro_precision": baseline["macro_precision"],
        "baseline_macro_recall": baseline["macro_recall"],
        "delta_macro_f1": scores["macro_f1"] - baseline["macro_f1"],
        "pairs": watched,
        "per_class_f1": per_class,
    }


def main() -> None:
    import argparse

    global DEST, WORD_LISTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="mentalbert_xai_ft")
    parser.add_argument("--word-lists", default="")
    parser.add_argument("--exempt-own-class", action="store_true")
    args = parser.parse_args()
    DEST = OUT / args.run_name
    WORD_LISTS = Path(args.word_lists) if args.word_lists else DEST / "word_lists.json"

    set_seed(SEED)
    random.seed(SEED)
    device = torch.device("cpu")
    if not SRC.exists():
        raise SystemExit(f"missing checkpoint {SRC}")
    DEST.mkdir(parents=True, exist_ok=True)
    metrics_path = DEST / "metrics.json"
    if metrics_path.exists():
        print(f"[skip] {metrics_path} already exists", flush=True)
        return

    train_df, test_df = reproduce_split(DATA)
    print(f"[data] train {len(train_df)} test {len(test_df)} exempt_own={args.exempt_own_class}", flush=True)
    model, tokenizer = load_model(device)
    lists = mine_word_lists(model, tokenizer, train_df, device)
    shared = set(lists["shared"])
    if len(shared) < 5:
        raise SystemExit(f"shared word list is too small: {sorted(shared)}")
    if args.exempt_own_class:
        kept = sorted(word for word in shared if word in {name.lower() for name in LABELS})
        print(f"[mask] own-class words stay visible: {', '.join(kept)}", flush=True)

    train_loader, test_loader = build_loaders(
        tokenizer, train_df, test_df, shared, exempt_own=args.exempt_own_class
    )
    scores = train(model, train_loader, test_loader, device)
    comparison = compare_baseline(scores)
    scores.update(
        {
            "hub_id": "mental/mental-bert-base-uncased",
            "init": str(SRC),
            "epochs_ran": EPOCHS,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH,
            "lr": LR,
            "penalty": PENALTY,
            "shared_words": sorted(shared),
            "exempt_own_class": args.exempt_own_class,
            "n_train": int(len(train_df)),
            "comparison": comparison,
        }
    )

    model.save_pretrained(DEST)
    tokenizer.save_pretrained(DEST)
    pt_path = DEST / f"{args.run_name}_full_model.pt"
    torch.save(model, pt_path)
    metrics_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    save_confusion(args.run_name, scores)
    print(f"[save] {pt_path}", flush=True)
    print(
        f"[compare] macro-F1 {scores['macro_f1']:.3f} vs {comparison['baseline_macro_f1']:.3f} "
        f"delta {comparison['delta_macro_f1']:+.3f}",
        flush=True,
    )
    for pair in comparison["pairs"]:
        print(
            f"[compare] {pair['pair']} {pair['baseline']} -> {pair['finetuned']} of {pair['support']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
