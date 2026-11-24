"""Integrated Gradients helpers for MentalBERT sequence classification."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Iterable

import torch
from captum.attr import IntegratedGradients


@dataclass
class WordAttribution:
    word: str
    score: float


def _encoder_module(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "bert"):
        return model.bert
    raise AttributeError(f"Unsupported model type: {type(model)}")


def _word_embedding_layer(model: torch.nn.Module) -> torch.nn.Module:
    return _encoder_module(model).embeddings.word_embeddings


def _special_token_ids(tokenizer) -> set[int]:
    ids = set()
    for attr in ("pad_token_id", "cls_token_id", "sep_token_id", "bos_token_id", "eos_token_id", "mask_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            ids.add(int(value))
    return ids


def _build_baseline_ids(input_ids: torch.Tensor, tokenizer, kind: str = "pad") -> torch.Tensor:
    """Keep special tokens; replace content tokens with pad or [MASK]."""
    if kind == "mask":
        fill_id = tokenizer.mask_token_id
        if fill_id is None:
            raise ValueError("tokenizer has no mask token")
    elif kind == "pad":
        fill_id = tokenizer.pad_token_id
        if fill_id is None:
            fill_id = 0
    else:
        raise ValueError(f"unknown IG baseline: {kind}")
    special_ids = _special_token_ids(tokenizer)
    baseline = torch.full_like(input_ids, int(fill_id))
    for token_id in special_ids:
        baseline = torch.where(input_ids == token_id, input_ids, baseline)
    return baseline


def compute_token_attributions(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    target_class: int,
    device: torch.device,
    n_steps: int = 32,
    max_length: int = 256,
    contrast_class: int | None = None,
    baseline: str = "pad",
) -> tuple[list[str], list[tuple[int, int]], list[float]]:
    """
    Integrated Gradients over word-embedding vectors (inputs_embeds).

    Uses embedding-space IG on inputs_embeds for stable sequence length.

    contrast_class, when set, attributes logit[target_class] - logit[contrast_class]
    instead of the target logit alone. The rival class is fixed by the caller
    from the original prediction and is not recomputed along the path.

    baseline is "pad" or "mask". Content tokens start from that token; special
    tokens stay in place. The default keeps the original pad path.

    Returns tokens, char offset mapping, and per-token attribution scores.
    """
    model.eval()
    encoding = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offsets = [(int(s), int(e)) for s, e in encoding["offset_mapping"][0].tolist()]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    baseline_ids = _build_baseline_ids(input_ids, tokenizer, baseline)

    embedding_layer = _word_embedding_layer(model)
    input_embeds = embedding_layer(input_ids)
    baseline_embeds = embedding_layer(baseline_ids)

    def forward_func(inputs_embeds: torch.Tensor) -> torch.Tensor:
        # Captum batches n_steps along dim 0; expand the mask to match.
        batch = inputs_embeds.size(0)
        mask = attention_mask.expand(batch, -1)
        return model(inputs_embeds=inputs_embeds, attention_mask=mask).logits

    if contrast_class is None:
        ig = IntegratedGradients(forward_func)
        attributions = ig.attribute(
            inputs=input_embeds,
            baselines=baseline_embeds,
            target=int(target_class),
            n_steps=n_steps,
            internal_batch_size=4,
        )
    else:
        rival = int(contrast_class)

        def forward_margin(inputs_embeds: torch.Tensor) -> torch.Tensor:
            logits = forward_func(inputs_embeds)
            return logits[:, int(target_class)] - logits[:, rival]

        ig = IntegratedGradients(forward_margin)
        attributions = ig.attribute(
            inputs=input_embeds,
            baselines=baseline_embeds,
            n_steps=n_steps,
            internal_batch_size=4,
        )

    # Sum over embedding dim -> one score per token
    token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu()
    special_ids = _special_token_ids(tokenizer)
    scores: list[float] = []
    for idx, token_id in enumerate(input_ids[0].tolist()):
        if token_id in special_ids or offsets[idx] == (0, 0):
            scores.append(0.0)
        else:
            scores.append(float(token_scores[idx].item()))
    return tokens, offsets, scores


def compute_sequential_token_attributions(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    target_class: int,
    device: torch.device,
    n_steps: int = 8,
    max_length: int = 256,
    baseline: str = "pad",
) -> tuple[list[str], list[tuple[int, int]], list[float]]:
    """Sequential IG: one integral per content token (Enguehard, 2023).

    The baseline for a token is the original sentence with only that token
    replaced. The path moves this token from the baseline embedding to the
    real one, and every other token stays at the input. The published setup
    uses a mask baseline; ``baseline`` defaults to pad so a comparison with
    ordinary IG changes only the path.

    Scores are the sum over the embedding axis. A global rescaling would
    not change the ranking, so the result is not L2-normalised.
    """
    model.eval()
    encoding = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offsets = [(int(s), int(e)) for s, e in encoding["offset_mapping"][0].tolist()]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    baseline_ids = _build_baseline_ids(input_ids, tokenizer, baseline)

    embedding_layer = _word_embedding_layer(model)
    input_embeds = embedding_layer(input_ids).detach()
    baseline_embeds = embedding_layer(baseline_ids).detach()

    def forward_func(inputs_embeds: torch.Tensor) -> torch.Tensor:
        batch = inputs_embeds.size(0)
        mask = attention_mask.expand(batch, -1)
        return model(inputs_embeds=inputs_embeds, attention_mask=mask).logits

    ig = IntegratedGradients(forward_func)
    special_ids = _special_token_ids(tokenizer)
    scores = [0.0] * input_ids.size(1)
    for idx, token_id in enumerate(input_ids[0].tolist()):
        if token_id in special_ids or offsets[idx] == (0, 0):
            continue
        if torch.allclose(input_embeds[0, idx], baseline_embeds[0, idx]):
            continue
        baselines = input_embeds.clone()
        baselines[0, idx] = baseline_embeds[0, idx]
        attributions = ig.attribute(
            inputs=input_embeds,
            baselines=baselines,
            target=int(target_class),
            n_steps=n_steps,
            internal_batch_size=n_steps,
        )
        scores[idx] = float(attributions[0, idx].sum().detach().cpu().item())
    return tokens, offsets, scores


def _whitespace_word_spans(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def aggregate_to_words(
    text: str,
    offsets: Sequence[tuple[int, int]],
    token_scores: Sequence[float],
) -> list[WordAttribution]:
    """Map subword attributions onto whitespace-separated words via char offsets."""
    words = _whitespace_word_spans(text)
    if not words:
        return []

    aggregated = [0.0] * len(words)
    for (start, end), score in zip(offsets, token_scores):
        if start == end:
            continue
        mid = (start + end) / 2
        for i, (_, w_start, w_end) in enumerate(words):
            if w_start <= mid < w_end:
                aggregated[i] += float(score)
                break

    return [WordAttribution(word=w, score=s) for (w, _, _), s in zip(words, aggregated)]


def _normalize_for_color(scores: Iterable[float]) -> list[float]:
    values = list(scores)
    if not values:
        return []
    max_abs = max(abs(v) for v in values) or 1.0
    return [v / max_abs for v in values]


def attributions_to_html(items: Sequence[WordAttribution]) -> str:
    """Render word attributions as colored HTML spans (green=support, red=oppose)."""
    if not items:
        return "<em>No tokens to explain.</em>"

    norms = _normalize_for_color(item.score for item in items)
    parts: list[str] = []
    for item, norm in zip(items, norms):
        intensity = min(abs(norm), 1.0)
        alpha = 0.12 + 0.55 * intensity
        if norm >= 0:
            color = f"rgba(34, 139, 34, {alpha:.3f})"
        else:
            color = f"rgba(178, 34, 34, {alpha:.3f})"
        safe_word = html.escape(item.word)
        parts.append(
            f'<span class="ig-token" style="background-color:{color}" '
            f'title="IG: {item.score:.4f}">{safe_word}</span>'
        )
    return " ".join(parts)


