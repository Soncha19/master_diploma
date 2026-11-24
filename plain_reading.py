"""Plain-language doctor reading for a MentalBERT report.

Builds a short paragraph from the label, probability, warning flag and the
five IG words. The raw post is never sent to the LLM. If the model cites a
Latin word outside the allowed list, the template is used instead.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request


_LATIN = re.compile(r"[A-Za-z][A-Za-z']*")
_LABELS = (
    "adhd",
    "anxiety",
    "bipolar",
    "bpd",
    "depression",
    "normal",
    "ocd",
    "ptsd",
    "mentalbert",
)


def allowed_words(label: str, words: list[str]) -> set[str]:
    allowed = {label.casefold(), *_LABELS}
    allowed.update(word.casefold() for word in words if word.strip())
    return allowed


def extra_latin(text: str, allowed: set[str]) -> list[str]:
    found = {match.group(0).casefold() for match in _LATIN.finditer(text)}
    return sorted(word for word in found if word not in allowed)


def covers_report(text: str, label: str, words: list[str]) -> bool:
    """Require the label and every report word; reject free paraphrase of the post."""
    folded = text.casefold()
    if label.casefold() not in folded:
        return False
    for word in words:
        pattern = rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])"
        if re.search(pattern, text, flags=re.IGNORECASE) is None:
            return False
    return True


def template_reading(
    label: str,
    probability: str,
    is_uncertain: bool,
    threshold: float,
    words: list[str],
    alt_label: str | None,
) -> str:
    if words:
        listed = ", ".join(words)
        support = (
            f"Пояснення спирається на слова {listed}. "
            "Саме вони дали найбільший додатний внесок у цю мітку."
        )
    else:
        support = "Слів із додатним внеском для цього допису немає."
    if alt_label and alt_label.casefold() != label.casefold():
        alternative = f" Наступний за ймовірністю клас — {alt_label}; він мітку не замінює."
    else:
        alternative = ""
    if is_uncertain:
        caution = (
            f" Ймовірність нижча за робочий поріг {threshold:.2f}, "
            "тому цей звіт треба читати обережно."
        )
    else:
        caution = f" Ймовірність не нижча за робочий поріг {threshold:.2f}."
    return (
        f"Модель віднесла допис до класу {label} з імовірністю {probability}. "
        "Це не діагноз і не порада щодо лікування. "
        f"{support}{alternative}{caution}"
    )


def _ollama_base() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def ollama_ready() -> bool:
    try:
        with urllib.request.urlopen(f"{_ollama_base()}/api/tags", timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _call_ollama(system: str, user: str) -> str:
    model = os.environ.get("OLLAMA_MODEL", "llama3.2").strip()
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "options": {"temperature": 0.2},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{_ollama_base()}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return str(data["message"]["content"]).strip()


def _prompt(label: str, probability: str, is_uncertain: bool, threshold: float, words: list[str], alt_label: str | None) -> tuple[str, str]:
    listed = ", ".join(words) if words else "(немає)"
    system = (
        "Ти пояснюєш уже готовий звіт класифікатора простою українською для лікаря. "
        "Пиши від третьої особи про модель, не від імені пацієнта. "
        "Це не діагноз і не порада щодо лікування. "
        "Обов'язково назви клас і кожне дозволене слово дослівно латиницею. "
        "Не додавай інших англійських слів, не цитуй допис, не згадуй ліки, "
        "критерії діагнозу й інші моделі. Напиши чотири короткі речення."
    )
    user = (
        f"Мітка: {label}. Імовірність: {probability}. "
        f"Поріг попередження: {threshold:.2f}. Попередження: {'так' if is_uncertain else 'ні'}. "
        f"Дозволені слова: {listed}. "
        f"Наступний клас: {alt_label or 'немає'}. Він мітку не замінює."
    )
    return system, user


def doctor_reading(
    label: str,
    probability: str,
    is_uncertain: bool,
    threshold: float,
    words: list[str],
    alt_label: str | None,
) -> dict:
    """Return paragraph text, source name and a short UI note."""
    fallback = template_reading(label, probability, is_uncertain, threshold, words, alt_label)
    provider = (os.environ.get("LLM_PROVIDER") or "ollama").strip().lower()
    if provider in {"off", "none", "template", "rule", "rules"} or not ollama_ready():
        return {
            "text": fallback,
            "source": "template",
            "note": "Локальна мовна модель не відповідає, тому абзац зібрано шаблоном із тих самих полів. Це не діагноз.",
        }
    system, user = _prompt(label, probability, is_uncertain, threshold, words, alt_label)
    try:
        raw = _call_ollama(system, user)
    except (urllib.error.URLError, TimeoutError, OSError, KeyError, json.JSONDecodeError, ValueError):
        return {
            "text": fallback,
            "source": "template",
            "note": "Локальна мовна модель не відповіла, тому абзац зібрано шаблоном із тих самих полів. Це не діагноз.",
        }
    extra = extra_latin(raw, allowed_words(label, words))
    if extra:
        note = "Вільний абзац назвав слово поза п’ятіркою, тому його замінено шаблоном. Це не діагноз."
    elif not covers_report(raw, label, words):
        note = "Вільний абзац не назвав мітку або слово зі звіту, тому його замінено шаблоном. Це не діагноз."
    else:
        note = ""
    if note:
        return {"text": fallback, "source": "template", "note": note}
    return {
        "text": raw,
        "source": "ollama",
        "note": "Абзац зібрала локальна мовна модель з мітки і п’яти слів. Це не діагноз і не порада. Допис у модель не передавався.",
    }
