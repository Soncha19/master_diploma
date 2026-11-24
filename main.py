"""Flask report page for MentalBERT: label, top-5 IG words, probability warning."""

import os
from pathlib import Path

import torch
from flask import Flask, render_template, request
from markupsafe import Markup
from transformers import AutoTokenizer
from transformers.models.bert.modeling_bert import BertForSequenceClassification

from explain import aggregate_to_words, attributions_to_html, compute_token_attributions
from plain_reading import doctor_reading
from preprocess import preprocess_text


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
MODEL_DIR = ROOT_DIR / "results" / "mentalbert" / "mentalbert"
PT_PATH = MODEL_DIR / "mentalbert_full_model.pt"

device = torch.device("cpu")
LABELS = {
    0: "ADHD",
    1: "Anxiety",
    2: "Bipolar",
    3: "BPD",
    4: "Depression",
    5: "Normal",
    6: "OCD",
    7: "PTSD",
}
MAX_LENGTH = 256
IG_STEPS = int(os.environ.get("IG_STEPS", "16"))
WARNING_THRESHOLD = float(os.environ.get("WARNING_THRESHOLD", "0.70"))
TOP_K = 5

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
with torch.serialization.safe_globals([BertForSequenceClassification]):
    model = torch.load(PT_PATH, map_location=device, weights_only=False)
model.to(device).eval()


def predict(text: str) -> dict:
    encoded = tokenizer(
        text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits[0]
        probs = torch.softmax(logits, dim=-1)
    order = torch.argsort(probs, descending=True)
    pred = int(order[0].item())
    ranked = [
        {
            "rank": rank,
            "label": LABELS[int(idx)],
            "prob": float(probs[int(idx)].item()),
            "prob_fmt": f"{float(probs[int(idx)].item()):.4f}",
        }
        for rank, idx in enumerate(order[:3].tolist(), start=1)
    ]
    return {
        "pred_id": pred,
        "label": LABELS[pred],
        "prob": float(probs[pred].item()),
        "top3": ranked,
    }


def five_words(text: str, target_class: int) -> tuple[list[dict], str]:
    _, offsets, scores = compute_token_attributions(
        model,
        tokenizer,
        text,
        target_class,
        device,
        n_steps=IG_STEPS,
        max_length=MAX_LENGTH,
        baseline="pad",
    )
    words = aggregate_to_words(text, offsets, scores)
    positive = sorted((item for item in words if item.score > 0), key=lambda item: item.score, reverse=True)
    report = [
        {"word": item.word, "score": f"{item.score:.4f}"}
        for item in positive[:TOP_K]
    ]
    return report, attributions_to_html(words)


@app.route("/")
def home():
    return render_template("index.html", warning_threshold=WARNING_THRESHOLD)


@app.route("/predict", methods=["POST"])
def predict_view():
    raw = request.form["text_input"]
    cleaned = preprocess_text(raw)
    result = predict(cleaned)
    words, highlighted = five_words(cleaned, result["pred_id"])
    low = result["prob"] < WARNING_THRESHOLD
    alt = result["top3"][1]["label"] if len(result["top3"]) > 1 else None
    reading = doctor_reading(
        result["label"],
        f"{result['prob']:.4f}",
        low,
        WARNING_THRESHOLD,
        [item["word"] for item in words],
        alt,
    )
    return render_template(
        "index.html",
        text_input=raw,
        cleaned_text=cleaned,
        prediction=result["label"],
        probability=f"{result['prob']:.4f}",
        is_uncertain=low,
        warning_threshold=WARNING_THRESHOLD,
        top3=result["top3"],
        report_words=words,
        highlighted=Markup(highlighted),
        doctor_text=reading["text"],
        doctor_note=reading["note"],
    )


if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    app.run(host=host, port=port, debug=False, use_reloader=False)
