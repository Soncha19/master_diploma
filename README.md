# MentalBERT XAI report

Code for the master's project on decision interpretability for an eight-class MentalBERT classifier.

## Features

- Fine-tune `mental/mental-bert-base-uncased`
- Compare LIME, SHAP, Integrated Gradients, occlusion and a random baseline
- Contrastive, mask-baseline and sequential IG variants
- Shared-word penalty fine-tuning
- Local Flask page: label, top-5 IG words, probability warning, short doctor reading

## Layout expected next to this folder

```
repo-parent/                      # e.g. MKR_MentalBERT/
  psycho-emotional-classifier/    # this package
  data/reddit_mental_preprocessed.csv
  results/mentalbert/             # trained checkpoints (local, not in Git)
  results/benchmark/pool.csv
  results/benchmark_mentalbert/
```

### Weights (`results/mentalbert/`)

Checkpoints stay on disk next to the package; they are not committed to Git (≈418 MB each).
Download folder: https://drive.google.com/drive/folders/1d07AgFZWjoygsIXkXUvPeZGSM-5fn_YD  
Place the contents under `results/mentalbert/` so the paths below match.

| File | Role |
| --- | --- |
| `results/mentalbert/mentalbert/mentalbert_full_model.pt` | Main fine-tuned MentalBERT (8 classes). Used by `main.py` and `benchmark_xai.py`. |
| `results/mentalbert/mentalbert_xai_ft/mentalbert_xai_ft_full_model.pt` | Extra epoch from that checkpoint with a shared-word penalty (`finetune_mentalbert_xai.py`). |
| `results/mentalbert/mentalbert_xai_ownclass/mentalbert_xai_ownclass_full_model.pt` | Same penalty setup with `--exempt-own-class` (class-name tokens for the gold label are not penalized). |

Each of those folders also has Hugging Face-style files (`config.json`, `model.safetensors`, tokenizer).

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Use `transformers==4.46.3` with the saved `.pt` checkpoints.

## Run the report page

```bash
set LLM_PROVIDER=ollama
set OLLAMA_MODEL=llama3.2
set WARNING_THRESHOLD=0.70
set IG_STEPS=16
python main.py
```

Open http://127.0.0.1:5000

If Ollama is unavailable, the doctor paragraph falls back to a template built from the same fields. The raw post is never sent to the LLM.

## Scripts

| Script | Role |
| --- | --- |
| `train_mentalbert.py` | Train MentalBERT |
| `benchmark_xai.py` | Explanation benchmark on the pool |
| `analyze_benchmark.py` | Aggregate metrics |
| `explain.py` | Integrated Gradients helpers |
| `contrastive_ig_mentalbert.py` | Contrastive IG |
| `mask_ig_mentalbert.py` | Mask-baseline IG |
| `sequential_ig_mentalbert.py` | Sequential IG |
| `finetune_mentalbert_xai.py` | Shared-word penalty epoch |
| `main.py` | Local report page |
| `plain_reading.py` | Doctor reading from report fields |
| `preprocess.py` | Lowercase, clean, lemmatize |

## Note

Predictions are not a clinical diagnosis.
