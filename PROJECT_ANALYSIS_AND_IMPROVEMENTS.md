# Fake News Detection Project — Analysis & Improvement Plan

---

## Deployment readiness — short answer

**Is it deployment ready?** **No.** It works for local experimentation and demos, but it is **not robust or production-ready** as-is.

**How good is functionality right now?**

| Aspect | Grade | Notes |
|--------|--------|--------|
| **Core ML pipeline** | Good | Train → eval → inference works; multi-language and two backbones. |
| **CLI / local use** | Good | `test.py` supports interactive and batch; model/tokenizer now aligned. |
| **API (Flask)** | Weak | Hardcoded model & path; crashes if checkpoint missing; no config. |
| **Config & env** | Poor | No single config; no `.env`; port/model path not configurable. |
| **Robustness** | Poor | No graceful failure, no request limits, bare `except`, no logging. |
| **Deployability** | Poor | No Docker, no health that reflects model load, no docs for deploy. |

**Summary:** Functionality is **good for development and evaluation**. For **deployment** you need: configurable API, graceful startup/failure, same preprocessing in API, request limits, logging, and (recommended) Docker + health checks.

---

## 1. Project overview

This is a **multilingual (Indian languages) fake news detection** project with:

- **Models**: XLM-RoBERTa, MuRIL, and an ensemble (XLM-R + MuRIL).
- **Data**: Multiple CSV datasets per language (Assamese, Bengali, English, Gujarati, Hindi, Malayalam, Marathi, Punjabi, Tamil, Urdu); Zenodo-style conversion for Hindi .txt.
- **Pipeline**: `prepare_data.py` → `train.py` → `evaluate.py`; inference via `inference.py` (class + Flask API) and `test.py` (CLI).
- **Utils**: Preprocessing, metrics, visualisation (incl. run loading and dashboard).

---

## 2. Bugs and inconsistencies (fix first)

### 2.1 Wrong import in `evaluate.py`

- **Issue**: `evaluate.py` imports `from data.datasets import MultilingualFakeNewsDataset` but the module is `data.dataset` (no `s`).
- **Fix**: Change to `from data.dataset import MultilingualFakeNewsDataset`.

### 2.2 Tokenizer mismatch in `test.py`

- **Issue**: `test.py` always loads `AutoTokenizer.from_pretrained('xlm-roberta-base')` even when the loaded checkpoint is MuRIL. MuRIL needs `google/muril-base-cased`.
- **Fix**: Choose tokenizer from the same `model_type` used for the model (e.g. from CONFIG or from checkpoint path/model type).

### 2.3 Flask API in `inference.py` hardcodes model and path

- **Issue**: The Flask app loads only `XLMRobertaFakeNewsClassifier` and `models/checkpoints/xlm-roberta_best.pt`. If you train MuRIL, the API still uses XLM-R and the wrong checkpoint.
- **Fix**: Make model type and checkpoint path configurable (env vars or config file), and load the matching tokenizer and model class (same pattern as in `test.py`).

### 2.4 `prepare_data.py` uses a single hardcoded file

- **Issue**: It only processes `data/raw/assamese_fake_news_dataset.csv`. The repo has many languages under `data/raw/MainData/`.
- **Fix**: Use `DataPreprocessor.prepare_dataset()` with a list of `(file_path, language)` for all languages you want (e.g. from `MainData`), then split and save train/val/test.

### 2.5 Bare `except` and fragile `history` in `train.py`

- **Issue**: `history` is taken from `locals().get("history", getattr(trainer, "history", {}))`; `trainer.train()` assigns `self.history = history` but does not return it, so `history` in `__main__` is only set when the block runs after `trainer.train()` and the variable is in scope. Fragile and confusing.
- **Fix**: Have `train()` return `(self.model, self.history)` and set `trained_model, history = trainer.train()`. Avoid bare `except:`; catch specific exceptions and log.

### 2.6 `inference.py` model forward return type

- **Issue**: In `FakeNewsDetector.predict()` you do `logits = self.model(input_ids, attention_mask)` then `torch.softmax(logits, dim=1)`. Your custom models return a tensor (logits), so this is correct. If you later switch to a Hugging Face `ForSequenceClassification` that returns a object with `.logits`, the code would still work as long as you handle both (e.g. `logits = outputs.logits if hasattr(outputs, 'logits') else outputs`). Currently fine; only relevant if you change model types.

---

## 3. Code quality and structure

### 3.1 Single entry point and config

- **Issue**: Model type, paths, and hyperparameters are scattered across `train.py`, `evaluate.py`, `test.py`, and inference.
- **Improvement**: Introduce a single `config.py` or `config.yaml` (and load it in train/evaluate/inference/test). Include: `model_type`, `model_name`, `tokenizer_name`, `checkpoint_path`, `max_length`, `batch_size`, `device`, etc.

### 3.2 Duplicate `FakeNewsDetector` logic

- **Issue**: `inference.py` and `test.py` each define or use a detector with similar predict logic; `test.py` has its own `FakeNewsDetector` with different init (checkpoint discovery, model type map).
- **Improvement**: Have one `FakeNewsDetector` in `inference.py` that accepts `model`, `tokenizer`, `device`, `max_length`. A small factory in `inference.py` or a separate `load_model.py` can do checkpoint discovery and load model + tokenizer; `test.py` and Flask should both use this.

### 3.3 Requirements and environment

- **Issue**: `requirements.txt` has no version upper bounds; `wandb` is optional but always listed.
- **Improvement**: Pin major versions (e.g. `torch>=2.0.0,<3.0`) and group optional deps (e.g. `wandb`) in a comment or optional extra. Add a `.env.example` for Flask (port, model path, model type).

### 3.4 Cleanup and small fixes

- Remove or stop committing `tempCodeRunnerFile.py` (and similar) via `.gitignore`.
- In `utils/visualisation.py`, fix the typo in `plot_class_distribution`: the first branch uses `f'{height:.3f}'` for bar counts; should be integer format (e.g. `f'{int(height)}'`) for count bars.
- In `utils/preprocessing.py`, `clean_text` uses `re.sub(r'\s+', '', text)` which removes all spaces; likely you want `re.sub(r'\s+', ' ', text)` to normalize spaces.

---

## 4. Functionality improvements

### 4.1 Data pipeline

- **Multilingual preparation**: One script (e.g. `prepare_data.py`) that:
  - Reads all CSVs from `data/raw/MainData/` (or a configurable list).
  - Maps filename or folder to language code.
  - Uses `DataPreprocessor.prepare_dataset()` and `split_data()` with stratification.
  - Writes `train.csv`, `val.csv`, `test.csv` and optionally a small `data_summary.json` (counts per language and label).

### 4.2 Training

- **CLI and reproducibility**: Add `argparse` (or Hydra/click) to `train.py` for: `--model_type`, `--epochs`, `--batch_size`, `--lr`, `--output_dir`, `--seed`. Save full config (and git commit hash if available) into the run folder.
- **Checkpointing**: Save best and last checkpoint; optionally save optimizer/scheduler state for resume.
- **Early stopping**: Stop when validation metric (e.g. F1 or accuracy) does not improve for N epochs.
- **Class imbalance**: Support weighted loss or oversampling (e.g. `WeightedRandomSampler`) when labels are imbalanced.

### 4.3 Evaluation

- **Fix import**: Use `data.dataset` in `evaluate.py`.
- **Unified metrics**: Use `utils.metrics.MetricsCalculator` in `evaluate.py` so train/eval share the same metric definitions and per-language logic.
- **Test set**: Ensure evaluation is run on the held-out test set and reported once; avoid tuning on test.

### 4.4 Inference and API

- **Single detector**: One `FakeNewsDetector` that takes model + tokenizer; factory that resolves checkpoint from `model_type` and optional path.
- **Flask**: Configurable model type and checkpoint path (env or config); load correct tokenizer; add a simple rate limit and request size limit; return 503 if model fails to load.
- **Preprocessing**: Apply the same `DataPreprocessor.clean_text()` (and optionally language detection) in the inference path so production matches training.

### 4.5 Ensemble

- **Usage**: `models/ensemble_model.py` is not used in train or inference. Add a small script or mode to load two checkpoints (XLM-R and MuRIL), build `EnsembleFakeNewsClassifier`, and run eval or inference. Optionally support different tokenizers (run each model with its tokenizer, then combine logits).

---

## 5. ML and data improvements

### 5.1 Text preprocessing

- **Consistency**: Use the same preprocessing in dataset creation, training, and inference (one place, e.g. `utils.preprocessing`).
- **Fix regex**: Keep spaces: `re.sub(r'\s+', ' ', text).strip()` instead of removing all spaces.
- **Optional**: Add optional transliteration for low-resource scripts if you use a model that expects a specific script.

### 5.2 Dataset

- **Caching**: Tokenization in `MultilingualFakeNewsDataset` is on-the-fly. For large data, consider caching tokenized outputs (e.g. to disk or a cache keyed by hash of text + tokenizer + max_length) to speed up subsequent runs.
- **Dynamic padding**: Use `padding='longest'` in the tokenizer and a custom `collate_fn` that pads to max length in the batch to reduce padding and speed up training.

### 5.3 Training details

- **Epochs**: CONFIG uses `num_epochs: 1`; typically 3–5 epochs are used for such models. Make it configurable.
- **Warmup**: You have `warmup_steps=0`; 10% warmup often helps. Make it configurable (e.g. ratio of total steps).
- **F1 vs accuracy**: For imbalanced fake/real, consider early stopping and checkpoint selection by F1 (macro or weighted) instead of accuracy.

### 5.4 Evaluation

- **Per-language and per-class**: You already have per-language accuracy; ensure per-class metrics (precision/recall/F1 for Fake and Real) are reported per language as well (e.g. using `utils.metrics`).
- **Uncertainty**: You save probabilities; consider reporting entropy or max-prob distribution for reliability.

---

## 6. DevOps and deployment

- **Logging**: Replace ad-hoc `print` with `logging` (one logger per module, configurable level).
- **Tests**: Add minimal unit tests: e.g. load a small CSV, run one batch through model, run `MetricsCalculator` on fixed y_true/y_pred.
- **Docker**: Add a `Dockerfile` that installs dependencies and runs the Flask app (or a single inference server), with config via env.
- **CI**: Optional GitHub Actions to run tests and lint on push.

---

## 7. Quick wins (priority order)

1. Fix `evaluate.py`: `data.datasets` → `data.dataset`.
2. Fix `test.py`: set tokenizer from `model_type` (e.g. MuRIL → `google/muril-base-cased`).
3. Make Flask in `inference.py` load model type and checkpoint from env/config; use correct tokenizer.
4. Fix `prepare_data.py`: use all languages from `MainData` (or a configurable list) via `prepare_dataset()` + split.
5. Fix `utils.preprocessing`: keep spaces in `clean_text` (replace `\s+` with single space).
6. In `train.py`: return `history` from `train()` and assign it in `__main__`; avoid bare `except`.
7. Add a `config.py` (or YAML) and use it in train, evaluate, inference, and test.

After these, you can iterate on: single detector + factory, unified metrics, CLI for train/eval, optional caching and dynamic padding, then Docker and logging.

---

## 8. Improvements to make it robust and deployment-ready

### Must-have (to be deployment-ready) — ✅ implemented

1. **Configurable Flask app** ✅
   - Read `MODEL_TYPE` and `CHECKPOINT_PATH` (and optionally `PORT`, `MAX_LENGTH`) from environment.
   - Load the correct model class and tokenizer for `MODEL_TYPE` (same logic as `test.py`).
   - If checkpoint is missing or load fails, **do not crash**: start the app and set a flag `model_loaded = False`.  
   - **Done**: `inference.py` uses `load_detector()` and env `MODEL_TYPE`, `CHECKPOINT_PATH`, `PORT`, `MAX_LENGTH`.

2. **Graceful failure and health** ✅
   - `/health` returns 200 with `{"status": "healthy", "model_loaded": true/false}`.
   - `/predict` and `/batch_predict` return **503** with a clear message when `model_loaded` is false.
   - Log errors on startup (e.g. missing file, load error) instead of raising and exiting.  
   - **Done**: `/health` returns `model_loaded`; predict endpoints return 503 when not loaded; logging in place.

3. **Preprocessing in API** ✅
   - Run the same `DataPreprocessor.clean_text()` on input text before tokenization so production matches training.  
   - **Done**: `clean_text_for_inference()` in `inference.py` uses `DataPreprocessor().clean_text()`.

4. **Request limits** ✅
   - Reject oversized payloads (e.g. max 1 MB JSON, max 10 items in `texts` for batch, max 50k chars per text) and return 400.  
   - **Done**: `MAX_CONTENT_LENGTH`, `MAX_TEXT_LENGTH`, `MAX_BATCH_SIZE` (env-configurable); 400 with `code` in body.

5. **Environment and docs** ✅
   - Add `.env.example` with `MODEL_TYPE`, `CHECKPOINT_PATH`, `PORT`, `MAX_LENGTH`.
   - In README or PROJECT_ANALYSIS, document how to run the API (env vars, optional Docker).  
   - **Done**: `.env.example` and `DEPLOYMENT.md` added.

### Should-have (robustness and performance)

6. **Logging**: Use Python `logging` in the API (request id, latency, errors) instead of print.
7. **Timeout and batch size**: Configurable timeout per request; cap batch size for `/batch_predict` to avoid OOM.
8. **Single detector factory**: One place that builds model + tokenizer from env; reuse in Flask and `test.py`.
9. **Docker**: `Dockerfile` that installs deps and runs the Flask app with env-based config; document `docker run` with `-e` flags.

### Nice-to-have

10. **Structured errors**: JSON error responses with `{"error": "...", "code": "NO_TEXT"}` etc. for clients.
11. **Metrics**: Optional Prometheus/health metrics (request count, latency percentiles) for monitoring.
12. **Gunicorn**: Run with a production WSGI server (e.g. `gunicorn -w 1 -b 0.0.0.0:5000 inference:app`) in Docker.
