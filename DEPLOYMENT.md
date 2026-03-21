# Deploying the Fake News Detection API

## Quick start (local)

1. Train a model (e.g. MuRIL): `python train.py` (uses CONFIG in `train.py`).
2. Copy env example and set checkpoint:
   ```bash
   copy .env.example .env
   # Edit .env: set MODEL_TYPE=muril and CHECKPOINT_PATH=models/checkpoints/muril_best.pt
   ```
3. Run the API:
   ```bash
   set PORT=5000
   set MODEL_TYPE=muril
   set CHECKPOINT_PATH=models/checkpoints/muril_best.pt
   python inference.py
   ```
   Or with a `.env` file (install `python-dotenv` and load in app if you want), or set env vars in your shell.

4. Check health: `GET http://localhost:5000/health`  
   Response includes `model_loaded: true/false`. If `false`, the app started but the model failed to load (wrong path or type).

5. Predict: `POST http://localhost:5000/predict` with JSON `{"text": "Your news text here"}`.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_TYPE` | `muril` | `muril` or `xlm-roberta` (must match checkpoint). |
| `CHECKPOINT_PATH` | `models/checkpoints/{MODEL_TYPE}_best.pt` | Path to the `.pt` checkpoint. |
| `PORT` | `5000` | Server port. |
| `MAX_LENGTH` | `512` | Max token length (should match training). |
| `MAX_CONTENT_LENGTH` | `1048576` | Max request body size (bytes). |
| `MAX_TEXT_LENGTH` | `50000` | Max characters per text. |
| `MAX_BATCH_SIZE` | `20` | Max items in `/batch_predict` body. |
| `FLASK_DEBUG` | `0` | Set to `1` for debug mode (do not use in production). |
| `LOG_LEVEL` | `INFO` | Logging level. |

## Behaviour

- **Graceful failure**: If the checkpoint is missing or loading fails, the server still starts. `/health` returns `model_loaded: false`; `/predict` and `/batch_predict` return **503** with `code: MODEL_NOT_LOADED`.
- **Preprocessing**: Input text is cleaned with the same logic as in training before prediction.
- **Limits**: Oversized payloads or text/batch size are rejected with **400** and a clear `code` in the JSON body.

## Production (e.g. Docker)

- Use a production WSGI server, e.g. Gunicorn:
  ```bash
  pip install gunicorn
  gunicorn -w 1 -b 0.0.0.0:5000 --timeout 120 inference:app
  ```
- Keep `FLASK_DEBUG=0`. Set `PORT` and `CHECKPOINT_PATH` (and optionally `MODEL_TYPE`) via the environment.
- For Docker, add a `Dockerfile` that installs dependencies and runs the above; pass env with `-e` or an env file.
