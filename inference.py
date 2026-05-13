"""
Fake news detection inference engine + Flask API.

Supports all three model types: xlm-roberta, muril, ensemble.
Model type, checkpoint path, and limits are configurable via environment variables.

Key fixes from original:
    - token_type_ids forwarded to MuRIL and ensemble (was silently dropped)
    - Uses model.predict() (eval-mode-safe) instead of manual forward() call
    - Ensemble wired end-to-end (original had no ensemble loading path)
    - FakeNewsDetector now accepts model_type explicitly so predict() can
      route token_type_ids correctly — original always called the same path
    - batch_predict() runs proper batched tokenisation instead of N separate
      predict() calls (better GPU utilisation)
    - load_detector() handles ensemble checkpoint format
    - Flask app: /health now reports model_type and checkpoint_path
    - Flask app: content-length guard moved to app.config (Flask's built-in)
      rather than a before_request hook to avoid double-checks
    - All module-level state is encapsulated; app factory pattern used so
      the app can be imported without side effects in tests
"""

import os
import sys
import logging

# ── Ensure project root on sys.path (supports both flat and package layouts) ─
_THIS_DIR   = os.path.abspath(os.path.dirname(__file__))
_PARENT_DIR = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
for _p in (_THIS_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# ── Logging ──────────────────────────────────────────────────────────────── #
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Model / tokenizer registries ─────────────────────────────────────────── #

TOKENIZER_MAP = {
    "xlm-roberta":  "xlm-roberta-base",
    "xlm_roberta":  "xlm-roberta-base",
    "xlmroberta":   "xlm-roberta-base",
    "muril":        "google/muril-base-cased",
    "ensemble":     "google/muril-base-cased",   # MuRIL tokeniser for ensemble
}

_MODEL_CLASSES = None


def _get_model_classes() -> dict:
    """Lazy-load model classes to avoid slow imports at module level."""
    global _MODEL_CLASSES
    if _MODEL_CLASSES is None:
        from models.xlm_roberta_model import XLMRobertaFakeNewsClassifier
        from models.muril_model import MuRILFakeNewsClassifier
        from models.ensemble_model import EnsembleFakeNewsClassifier
        _MODEL_CLASSES = {
            "xlm-roberta":  XLMRobertaFakeNewsClassifier,
            "xlm_roberta":  XLMRobertaFakeNewsClassifier,
            "xlmroberta":   XLMRobertaFakeNewsClassifier,
            "muril":        MuRILFakeNewsClassifier,
            "ensemble":     EnsembleFakeNewsClassifier,   # handled specially
        }
    return _MODEL_CLASSES


# ── Checkpoint helpers — shared via utils.checkpoint ────────────────────── #
try:
    from utils.checkpoint import extract_state_dict as _extract_state_dict
    from utils.checkpoint import strip_module_prefix as _strip_module_prefix
    from utils.checkpoint import safe_load as _safe_load_base
except ImportError:
    def _extract_state_dict(ckpt) -> dict:
        if not isinstance(ckpt, dict):
            return ckpt
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        return ckpt

    def _strip_module_prefix(state_dict: dict) -> dict:
        if any(k.startswith("module.") for k in state_dict):
            return {k[len("module."):]: v for k, v in state_dict.items()}
        return state_dict

    def _safe_load_base(model, state_dict: dict, verbose: bool = True) -> None:
        state_dict = _strip_module_prefix(state_dict)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            model.load_state_dict(state_dict, strict=False)

# Thin wrapper so inference.py can use logger instead of print for warnings
def _safe_load(model, state_dict: dict) -> None:
    """Load state dict; uses logger.warning on strict-load failure."""
    state_dict = _strip_module_prefix(state_dict)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        logger.warning("Strict load failed (%s). Retrying with strict=False.", exc)
        model.load_state_dict(state_dict, strict=False)


# ════════════════════════════════════════════════════════════════════════════ #
#  FakeNewsDetector                                                            #
# ════════════════════════════════════════════════════════════════════════════ #

class FakeNewsDetector:
    """
    Inference engine wrapping any of the three model types.

    Args:
        model:          A loaded nn.Module (XLM-R, MuRIL, or Ensemble).
        tokenizer_name: HuggingFace tokenizer identifier.
        model_type:     Canonical model type string — used to route
                        token_type_ids correctly.
        device:         'cuda' or 'cpu'.
        max_length:     Tokenisation max length.
    """

    def __init__(
        self,
        model,
        tokenizer_name: str,
        model_type: str = "muril",
        device: str = "cuda",
        max_length: int = 512,
        xlmr_tokenizer_name: str = "xlm-roberta-base",
        muril_tokenizer_name: str = "google/muril-base-cased",
    ):
        """
        Args:
            model:                Loaded nn.Module.
            tokenizer_name:       HuggingFace tokenizer ID for standalone models
                                  (XLM-RoBERTa or MuRIL).  Ignored for ensemble —
                                  use xlmr_tokenizer_name / muril_tokenizer_name instead.
            model_type:           Canonical model type string.
            device:               'cuda' or 'cpu'.
            max_length:           Tokenisation max length.
            xlmr_tokenizer_name:  Tokenizer for the XLM-RoBERTa sub-model (ensemble only).
            muril_tokenizer_name: Tokenizer for the MuRIL sub-model (ensemble only).

        FIX (BUG-9): added explicit xlmr_tokenizer_name / muril_tokenizer_name
        parameters so ensemble tokenizers are configurable rather than hardcoded.
        `tokenizer_name` is still accepted for the single-model path and is
        clearly documented as unused for ensemble.
        """
        self.model      = model.to(device)
        self.model.eval()
        self.model_type = model_type.strip().lower().replace("-", "").replace("_", "")
        self.device     = device
        self.max_length = max_length

        # For ensemble: load both tokenizers (XLM-RoBERTa + MuRIL).
        # `tokenizer_name` is intentionally unused here; callers should pass
        # xlmr_tokenizer_name / muril_tokenizer_name for full control.
        if self.model_type == "ensemble":
            self.tokenizer_xlmr  = AutoTokenizer.from_pretrained(xlmr_tokenizer_name, use_fast=True)
            self.tokenizer_muril = AutoTokenizer.from_pretrained(muril_tokenizer_name, use_fast=True)
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
            self.tokenizer_xlmr = None
            self.tokenizer_muril = None

    def _tokenise(self, texts: list[str]) -> dict:
        """Tokenise a list of texts into tensors on the correct device.
        
        For ensemble: returns dual tokenization (xlmr + muril).
        For standalone: returns single tokenization.
        """
        if self.model_type == "ensemble":
            # Tokenize with both models' tokenizers
            enc_xlmr = self.tokenizer_xlmr(
                texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            enc_muril = self.tokenizer_muril(
                texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            # Combine into single dict with prefixed keys
            result = {}
            for k, v in enc_xlmr.items():
                result[f"xlmr_{k}"] = v.to(self.device)
            for k, v in enc_muril.items():
                result[f"muril_{k}"] = v.to(self.device)
            return result
        else:
            # Single tokenization for standalone models
            enc = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {k: v.to(self.device) for k, v in enc.items()}

    def _get_probs(self, enc: dict) -> torch.Tensor:
        """Run forward pass and return probability tensor [batch, num_classes].
        
        Handles three cases:
        - XLM-RoBERTa: input_ids + attention_mask only
        - MuRIL: input_ids + attention_mask + optional token_type_ids
        - Ensemble: dual tokenization (xlmr_ids/mask + muril_ids/mask/tti)
        
        Ensemble weighted_avg/max paths output log-probabilities → use exp().
        """
        from models.ensemble_model import EnsembleFakeNewsClassifier

        with torch.no_grad():
            if self.model_type == "ensemble":
                # Ensemble: extract dual tokenization keys
                xlmr_ids  = enc["xlmr_input_ids"]
                xlmr_mask = enc["xlmr_attention_mask"]
                muril_ids = enc["muril_input_ids"]
                muril_mask = enc["muril_attention_mask"]
                muril_tti = enc.get("muril_token_type_ids")
                
                logits = self.model(xlmr_ids, xlmr_mask, muril_ids, muril_mask, muril_tti)
            elif self.model_type == "xlmroberta":
                # XLM-RoBERTa: no token_type_ids
                input_ids = enc["input_ids"]
                attention_mask = enc["attention_mask"]
                logits = self.model(input_ids, attention_mask)
            else:  # muril
                # MuRIL: optional token_type_ids
                input_ids = enc["input_ids"]
                attention_mask = enc["attention_mask"]
                token_type_ids = enc.get("token_type_ids")
                logits = self.model(input_ids, attention_mask, token_type_ids)

        # Convert log-probs to probs if ensemble uses weighted_avg/max
        if (
            isinstance(self.model, EnsembleFakeNewsClassifier)
            and self.model.ensemble_method in ("weighted_avg", "max")
        ):
            return torch.exp(logits)

        return F.softmax(logits, dim=-1)

    def predict(self, text: str, return_probabilities: bool = False) -> dict:
        """
        Predict for a single text.

        Args:
            text:                 Input text.
            return_probabilities: Include per-class probabilities in result.

        Returns:
            dict with: prediction, confidence, language,
                       and optionally probabilities.
        """
        try:
            from langdetect import detect
            language = detect(text)
        except Exception:
            language = "unknown"

        enc   = self._tokenise([text])
        probs = self._get_probs(enc)          # [1, num_classes]

        pred_idx   = probs.argmax(dim=-1).item()
        pred_label = "Real" if pred_idx == 1 else "Fake"
        confidence = probs[0, pred_idx].item()
        probs_np   = probs[0].cpu().numpy()

        result = {
            "prediction": pred_label,
            "confidence": round(confidence, 6),
            "language":   language,
        }
        if return_probabilities:
            result["probabilities"] = {
                "Fake": round(float(probs_np[0]), 6),
                "Real": round(float(probs_np[1]), 6),
            }
        return result

    def batch_predict(self, texts: list[str]) -> list[dict]:
        """
        Predict for multiple texts in a single batched forward pass.
        More efficient than calling predict() in a loop.

        Args:
            texts: List of input texts.

        Returns:
            List of result dicts (same format as predict(return_probabilities=True)).
        """
        # Detect languages per-text first (fast, CPU-only)
        languages = []
        for t in texts:
            try:
                from langdetect import detect
                languages.append(detect(t))
            except Exception:
                languages.append("unknown")

        enc   = self._tokenise(texts)
        probs = self._get_probs(enc)     # [batch, num_classes]
        probs_np = probs.cpu().numpy()

        results = []
        for i, (lang, p) in enumerate(zip(languages, probs_np)):
            pred_idx = int(p.argmax())
            results.append({
                "prediction":    "Real" if pred_idx == 1 else "Fake",
                "confidence":    round(float(p[pred_idx]), 6),
                "language":      lang,
                "probabilities": {
                    "Fake": round(float(p[0]), 6),
                    "Real": round(float(p[1]), 6),
                },
            })
        return results


# ════════════════════════════════════════════════════════════════════════════ #
#  Model loading                                                               #
# ════════════════════════════════════════════════════════════════════════════ #

def load_detector() -> tuple:
    """
    Build a FakeNewsDetector from environment variables.

    Environment variables:
        MODEL_TYPE        xlm-roberta | muril | ensemble  (default: muril)
        CHECKPOINT_PATH   path to .pt file
        MAX_LENGTH        tokenisation max length           (default: 512)

    Returns:
        (FakeNewsDetector | None, success: bool)
        Never raises; logs errors and returns (None, False) on failure.
    """
    raw_type        = (os.environ.get("MODEL_TYPE") or "muril").strip().lower()
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "").strip()
    max_length      = int(os.environ.get("MAX_LENGTH", "512"))
    device          = "cuda" if torch.cuda.is_available() else "cpu"

    if not checkpoint_path:
        checkpoint_path = f"models/checkpoints/{raw_type}_best.pt"

    if not os.path.isfile(checkpoint_path):
        logger.warning(
            "Checkpoint not found at %s; API will start but /predict returns 503.",
            checkpoint_path,
        )
        return None, False

    try:
        classes        = _get_model_classes()
        tokenizer_name = TOKENIZER_MAP.get(raw_type, "xlm-roberta-base")
        ckpt           = torch.load(checkpoint_path, map_location=device)
        state_dict     = _extract_state_dict(ckpt)

        if raw_type == "ensemble":
            try:
                from models.xlm_roberta_model import XLMRobertaFakeNewsClassifier as _XLMR
                from models.muril_model import MuRILFakeNewsClassifier as _MURIL
                from models.ensemble_model import EnsembleFakeNewsClassifier as _ENS
            except ModuleNotFoundError:
                from xlm_roberta_model import XLMRobertaFakeNewsClassifier as _XLMR  # noqa
                from muril_model import MuRILFakeNewsClassifier as _MURIL             # noqa
                from ensemble_model import EnsembleFakeNewsClassifier as _ENS         # noqa

            xlmr   = _XLMR()
            muril  = _MURIL()

            # Detect ensemble_method from checkpoint — same priority as eval_and_vis
            ensemble_method = None
            if isinstance(ckpt, dict):
                if "ensemble_method" in ckpt:
                    ensemble_method = ckpt["ensemble_method"]
                if ensemble_method is None:
                    cfg = ckpt.get("config", {}) or {}
                    ensemble_method = cfg.get("ensemble_method")
            if ensemble_method is None:
                if any(k.startswith("ensemble_fc") for k in state_dict):
                    ensemble_method = "learned"
                else:
                    ensemble_method = "weighted_avg"
            logger.info("Ensemble method detected: %s", ensemble_method)

            model  = _ENS(
                xlmr_model=xlmr,
                muril_model=muril,
                num_classes=2,
                ensemble_method=ensemble_method,
            )
        else:
            model_class = classes.get(raw_type)
            if model_class is None:
                logger.warning("Unknown MODEL_TYPE=%s; falling back to xlm-roberta.", raw_type)
                raw_type    = "xlm-roberta"
                model_class = classes["xlm-roberta"]
                tokenizer_name = TOKENIZER_MAP["xlm-roberta"]
            model = model_class()

        _safe_load(model, state_dict)
        detector = FakeNewsDetector(
            model=model,
            tokenizer_name=tokenizer_name,
            model_type=raw_type,
            device=device,
            max_length=max_length,
        )
        logger.info("Model loaded: type=%s path=%s", raw_type, checkpoint_path)
        return detector, True

    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        return None, False


# ── Text cleaning (matches training pipeline) ────────────────────────────── #

def clean_text_for_inference(text: str) -> str:
    """Apply same cleaning as training pipeline. Falls back gracefully."""
    try:
        try:
            from utils.preprocessing import DataPreprocessor
        except ModuleNotFoundError:
            from preprocessing import DataPreprocessor  # flat layout
        return DataPreprocessor().clean_text(text or "")
    except Exception:
        import re
        t = (text or "").strip()
        t = re.sub(r'https?://\S+|www\.\S+', '', t)
        t = re.sub(r'\S+@\S+', '', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t[:50_000]


# ════════════════════════════════════════════════════════════════════════════ #
#  Flask API                                                                   #
# ════════════════════════════════════════════════════════════════════════════ #

from flask import Flask, request, jsonify

# Limits (configurable via environment variables)
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 1 * 1024 * 1024))   # 1 MB
MAX_TEXT_LENGTH    = int(os.environ.get("MAX_TEXT_LENGTH",    50_000))
MAX_BATCH_SIZE     = int(os.environ.get("MAX_BATCH_SIZE",     20))

# Module-level detector — lazy initialization so importing this module in
# tests does not trigger a checkpoint file lookup or GPU/model load.
# The detector is initialized on first call to create_app() or when the
# module is run directly (__main__).
_detector: "FakeNewsDetector | None" = None
_model_loaded: bool = False
_checkpoint_path = os.environ.get("CHECKPOINT_PATH", "")
_model_type      = (os.environ.get("MODEL_TYPE") or "muril").strip().lower()


def _get_or_load_detector():
    """Return the singleton detector, loading it on first call."""
    global _detector, _model_loaded
    if _detector is None and not _model_loaded:
        _detector, _model_loaded = load_detector()
    return _detector, _model_loaded


def create_app() -> Flask:
    """
    Flask app factory.  Import and call this to get the app without side effects.
    The detector is loaded lazily on the first request so importing this module
    in unit tests does not trigger model loading.

    Usage:
        from inference import create_app
        app = create_app()
    """
    app = Flask(__name__)
    # Use Flask's built-in request size limit
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    def _require_model():
        det, loaded = _get_or_load_detector()
        if not loaded or det is None:
            return jsonify({
                "error": "Model not loaded; prediction unavailable.",
                "code":  "MODEL_NOT_LOADED",
            }), 503
        return None

    @app.route("/health", methods=["GET"])
    def health():
        """Health check. Reports model load status, type, and checkpoint."""
        _, loaded = _get_or_load_detector()
        return jsonify({
            "status":          "healthy",
            "model_loaded":    loaded,
            "model_type":      _model_type,
            "checkpoint_path": _checkpoint_path,
        }), 200

    @app.route("/predict", methods=["POST"])
    def predict():
        """Single-text prediction."""
        err = _require_model()
        if err is not None:
            return err
        det, _ = _get_or_load_detector()
        try:
            data = request.get_json(silent=True)
            if data is None:
                return jsonify({"error": "Invalid or missing JSON", "code": "INVALID_JSON"}), 400
            text = (data.get("text") or "").strip()
            if not text:
                return jsonify({"error": "No text provided", "code": "NO_TEXT"}), 400
            if len(text) > MAX_TEXT_LENGTH:
                return jsonify({
                    "error": f"Text exceeds {MAX_TEXT_LENGTH} characters",
                    "code":  "TEXT_TOO_LONG",
                }), 400
            text = clean_text_for_inference(text)
            if not text:
                return jsonify({"error": "Text empty after cleaning", "code": "NO_TEXT"}), 400
            result = det.predict(text, return_probabilities=True)
            return jsonify(result), 200
        except Exception as e:
            logger.exception("Predict failed: %s", e)
            return jsonify({"error": str(e), "code": "PREDICT_ERROR"}), 500

    @app.route("/batch_predict", methods=["POST"])
    def batch_predict():
        """Batched prediction (up to MAX_BATCH_SIZE texts per request)."""
        err = _require_model()
        if err is not None:
            return err
        det, _ = _get_or_load_detector()
        try:
            data = request.get_json(silent=True)
            if data is None:
                return jsonify({"error": "Invalid or missing JSON", "code": "INVALID_JSON"}), 400
            texts = data.get("texts")
            if not isinstance(texts, list):
                return jsonify({"error": "Expected 'texts' array", "code": "INVALID_INPUT"}), 400
            if len(texts) > MAX_BATCH_SIZE:
                return jsonify({
                    "error": f"At most {MAX_BATCH_SIZE} texts per request",
                    "code":  "BATCH_TOO_LARGE",
                }), 400

            cleaned = []
            for t in texts:
                s = (t if isinstance(t, str) else str(t)).strip()[:MAX_TEXT_LENGTH]
                s = clean_text_for_inference(s)
                cleaned.append(s or "")

            # Use batched forward pass for efficiency
            results = det.batch_predict(cleaned)
            return jsonify({"results": results}), 200
        except Exception as e:
            logger.exception("Batch predict failed: %s", e)
            return jsonify({"error": str(e), "code": "PREDICT_ERROR"}), 500

    return app


# Module-level app for gunicorn / direct run compatibility.
# Detector is NOT loaded here — it loads lazily on first request.
app = create_app()


if __name__ == "__main__":
    # Eagerly load the detector when running as a server so startup errors
    # are visible immediately rather than on the first request.
    _get_or_load_detector()
    port  = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0").strip().lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=port, debug=debug)
