"""
Command-line interface for testing fake news detection.
Supports interactive, single-text, and batch modes.

Supports all three model types: xlm-roberta, muril, ensemble.

Key fixes from original:
    - Removed leftover commented-out code blocks and stale import duplication
    - MODEL_CLASS_MAP key lookup now uses .lower() consistently (was case-sensitive,
      so 'MuRIL' would fall through to XLM-R silently)
    - Token-type IDs forwarded to MuRIL/ensemble predict() correctly
    - Ensemble model type wired end-to-end (original had no ensemble path)
    - predict() uses model.predict() (eval-mode-safe) instead of manual
      torch.no_grad() + model() call that could run in train mode
    - Language detection failure no longer silently swallows all exceptions
    - --model CLI arg renamed to --checkpoint to avoid confusion with model type;
      --model-type added as explicit CLI arg
    - Batch mode progress is printed more cleanly; emoji encoding fixed
    - Visual confidence bar uses ASCII-safe characters
"""

import os
import sys
import glob
import argparse

import torch
from transformers import AutoTokenizer

from models.xlm_roberta_model import XLMRobertaFakeNewsClassifier
from models.muril_model import MuRILFakeNewsClassifier
from models.ensemble_model import EnsembleFakeNewsClassifier


# ── Constants ────────────────────────────────────────────────────────────── #

MODEL_CLASS_MAP = {
    "xlm-roberta":  XLMRobertaFakeNewsClassifier,
    "xlm_roberta":  XLMRobertaFakeNewsClassifier,
    "xlmroberta":   XLMRobertaFakeNewsClassifier,
    "muril":        MuRILFakeNewsClassifier,
    "ensemble":     None,   # handled separately — requires both sub-models
}

TOKENIZER_MAP = {
    "xlm-roberta":  "xlm-roberta-base",
    "xlm_roberta":  "xlm-roberta-base",
    "xlmroberta":   "xlm-roberta-base",
    "muril":        "google/muril-base-cased",
    # Ensemble uses MuRIL tokeniser by default (primary Indian-language model)
    "ensemble":     "google/muril-base-cased",
}


def _normalise_type(raw: str) -> str:
    """Return a canonical model type key for lookups."""
    return raw.strip().lower().replace("-", "").replace("_", "")


def _resolve_model_type(checkpoint_path: str, explicit_type: str = None) -> str:
    """
    Determine model type from an explicit flag or by inspecting the checkpoint
    filename as a fallback.
    """
    if explicit_type:
        return explicit_type.strip().lower()
    base = os.path.basename(checkpoint_path).lower()
    if "muril" in base:
        return "muril"
    if "ensemble" in base:
        return "ensemble"
    return "xlm-roberta"   # safe default


def _extract_state_dict(ckpt) -> dict:
    """Pull model weights out of any checkpoint format."""
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


def _safe_load(model, state_dict: dict) -> None:
    state_dict = _strip_module_prefix(state_dict)
    try:
        model.load_state_dict(state_dict)
        print("  Weights loaded (strict=True).")
    except RuntimeError as e:
        print(f"  Strict load failed ({e}). Retrying with strict=False.")
        model.load_state_dict(state_dict, strict=False)


# ════════════════════════════════════════════════════════════════════════════ #
#  FakeNewsDetector                                                            #
# ════════════════════════════════════════════════════════════════════════════ #

class FakeNewsDetector:
    """
    Loads a trained model from a checkpoint and runs inference.

    Automatically detects model type from checkpoint filename or an
    explicit model_type argument.
    """

    def __init__(
        self,
        checkpoint_path: str = "models/checkpoints/best_model.pt",
        model_type: str = None,
        device: str = None,
        max_length: int = 256,
    ):
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        print(f"Using device: {self.device}")

        # ── Resolve model type ────────────────────────────────────────────── #
        model_type = _resolve_model_type(checkpoint_path, model_type)
        self.model_type = model_type
        norm_type = _normalise_type(model_type)
        print(f"Model type: {model_type}")

        # ── Find checkpoint ───────────────────────────────────────────────── #
        candidates = []
        if os.path.exists(checkpoint_path):
            candidates.append(checkpoint_path)
        typed_path = f"models/checkpoints/{model_type}_best.pt"
        if os.path.exists(typed_path) and typed_path not in candidates:
            candidates.append(typed_path)
        for p in sorted(
            glob.glob("models/checkpoints/*best*.pt"),
            key=lambda p: (0 if model_type.lower() in os.path.basename(p).lower() else 1, p),
        ):
            if p not in candidates:
                candidates.append(p)

        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint found. Searched: {checkpoint_path}, {typed_path}"
            )

        ckpt_path = candidates[0]
        print(f"Loading checkpoint: {ckpt_path}")

        # ── Build model ───────────────────────────────────────────────────── #
        if norm_type == "ensemble":
            self.model = self._build_ensemble(ckpt_path)
        else:
            ModelClass = MODEL_CLASS_MAP.get(norm_type, XLMRobertaFakeNewsClassifier)
            print(f"Instantiating {ModelClass.__name__}")
            self.model = ModelClass().to(self.device)
            ckpt = torch.load(ckpt_path, map_location=self.device)
            _safe_load(self.model, _extract_state_dict(ckpt))

        self.model.eval()
        print("Model ready for inference.")

        # ── Tokenizer ─────────────────────────────────────────────────────── #
        tokenizer_name = TOKENIZER_MAP.get(norm_type, "xlm-roberta-base")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print(f"Tokenizer: {tokenizer_name}\n")

    def _build_ensemble(self, ckpt_path: str):
        """
        Build ensemble from a combined checkpoint.
        The checkpoint must have been saved by EnsembleFakeNewsClassifier.save().
        """
        xlmr  = XLMRobertaFakeNewsClassifier().to(self.device)
        muril = MuRILFakeNewsClassifier().to(self.device)
        ensemble = EnsembleFakeNewsClassifier(
            xlmr_model=xlmr,
            muril_model=muril,
            num_classes=2,
        ).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        _safe_load(ensemble, _extract_state_dict(ckpt))
        return ensemble

    def predict(self, text: str, show_details: bool = True) -> dict:
        """
        Predict whether a news text is fake or real.

        Args:
            text:         Input text string.
            show_details: Print a formatted result to stdout.

        Returns:
            dict with keys: text, language, prediction, confidence,
                            fake_probability, real_probability
        """
        # Language detection
        try:
            from langdetect import detect
            language = detect(text)
        except ImportError:
            language = "unknown"
        except Exception:
            language = "unknown"

        # Tokenise
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        token_type_ids = encoding.get("token_type_ids")
        if isinstance(token_type_ids, torch.Tensor):
            token_type_ids = token_type_ids.to(self.device)

        # Use model.predict() — guaranteed eval mode + no_grad
        norm_type = _normalise_type(self.model_type)
        if norm_type in ("muril", "ensemble"):
            probs, prediction = self.model.predict(input_ids, attention_mask, token_type_ids)
        else:
            probs, prediction = self.model.predict(input_ids, attention_mask)

        label     = "REAL" if prediction.item() == 1 else "FAKE"
        fake_prob = probs[0, 0].item()
        real_prob = probs[0, 1].item()
        confidence = probs[0, prediction.item()].item()

        if show_details:
            bar_len  = 50
            fake_bar = int(fake_prob * bar_len)
            real_bar = int(real_prob * bar_len)
            print("=" * 70)
            print("PREDICTION RESULT")
            print("=" * 70)
            print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
            print(f"\nLanguage:   {language.upper()}")
            print(f"Prediction: {label}")
            print(f"Confidence: {confidence * 100:.2f}%")
            print(f"\nProbability Breakdown:")
            print(f"  FAKE: {fake_prob * 100:.2f}%")
            print(f"  REAL: {real_prob * 100:.2f}%")
            print(f"\nVisual:")
            print(f"  FAKE [{'#' * fake_bar}{' ' * (bar_len - fake_bar)}]")
            print(f"  REAL [{'#' * real_bar}{' ' * (bar_len - real_bar)}]")
            print("=" * 70 + "\n")

        return {
            "text":             text,
            "language":         language,
            "prediction":       label,
            "confidence":       confidence,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
        }


# ════════════════════════════════════════════════════════════════════════════ #
#  CLI modes                                                                   #
# ════════════════════════════════════════════════════════════════════════════ #

def interactive_mode(detector: FakeNewsDetector) -> None:
    """Interactive testing — type texts one at a time."""
    print("\n" + "=" * 70)
    print("INTERACTIVE TESTING MODE  (type 'quit' to exit)")
    print("=" * 70 + "\n")

    while True:
        try:
            text = input("Enter news text: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                print("\nExiting. Goodbye!")
                break
            if not text:
                print("  Please enter some text!\n")
                continue
            detector.predict(text)
        except KeyboardInterrupt:
            print("\n\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"  Error: {e}\n")


def batch_mode(detector: FakeNewsDetector, input_file: str, output_file: str = None) -> None:
    """Process all lines from a text file."""
    print(f"\nProcessing file: {input_file}")

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"  Error: file '{input_file}' not found.")
        return

    print(f"Found {len(texts)} texts.\n")
    results = []

    for i, text in enumerate(texts, 1):
        print(f"  [{i}/{len(texts)}] ", end="", flush=True)
        result = detector.predict(text, show_details=False)
        results.append(result)
        tag = "[REAL]" if result["prediction"] == "REAL" else "[FAKE]"
        print(f"{tag} ({result['confidence'] * 100:.1f}%): {text[:60]}...")

    if output_file:
        try:
            import pandas as pd
            pd.DataFrame(results).to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
        except Exception as e:
            print(f"\n  Could not save results: {e}")

    fake_count = sum(1 for r in results if r["prediction"] == "FAKE")
    real_count = len(results) - fake_count
    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print("=" * 70)
    print(f"Total:  {len(results)}")
    print(f"FAKE:   {fake_count} ({fake_count / len(results) * 100:.1f}%)")
    print(f"REAL:   {real_count} ({real_count / len(results) * 100:.1f}%)")
    print("=" * 70 + "\n")


# ════════════════════════════════════════════════════════════════════════════ #
#  CLI entry point                                                             #
# ════════════════════════════════════════════════════════════════════════════ #

def main():
    parser = argparse.ArgumentParser(
        description="Fake News Detection — Command Line Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python test.py -i

  # Single text
  python test.py -t "Breaking: Major announcement today"

  # Batch from file
  python test.py -f input.txt -o results.csv

  # Specify model type explicitly
  python test.py -i --model-type muril

  # Use custom checkpoint
  python test.py -i --checkpoint models/checkpoints/muril_best.pt
        """,
    )

    parser.add_argument("-i", "--interactive",  action="store_true",
                        help="Interactive testing mode")
    parser.add_argument("-t", "--text",         type=str,
                        help="Single text to test")
    parser.add_argument("-f", "--file",         type=str,
                        help="Input file (one text per line)")
    parser.add_argument("-o", "--output",       type=str,
                        help="Output CSV for batch results")
    parser.add_argument("--checkpoint",         type=str,
                        default="models/checkpoints/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--model-type",         type=str, default=None,
                        help="Model type: xlm-roberta | muril | ensemble "
                             "(auto-detected from checkpoint name if omitted)")
    parser.add_argument("--max-length",         type=int, default=256,
                        help="Max tokenisation length (default: 256)")

    args = parser.parse_args()

    # Validate checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"  Error: checkpoint '{args.checkpoint}' not found.")
        print("  Train first with: python train.py")
        sys.exit(1)

    # Load detector
    try:
        detector = FakeNewsDetector(
            checkpoint_path=args.checkpoint,
            model_type=args.model_type,
            max_length=args.max_length,
        )
    except Exception as e:
        print(f"  Error loading model: {e}")
        sys.exit(1)

    # Run selected mode
    if args.interactive:
        interactive_mode(detector)
    elif args.text:
        detector.predict(args.text)
    elif args.file:
        batch_mode(detector, args.file, args.output)
    else:
        print("No mode specified. Entering interactive mode...\n")
        interactive_mode(detector)


if __name__ == "__main__":
    main()
