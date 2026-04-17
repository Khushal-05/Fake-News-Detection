"""
Evaluation script for fake news detection models.

Supports all three model types: xlm-roberta, muril, ensemble.

Key fixes from original:
    - token_type_ids forwarded to MuRIL and ensemble (was silently dropped)
    - Ensemble log-prob output converted correctly with exp() not softmax()
    - Model loading is robust: handles all checkpoint formats, strips
      'module.' prefix (DDP), falls back to strict=False
    - CONFIG wired for all three model types (original only had xlm-roberta/muril)
    - evaluate() uses _forward() helper to respect each model's signature
    - compute_metrics() zero_division=0 on precision_recall to avoid warnings
      on languages with very few samples
    - Visualisation: per-language plot sorted by F1 (more meaningful than
      insertion order); both plots saved to a timestamped output dir
    - print_report() handles missing ROC-AUC (e.g. single-class batches)
      without crashing
    - __main__ checkpoint loading uses same _safe_load / _extract_state_dict
      helpers as train.py for consistency
"""

import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from data.dataset import MultilingualFakeNewsDataset
from models.xlm_roberta_model import XLMRobertaFakeNewsClassifier
from models.muril_model import MuRILFakeNewsClassifier
from models.ensemble_model import EnsembleFakeNewsClassifier


# ── Checkpoint helpers (shared with train.py) ────────────────────────────── #

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


def _safe_load(model, state_dict: dict) -> None:
    state_dict = _strip_module_prefix(state_dict)
    try:
        model.load_state_dict(state_dict)
        print("  Weights loaded (strict=True).")
    except RuntimeError as e:
        print(f"  Strict load failed ({e}). Retrying with strict=False.")
        model.load_state_dict(state_dict, strict=False)


def _normalise_model_type(raw: str) -> str:
    key = raw.strip().lower().replace("-", "").replace("_", "")
    if key == "xlmroberta":
        return "xlm-roberta"
    if key == "muril":
        return "muril"
    if key == "ensemble":
        return "ensemble"
    raise ValueError(f"Unknown model_type='{raw}'. Choose: xlm-roberta | muril | ensemble")


# ════════════════════════════════════════════════════════════════════════════ #
#  FakeNewsEvaluator                                                           #
# ════════════════════════════════════════════════════════════════════════════ #

class FakeNewsEvaluator:
    """
    Evaluator for XLM-RoBERTa, MuRIL, and Ensemble fake news classifiers.

    Args:
        model:        A loaded nn.Module.
        test_dataset: Dataset returning dicts with 'input_ids',
                      'attention_mask', optionally 'token_type_ids',
                      'label', 'language'.
        model_type:   Canonical model type string (used to route inputs).
        device:       'cuda' or 'cpu'.
        batch_size:   Evaluation batch size.
    """

    def __init__(
        self,
        model,
        test_dataset,
        model_type: str = "muril",
        device: str = "cuda",
        batch_size: int = 16,
    ):
        self.model      = model.to(device)
        self.model_type = _normalise_model_type(model_type)
        self.device     = device

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=(device == "cuda" and torch.cuda.is_available()),
        )

    def _forward(self, batch: dict) -> torch.Tensor:
        """Run a single forward pass, routing inputs correctly per model type.
        
        Handles three batch formats:
        - XLM-RoBERTa: input_ids + attention_mask
        - MuRIL: input_ids + attention_mask + token_type_ids (optional)
        - Ensemble: xlmr_ids/mask + muril_ids/mask + muril_tti (from dataset dual tokenization)
        
        Returns probability tensor [batch, num_classes].
        """
        with torch.no_grad():
            if self.model_type == "ensemble":
                # Ensemble: extract keys from dual tokenization batch format
                xlmr_ids = batch["xlmr_ids"].to(self.device)
                xlmr_mask = batch["xlmr_mask"].to(self.device)
                muril_ids = batch["muril_ids"].to(self.device)
                muril_mask = batch["muril_mask"].to(self.device)
                muril_tti = batch.get("muril_tti")
                if muril_tti is not None:
                    muril_tti = muril_tti.to(self.device)
                
                logits = self.model(xlmr_ids, xlmr_mask, muril_ids, muril_mask, muril_tti)
            
            elif self.model_type == "xlm-roberta":
                # XLM-RoBERTa: no token_type_ids
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                logits = self.model(input_ids, attention_mask)
            
            else:  # muril
                # MuRIL: optional token_type_ids
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                logits = self.model(input_ids, attention_mask, token_type_ids)

        # Convert log-probs to probs if ensemble uses weighted_avg/max
        if (
            isinstance(self.model, EnsembleFakeNewsClassifier)
            and self.model.ensemble_method in ("weighted_avg", "max")
        ):
            return torch.exp(logits)

        return F.softmax(logits, dim=-1)

    # ── evaluate ─────────────────────────────────────────────────────────── #

    def evaluate(self) -> tuple:
        """
        Run inference over the full test set.

        Returns:
            (labels, predictions, probabilities, languages)
            All numpy arrays; languages is a list of strings.
        """
        self.model.eval()
        all_predictions  = []
        all_labels       = []
        all_probabilities = []
        all_languages    = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                probs       = self._forward(batch)          # [B, C]
                predictions = probs.argmax(dim=-1)
                labels      = batch["label"]

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probabilities.extend(probs.cpu().numpy())

                lang_batch = batch.get("language")
                if lang_batch is not None:
                    all_languages.extend(
                        lang_batch.cpu().numpy().tolist()
                        if isinstance(lang_batch, torch.Tensor)
                        else list(lang_batch)
                    )

        return (
            np.array(all_labels),
            np.array(all_predictions),
            np.array(all_probabilities),
            all_languages,
        )

    # ── compute_metrics ──────────────────────────────────────────────────── #

    def compute_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> dict:
        """Compute overall classification metrics."""
        accuracy  = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted", zero_division=0
        )
        class_report = classification_report(
            labels, predictions,
            target_names=["Fake", "Real"],
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(labels, predictions)

        roc_auc = None
        try:
            if len(np.unique(labels)) > 1:       # needs both classes present
                roc_auc = roc_auc_score(labels, probabilities[:, 1])
        except Exception:
            pass

        return {
            "accuracy":               accuracy,
            "precision":              precision,
            "recall":                 recall,
            "f1":                     f1,
            "roc_auc":                roc_auc,
            "classification_report":  class_report,
            "confusion_matrix":       cm,
        }

    # ── per-language metrics ─────────────────────────────────────────────── #

    def compute_per_language_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        languages: list,
    ) -> dict:
        """Compute accuracy/precision/recall/F1 for each language."""
        df = pd.DataFrame({
            "label":      labels,
            "prediction": predictions,
            "language":   languages,
        })

        per_lang = {}
        for lang in df["language"].unique():
            ldf  = df[df["language"] == lang]
            ll   = ldf["label"].values
            lp   = ldf["prediction"].values
            acc  = accuracy_score(ll, lp)
            prec, rec, f1, _ = precision_recall_fscore_support(
                ll, lp, average="weighted", zero_division=0
            )
            per_lang[lang] = {
                "accuracy":  acc,
                "precision": prec,
                "recall":    rec,
                "f1_score":  f1,      # key must be 'f1_score' — visualisation.py reads this key
                "samples":   len(ldf),
            }
        return per_lang

    # ── visualisation ────────────────────────────────────────────────────── #

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        save_path: str = "confusion_matrix.png",
    ) -> None:
        """Plot and save a labelled confusion matrix heatmap."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fake", "Real"],
            yticklabels=["Fake", "Real"],
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix saved → {save_path}")

    def plot_per_language_performance(
        self,
        per_lang_metrics: dict,
        save_path: str = "per_language_performance.png",
    ) -> None:
        """Bar chart of accuracy and F1 per language, sorted by F1 descending."""
        # Sort by F1 (descending) for readability
        sorted_langs = sorted(
            per_lang_metrics.keys(),
            key=lambda l: per_lang_metrics[l]["f1_score"],
            reverse=True,
        )
        accuracies = [per_lang_metrics[l]["accuracy"] for l in sorted_langs]
        f1_scores  = [per_lang_metrics[l]["f1_score"] for l in sorted_langs]
        samples    = [per_lang_metrics[l]["samples"]   for l in sorted_langs]

        x     = np.arange(len(sorted_langs))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(10, len(sorted_langs) * 0.8), 6))
        bars_acc = ax.bar(x - width / 2, accuracies, width, label="Accuracy", alpha=0.8, color="steelblue")
        bars_f1  = ax.bar(x + width / 2, f1_scores,  width, label="F1 Score",  alpha=0.8, color="coral")

        # Annotate sample counts above each language tick
        for i, (lang, n) in enumerate(zip(sorted_langs, samples)):
            ax.text(i, -0.08, f"n={n}", ha="center", va="top",
                    fontsize=7, transform=ax.get_xaxis_transform())

        ax.set_xlabel("Language")
        ax.set_ylabel("Score")
        ax.set_title("Per-Language Performance (sorted by F1)")
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_langs, rotation=45, ha="right")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Per-language performance plot saved → {save_path}")

    # ── print_report ─────────────────────────────────────────────────────── #

    def print_report(self, metrics: dict, per_lang_metrics: dict) -> None:
        """Print a formatted evaluation report to stdout."""
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)

        print("\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        if metrics.get("roc_auc") is not None:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        cr = metrics.get("classification_report", {})
        for cls in ("Fake", "Real"):
            if cls in cr:
                print(f"\n  {cls} News:")
                print(f"    Precision: {cr[cls]['precision']:.4f}")
                print(f"    Recall:    {cr[cls]['recall']:.4f}")
                print(f"    F1 Score:  {cr[cls]['f1-score']:.4f}")
                print(f"    Support:   {int(cr[cls]['support'])}")

        print("\nPer-Language Performance (sorted by F1):")
        for lang, lm in sorted(
            per_lang_metrics.items(),
            key=lambda x: x[1]["f1_score"],
            reverse=True,
        ):
            print(
                f"  {lang.upper():>8}  ({lm['samples']:>5} samples) "
                f"acc={lm['accuracy']:.4f}  f1={lm['f1_score']:.4f}"
            )

        print("\n" + "=" * 60)


# ════════════════════════════════════════════════════════════════════════════ #
#  __main__ entry point                                                        #
# ════════════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":

    CONFIG = {
        "model_type":        "muril",          # 'xlm-roberta' | 'muril' | 'ensemble'
        "model_checkpoint":  "models/checkpoints/muril_best.pt",
        # For ensemble, also set:
        # "xlmr_checkpoint": "models/checkpoints/xlm-roberta_best.pt",
        # "muril_checkpoint": "models/checkpoints/muril_best.pt",
        # "ensemble_method": "weighted_avg",
        "batch_size":        16,
        "max_length":        512,
        "device":            "cuda" if torch.cuda.is_available() else "cpu",
        "output_dir":        "outputs/evaluation",
    }

    model_type = _normalise_model_type(CONFIG["model_type"])
    device     = CONFIG["device"]
    print(f"Device: {device}  |  Model type: {model_type}")

    # ── Tokenizer ─────────────────────────────────────────────────────────── #
    TOKENIZER_MAP = {
        "xlm-roberta": "xlm-roberta-base",
        "muril":       "google/muril-base-cased",
        "ensemble":    "ensemble",
    }
    tokenizer_name = TOKENIZER_MAP[model_type]

    # ── Test data ─────────────────────────────────────────────────────────── #
    print("Loading test data...")
    test_df = pd.read_csv("data/processed/test.csv")

    test_dataset = MultilingualFakeNewsDataset(
        texts=test_df["cleaned_text"].values,
        labels=test_df["label"].values,
        languages=test_df["language"].values,
        tokenizer_name=tokenizer_name,
        max_length=CONFIG["max_length"],
    )

    # ── Build + load model ────────────────────────────────────────────────── #
    print(f"Loading {model_type} model from {CONFIG['model_checkpoint']}...")

    common_kwargs = dict(num_classes=2)

    if model_type == "xlm-roberta":
        model = XLMRobertaFakeNewsClassifier(**common_kwargs)
        _safe_load(model, _extract_state_dict(
            torch.load(CONFIG["model_checkpoint"], map_location=device)
        ))

    elif model_type == "muril":
        model = MuRILFakeNewsClassifier(**common_kwargs)
        _safe_load(model, _extract_state_dict(
            torch.load(CONFIG["model_checkpoint"], map_location=device)
        ))

    else:  # ensemble
        xlmr  = XLMRobertaFakeNewsClassifier(**common_kwargs)
        muril = MuRILFakeNewsClassifier(**common_kwargs)

        # Optionally load individual sub-model checkpoints
        for sub, key in [(xlmr, "xlmr_checkpoint"), (muril, "muril_checkpoint")]:
            path = CONFIG.get(key, "")
            if path and os.path.isfile(path):
                _safe_load(sub, _extract_state_dict(torch.load(path, map_location=device)))

        model = EnsembleFakeNewsClassifier(
            xlmr_model=xlmr,
            muril_model=muril,
            num_classes=2,
            ensemble_method=CONFIG.get("ensemble_method", "weighted_avg"),
        )
        # Load ensemble-level checkpoint if present (overrides sub-model weights)
        if os.path.isfile(CONFIG["model_checkpoint"]):
            _safe_load(model, _extract_state_dict(
                torch.load(CONFIG["model_checkpoint"], map_location=device)
            ))

    model.to(device)

    # ── Evaluator ─────────────────────────────────────────────────────────── #
    evaluator = FakeNewsEvaluator(
        model=model,
        test_dataset=test_dataset,
        model_type=model_type,
        device=device,
        batch_size=CONFIG["batch_size"],
    )

    # ── Run evaluation ────────────────────────────────────────────────────── #
    print("Evaluating...")
    labels, predictions, probabilities, languages = evaluator.evaluate()

    metrics          = evaluator.compute_metrics(labels, predictions, probabilities)
    per_lang_metrics = evaluator.compute_per_language_metrics(labels, predictions, languages)

    evaluator.print_report(metrics, per_lang_metrics)

    # ── Save outputs ──────────────────────────────────────────────────────── #
    timestamp  = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(CONFIG["output_dir"], f"{model_type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    evaluator.plot_confusion_matrix(
        metrics["confusion_matrix"],
        save_path=os.path.join(output_dir, "confusion_matrix.png"),
    )
    evaluator.plot_per_language_performance(
        per_lang_metrics,
        save_path=os.path.join(output_dir, "per_language_performance.png"),
    )

    # Save metrics JSON
    import json
    metrics_to_save = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in metrics.items()
        if k != "classification_report"
    }
    metrics_to_save["classification_report"] = metrics["classification_report"]
    with open(os.path.join(output_dir, "metrics.json"), "w") as fh:
        json.dump(metrics_to_save, fh, indent=2)

    # Save per-language metrics JSON
    with open(os.path.join(output_dir, "per_language_metrics.json"), "w") as fh:
        json.dump(per_lang_metrics, fh, indent=2)

    print(f"\nEvaluation complete. Outputs saved to: {output_dir}")
