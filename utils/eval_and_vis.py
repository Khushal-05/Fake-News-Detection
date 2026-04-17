"""
Evaluate and Visualize Performance of Trained Models (.pt checkpoint)

Supports all three model types: xlm-roberta, muril, ensemble.
Loads a trained checkpoint, runs inference on test data, computes all
metrics, and generates every visualization from visualisation.py.

Usage:
    # Auto-detect model type from checkpoint filename:
    python -m utils.eval_and_vis --checkpoint models/checkpoints/muril_best.pt

    # Explicit model type:
    python -m utils.eval_and_vis --checkpoint models/checkpoints/ensemble_best.pt --model-type ensemble

    # Custom paths:
    python -m utils.eval_and_vis \
        --checkpoint models/checkpoints/xlm-roberta_best.pt \
        --test-csv data/processed/test.csv \
        --output-dir outputs/eval_xlmr \
        --batch-size 32
"""

import os
import sys
import json
import argparse
from datetime import datetime

# ── Ensure project root is on sys.path ───────────────────────────────────── #
# This file lives in  fnd/utils/eval_and_vis.py
# Project root is     fnd/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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
    roc_curve,
)
from tqdm import tqdm

# ── Project imports ──────────────────────────────────────────────────────── #
from data.dataset import MultilingualFakeNewsDataset
from models.xlm_roberta_model import XLMRobertaFakeNewsClassifier
from models.muril_model import MuRILFakeNewsClassifier
from models.ensemble_model import EnsembleFakeNewsClassifier
from utils.visualisation import ModelVisualizer


# ── Checkpoint helpers (same as train.py) ────────────────────────────────── #

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
        print(f"  Strict load failed ({e}). Retrying strict=False.")
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


def _detect_model_type(checkpoint_path: str) -> str:
    """Guess model type from checkpoint filename."""
    base = os.path.basename(checkpoint_path).lower()
    if "ensemble" in base:
        return "ensemble"
    if "muril" in base:
        return "muril"
    return "xlm-roberta"


# ════════════════════════════════════════════════════════════════════════════ #
#  ModelEvaluatorVisualizer                                                    #
# ════════════════════════════════════════════════════════════════════════════ #

class ModelEvaluatorVisualizer:
    """
    Complete evaluation and visualization pipeline for any trained model.

    Supports: xlm-roberta, muril, ensemble.
    Generates all plots from visualisation.py's ModelVisualizer plus extras.
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = None,
        output_dir: str = None,
        device: str = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Auto-detect or normalise model type
        if model_type:
            self.model_type = _normalise_model_type(model_type)
        else:
            self.model_type = _detect_model_type(checkpoint_path)
            print(f"  Auto-detected model type: {self.model_type}")

        # Output directory
        if output_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("outputs", "evaluation", f"{self.model_type}_{ts}")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Tokenizer name for dataset
        TOKENIZER_MAP = {
            "xlm-roberta": "xlm-roberta-base",
            "muril":       "google/muril-base-cased",
            "ensemble":    "ensemble",
        }
        self.tokenizer_name = TOKENIZER_MAP[self.model_type]

        # Visualizer from visualisation.py
        self.viz = ModelVisualizer(class_names=["Fake", "Real"])

        print("=" * 70)
        print(" MODEL EVALUATION & VISUALIZATION ".center(70, "="))
        print("=" * 70)
        print(f"  Device:     {self.device}")
        print(f"  Model type: {self.model_type}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Output dir: {output_dir}")
        print("=" * 70)

    # ── Model loading ────────────────────────────────────────────────────── #

    def load_model(self) -> None:
        """Load model from .pt checkpoint. Handles all 3 model types."""
        print("\n  Loading model from checkpoint...")

        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = _extract_state_dict(ckpt)

        if self.model_type == "xlm-roberta":
            self.model = XLMRobertaFakeNewsClassifier(num_classes=2)
            _safe_load(self.model, state_dict)

        elif self.model_type == "muril":
            self.model = MuRILFakeNewsClassifier(num_classes=2)
            _safe_load(self.model, state_dict)

        else:  # ensemble
            xlmr = XLMRobertaFakeNewsClassifier(num_classes=2)
            muril = MuRILFakeNewsClassifier(num_classes=2)

            # Determine ensemble_method from checkpoint metadata
            ensemble_method = "learned"
            if isinstance(ckpt, dict):
                cfg = ckpt.get("config", {})
                ensemble_method = cfg.get("ensemble_method", "learned")
                # Also try from model_type metadata
                if "ensemble_method" not in cfg:
                    # Check if ensemble_fc keys exist in state_dict
                    if any("ensemble_fc" in k for k in state_dict):
                        ensemble_method = "learned"

            self.model = EnsembleFakeNewsClassifier(
                xlmr_model=xlmr,
                muril_model=muril,
                num_classes=2,
                ensemble_method=ensemble_method,
            )
            _safe_load(self.model, state_dict)

        self.model.to(self.device)
        self.model.eval()

        # Print parameter info
        param_info = self.model.count_parameters()
        t_key = "total_trainable" if "total_trainable" in param_info else "trainable"
        f_key = "total_frozen" if "total_frozen" in param_info else "frozen"
        print(f"  Trainable: {param_info[t_key]:,} | Frozen: {param_info[f_key]:,} | Total: {param_info['total']:,}")

        # Print checkpoint metadata
        if isinstance(ckpt, dict):
            if "epoch" in ckpt:
                print(f"  Checkpoint epoch: {ckpt['epoch']}")
            if "best_val_f1" in ckpt:
                print(f"  Best val F1: {ckpt['best_val_f1']:.4f}")
            if "val_metrics" in ckpt and isinstance(ckpt["val_metrics"], dict):
                vm = ckpt["val_metrics"]
                if "accuracy" in vm:
                    print(f"  Val accuracy: {vm['accuracy']:.4f}")

        print("  Model loaded successfully!\n")

    # ── Data loading ─────────────────────────────────────────────────────── #

    def load_test_data(
        self,
        test_csv_path: str,
        max_length: int = 512,
        batch_size: int = 16,
    ) -> pd.DataFrame:
        """Load test dataset and create DataLoader."""
        print(f"  Loading test data from: {test_csv_path}")

        test_df = pd.read_csv(test_csv_path)
        print(f"  Loaded {len(test_df):,} test samples")

        if "language" in test_df.columns:
            print(f"  Languages: {sorted(test_df['language'].unique().tolist())}")
        if "label" in test_df.columns:
            print(f"  Labels: {test_df['label'].value_counts().to_dict()}")

        test_dataset = MultilingualFakeNewsDataset(
            texts=test_df["cleaned_text"].values,
            labels=test_df["label"].values,
            languages=test_df["language"].values if "language" in test_df.columns else ["unknown"] * len(test_df),
            tokenizer_name=self.tokenizer_name,
            max_length=max_length,
        )

        pin = self.device == "cuda" and torch.cuda.is_available()
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # safe for all platforms
            pin_memory=pin,
        )

        self.test_df = test_df
        return test_df

    # ── Forward pass routing ─────────────────────────────────────────────── #

    def _forward_batch(self, batch: dict) -> torch.Tensor:
        """Run model-type-aware forward pass. Returns logits."""
        if self.model_type == "ensemble":
            logits = self.model(
                batch["xlmr_ids"].to(self.device),
                batch["xlmr_mask"].to(self.device),
                batch["muril_ids"].to(self.device),
                batch["muril_mask"].to(self.device),
                batch["muril_tti"].to(self.device),
            )
        elif self.model_type == "xlm-roberta":
            logits = self.model(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
            )
        else:  # muril
            tti = batch.get("token_type_ids")
            if isinstance(tti, torch.Tensor):
                tti = tti.to(self.device)
            logits = self.model(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
                tti,
            )
        return logits

    def _logits_to_probs(self, logits: torch.Tensor) -> np.ndarray:
        """Convert logits to probability array, handling ensemble log-probs."""
        if (
            self.model_type == "ensemble"
            and isinstance(self.model, EnsembleFakeNewsClassifier)
            and self.model.ensemble_method in ("weighted_avg", "max")
        ):
            return torch.exp(logits).cpu().numpy()
        else:
            return F.softmax(logits, dim=-1).cpu().numpy()

    # ── Evaluation ───────────────────────────────────────────────────────── #

    def evaluate_model(self) -> tuple:
        """Run inference on the full test set.

        Returns:
            (y_true, y_pred, y_proba, languages)
        """
        print("  Evaluating model on test set...")
        self.model.eval()

        all_labels = []
        all_preds = []
        all_probs = []
        all_languages = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                labels = batch["label"]
                logits = self._forward_batch(batch)
                probs = self._logits_to_probs(logits)

                all_preds.extend(probs.argmax(axis=1).tolist())
                all_labels.extend(labels.numpy().tolist())
                all_probs.append(probs)

                lang_batch = batch.get("language")
                if lang_batch is not None:
                    all_languages.extend(
                        lang_batch.cpu().numpy().tolist()
                        if isinstance(lang_batch, torch.Tensor)
                        else list(lang_batch)
                    )

        y_true = np.array(all_labels, dtype=int)
        y_pred = np.array(all_preds, dtype=int)
        y_proba = np.vstack(all_probs) if all_probs else np.zeros((0, 2))

        print(f"  Evaluation complete — {len(y_true):,} samples\n")
        return y_true, y_pred, y_proba, all_languages

    # ── Metrics ──────────────────────────────────────────────────────────── #

    def compute_metrics(self, y_true, y_pred, y_proba) -> dict:
        """Compute overall classification metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)
        class_report = classification_report(
            y_true, y_pred, target_names=["Fake", "Real"],
            output_dict=True, zero_division=0,
        )

        roc_auc = None
        fpr, tpr = None, None
        try:
            if len(np.unique(y_true)) > 1 and y_proba.shape[1] >= 2:
                roc_auc = roc_auc_score(y_true, y_proba[:, 1])
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        except Exception:
            pass

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "classification_report": class_report,
            "fpr": fpr,
            "tpr": tpr,
        }

    def compute_per_language_metrics(self, y_true, y_pred, languages) -> dict:
        """Compute accuracy/precision/recall/F1 per language."""
        if not languages:
            return {}

        df = pd.DataFrame({
            "label": y_true, "prediction": y_pred, "language": languages,
        })

        per_lang = {}
        for lang in sorted(df["language"].unique()):
            ldf = df[df["language"] == lang]
            ll, lp = ldf["label"].values, ldf["prediction"].values
            acc = accuracy_score(ll, lp)
            prec, rec, f1, _ = precision_recall_fscore_support(
                ll, lp, average="weighted", zero_division=0
            )
            per_lang[lang] = {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
                "samples": len(ldf),
            }
        return per_lang

    # ── Print report ─────────────────────────────────────────────────────── #

    def print_results(self, metrics: dict, per_lang_metrics: dict) -> None:
        print("\n" + "=" * 70)
        print(" EVALUATION RESULTS ".center(70, "="))
        print("=" * 70)

        print(f"\n  Model type: {self.model_type}")
        print(f"\n  Overall Metrics:")
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1-Score:  {metrics['f1']:.4f}")
        if metrics.get("roc_auc") is not None:
            print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")

        print("\n  Per-Class Metrics:")
        for cls in ["Fake", "Real"]:
            cr = metrics["classification_report"].get(cls, {})
            if cr:
                print(f"    {cls}:")
                print(f"      Precision: {cr['precision']:.4f}")
                print(f"      Recall:    {cr['recall']:.4f}")
                print(f"      F1-Score:  {cr['f1-score']:.4f}")
                print(f"      Support:   {int(cr['support'])}")

        if per_lang_metrics:
            print("\n  Per-Language Performance:")
            for lang in sorted(per_lang_metrics, key=lambda l: per_lang_metrics[l]["f1_score"], reverse=True):
                lm = per_lang_metrics[lang]
                print(
                    f"    {lang.upper():>8}  ({lm['samples']:>5} samples) "
                    f"acc={lm['accuracy']:.4f}  f1={lm['f1_score']:.4f}"
                )

        print("\n" + "=" * 70)

    # ── Generate ALL visualizations ──────────────────────────────────────── #

    def generate_all_visualizations(
        self, y_true, y_pred, y_proba, languages, metrics, per_lang_metrics, history=None,
    ) -> None:
        """Generate every plot from visualisation.py's ModelVisualizer."""
        print("\n  Generating visualizations...\n")
        viz = self.viz
        od = self.output_dir

        # 1. Confusion Matrix (raw)
        try:
            viz.plot_confusion_matrix(
                y_true, y_pred, normalize=False,
                title=f"Confusion Matrix — {self.model_type}",
                save_path=os.path.join(od, "confusion_matrix_raw.png"),
            )
        except Exception as e:
            print(f"  [warn] Confusion matrix (raw) failed: {e}")

        # 2. Confusion Matrix (normalized)
        try:
            viz.plot_confusion_matrix(
                y_true, y_pred, normalize=True,
                title=f"Normalized Confusion Matrix — {self.model_type}",
                save_path=os.path.join(od, "confusion_matrix_normalized.png"),
            )
        except Exception as e:
            print(f"  [warn] Confusion matrix (norm) failed: {e}")

        # 3. ROC Curve
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                pos_probs = y_proba[:, 1]
            else:
                pos_probs = y_proba.ravel()
            viz.plot_roc_curve(
                y_true, pos_probs,
                title=f"ROC Curve — {self.model_type}",
                save_path=os.path.join(od, "roc_curve.png"),
            )
        except Exception as e:
            print(f"  [warn] ROC curve failed: {e}")

        # 4. Class Distribution (overall + per language)
        try:
            viz.plot_class_distribution(
                y_true, languages=languages if languages else None,
                title=f"Class Distribution — {self.model_type}",
                save_path=os.path.join(od, "class_distribution.png"),
            )
        except Exception as e:
            print(f"  [warn] Class distribution failed: {e}")

        # 5. Prediction Confidence Distribution
        try:
            viz.plot_prediction_confidence_distribution(
                y_true, y_proba,
                title=f"Prediction Confidence — {self.model_type}",
                save_path=os.path.join(od, "confidence_distribution.png"),
            )
        except Exception as e:
            print(f"  [warn] Confidence distribution failed: {e}")

        # 6. Per-Language Performance (bar chart)
        if per_lang_metrics:
            try:
                viz.plot_per_language_performance(
                    per_lang_metrics,
                    title=f"Per-Language Performance — {self.model_type}",
                    save_path=os.path.join(od, "per_language_performance.png"),
                )
            except Exception as e:
                print(f"  [warn] Per-language performance failed: {e}")

        # 7. Training History (if history.json exists in run dir or checkpoint)
        if history:
            try:
                viz.plot_training_history(
                    history,
                    metrics=["loss", "accuracy", "f1"],
                    title=f"Training History — {self.model_type}",
                    save_path=os.path.join(od, "training_history.png"),
                )
            except Exception as e:
                print(f"  [warn] Training history failed: {e}")

        # 8. Full Evaluation Dashboard (combined view)
        try:
            viz.create_evaluation_dashboard(
                y_true=y_true,
                y_pred=y_pred,
                y_proba=y_proba,
                languages=languages if languages else None,
                per_lang_metrics=per_lang_metrics if per_lang_metrics else None,
                save_path=os.path.join(od, "evaluation_dashboard.png"),
            )
        except Exception as e:
            print(f"  [warn] Evaluation dashboard failed: {e}")

        # 9. Extra: Metrics comparison bar chart
        try:
            self._plot_metrics_comparison(metrics)
        except Exception as e:
            print(f"  [warn] Metrics comparison failed: {e}")

        # 10. Extra: Detailed confidence analysis (4-panel)
        try:
            self._plot_detailed_confidence(y_true, y_proba)
        except Exception as e:
            print(f"  [warn] Detailed confidence failed: {e}")

        print(f"\n  All visualizations saved to: {od}\n")

    # ── Extra plots not in visualisation.py ───────────────────────────────── #

    def _plot_metrics_comparison(self, metrics: dict) -> None:
        """Bar chart of overall accuracy/precision/recall/F1/AUC."""
        import matplotlib.pyplot as plt

        metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
        values = [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]]
        colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]

        if metrics.get("roc_auc") is not None:
            metric_names.append("ROC-AUC")
            values.append(metrics["roc_auc"])
            colors.append("#9b59b6")

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(metric_names, values, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.4f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.set_title(f"Overall Performance — {self.model_type}", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "metrics_comparison.png"), dpi=300, bbox_inches="tight")
        print(f"  Metrics comparison saved.")
        plt.close()

    def _plot_detailed_confidence(self, y_true, y_proba) -> None:
        """4-panel confidence analysis."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        probs = y_proba[:, 1] if y_proba.ndim == 2 and y_proba.shape[1] >= 2 else y_proba.ravel()
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Overall probability distribution
        axes[0, 0].hist(probs, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
        axes[0, 0].axvline(x=0.5, color="red", linestyle="--", linewidth=2, label="Threshold")
        axes[0, 0].set_xlabel("P(Real)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Overall Probability Distribution", fontweight="bold")
        axes[0, 0].legend()
        axes[0, 0].grid(axis="y", alpha=0.3)

        # 2. By true class
        fake_probs = probs[y_true == 0]
        real_probs = probs[y_true == 1]
        axes[0, 1].hist(fake_probs, bins=30, alpha=0.7, label="True: Fake", color="coral", edgecolor="black")
        axes[0, 1].hist(real_probs, bins=30, alpha=0.7, label="True: Real", color="lightgreen", edgecolor="black")
        axes[0, 1].set_xlabel("P(Real)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Probability by True Class", fontweight="bold")
        axes[0, 1].legend()
        axes[0, 1].grid(axis="y", alpha=0.3)

        # 3. Confidence (max prob)
        confidence = np.maximum(probs, 1 - probs)
        axes[1, 0].hist(confidence, bins=50, alpha=0.7, color="mediumpurple", edgecolor="black")
        axes[1, 0].set_xlabel("Confidence")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Model Confidence Distribution", fontweight="bold")
        axes[1, 0].grid(axis="y", alpha=0.3)

        # 4. Box plot by class
        if len(fake_probs) > 0 and len(real_probs) > 0:
            bp = axes[1, 1].boxplot(
                [fake_probs, real_probs],
                labels=["True: Fake", "True: Real"],
                patch_artist=True, notch=True, showmeans=True,
            )
            colors_box = ["lightcoral", "lightgreen"]
            for patch, c in zip(bp["boxes"], colors_box):
                patch.set_facecolor(c)
        axes[1, 1].set_ylabel("P(Real)")
        axes[1, 1].set_title("Probability Box Plot", fontweight="bold")
        axes[1, 1].grid(axis="y", alpha=0.3)

        fig.suptitle(f"Detailed Confidence Analysis — {self.model_type}",
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "detailed_confidence.png"), dpi=300, bbox_inches="tight")
        print(f"  Detailed confidence analysis saved.")
        plt.close()

    # ── Save artifacts ───────────────────────────────────────────────────── #

    def save_artifacts(self, y_true, y_pred, y_proba, languages, metrics, per_lang_metrics) -> None:
        """Save predictions CSV, metrics JSON, and per-language JSON."""
        od = self.output_dir

        # Predictions CSV
        pred_df = pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "prob_fake": y_proba[:, 0] if y_proba.shape[1] >= 2 else 1 - y_proba.ravel(),
            "prob_real": y_proba[:, 1] if y_proba.shape[1] >= 2 else y_proba.ravel(),
        })
        if languages:
            pred_df["language"] = languages
        pred_path = os.path.join(od, "predictions.csv")
        pred_df.to_csv(pred_path, index=False)
        print(f"  Predictions saved to: {pred_path}")

        # Predictions NPZ
        np.savez_compressed(
            os.path.join(od, "predictions.npz"),
            y_true=np.array(y_true),
            y_pred=np.array(y_pred),
            y_proba=y_proba,
            languages=np.array(languages if languages else [], dtype=object),
        )

        # Metrics JSON
        metrics_save = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in metrics.items()
            if k != "classification_report" and v is not None
        }
        metrics_save["classification_report"] = metrics.get("classification_report", {})
        with open(os.path.join(od, "metrics.json"), "w") as f:
            json.dump(metrics_save, f, indent=2, default=str)

        # Per-language JSON
        if per_lang_metrics:
            with open(os.path.join(od, "per_language_metrics.json"), "w") as f:
                json.dump(per_lang_metrics, f, indent=2)

        print(f"  All artifacts saved to: {od}")

    # ── Full pipeline ────────────────────────────────────────────────────── #

    # ── Load pre-saved predictions ────────────────────────────────────────── #

    def load_predictions_from_file(self, predictions_path: str) -> tuple:
        """Load predictions from a previously saved .npz or .csv file.

        Supported formats:
            - .npz  (from save_artifacts or visualisation.py) with keys:
                     y_true, y_pred, y_proba, languages
            - .csv  with columns: y_true, y_pred, prob_fake, prob_real,
                     and optionally language

        Returns:
            (y_true, y_pred, y_proba, languages)
        """
        print(f"\n  Loading predictions from: {predictions_path}")

        if not os.path.isfile(predictions_path):
            raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

        ext = os.path.splitext(predictions_path)[1].lower()

        if ext == ".npz":
            npz = np.load(predictions_path, allow_pickle=True)
            y_true = npz["y_true"] if "y_true" in npz else np.array([], dtype=int)
            y_pred = npz["y_pred"] if "y_pred" in npz else np.array([], dtype=int)
            y_proba = npz["y_proba"] if "y_proba" in npz else np.zeros((0, 2))
            languages = npz["languages"].tolist() if "languages" in npz else []
        elif ext == ".csv":
            df = pd.read_csv(predictions_path)
            y_true = df["y_true"].values if "y_true" in df else df["true_label"].values
            y_pred = df["y_pred"].values if "y_pred" in df else df["predicted_label"].values
            if "prob_fake" in df and "prob_real" in df:
                y_proba = np.column_stack([df["prob_fake"].values, df["prob_real"].values])
            elif "y_proba" in df:
                y_proba = df["y_proba"].values[:, None]
            else:
                y_proba = np.zeros((len(y_true), 2))
            languages = df["language"].tolist() if "language" in df else []
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use .npz or .csv")

        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        y_proba = np.asarray(y_proba, dtype=float)
        if y_proba.ndim == 1:
            y_proba = y_proba[:, None]

        print(f"  Loaded {len(y_true):,} predictions")
        print(f"  Unique labels: {np.unique(y_true).tolist()}")
        if languages:
            unique_langs = sorted(set(languages))
            print(f"  Languages: {unique_langs}")

        return y_true, y_pred, y_proba, languages

    def _find_existing_predictions(self) -> str | None:
        """Auto-search for existing prediction files for this model type."""
        import glob
        search_dirs = ["outputs/runs", "outputs/evaluation", "outputs"]
        for root in search_dirs:
            if not os.path.isdir(root):
                continue
            # Look for model-type-specific directories first
            patterns = [
                os.path.join(root, f"{self.model_type}_*", "predictions.npz"),
                os.path.join(root, f"{self.model_type}_*", "predictions.csv"),
                os.path.join(root, "**", "predictions.npz"),
            ]
            for pat in patterns:
                matches = sorted(glob.glob(pat, recursive=True), key=os.path.getmtime, reverse=True)
                if matches:
                    return matches[0]
        return None

    # ── Full pipeline ────────────────────────────────────────────────────── #

    def run_full_evaluation(
        self,
        test_csv_path: str = "data/processed/test.csv",
        batch_size: int = 16,
        max_length: int = 512,
        history_json_path: str = None,
        predictions_path: str = None,
        skip_eval: bool = False,
    ) -> tuple:
        """Run the complete evaluation + visualization pipeline.

        Args:
            test_csv_path:     Path to test CSV (needs 'cleaned_text', 'label', 'language').
            batch_size:        Evaluation batch size.
            max_length:        Tokenisation max length.
            history_json_path: Optional path to training history.json for history plots.
            predictions_path:  Path to pre-saved predictions (.npz or .csv).
                               If provided, skips model inference entirely.
            skip_eval:         If True, search for existing predictions automatically.

        Returns:
            (metrics, per_lang_metrics)
        """
        run_inference = True

        # ── Decide whether to run evaluation or load existing predictions ── #
        if predictions_path:
            # Explicit predictions file provided
            run_inference = False
        elif skip_eval:
            # Auto-find predictions
            found = self._find_existing_predictions()
            if found:
                print(f"\n  Found existing predictions: {found}")
                predictions_path = found
                run_inference = False
            else:
                print("\n  No existing predictions found. Will run evaluation.")
        else:
            # Interactive prompt
            print("\n" + "-" * 70)
            print("  Do you want to run model evaluation on test data?")
            print("  This loads the model and runs inference on the full test set.")
            print("  (Can take a long time for large datasets)")
            print()

            # Check for existing predictions
            found = self._find_existing_predictions()
            if found:
                print(f"  Found existing predictions: {found}")
                print()

            choice = input("  Run evaluation? [y/N]: ").strip().lower()
            print("-" * 70)

            if choice in ("y", "yes"):
                run_inference = True
            else:
                run_inference = False
                if found:
                    predictions_path = found
                else:
                    # Ask for path
                    custom = input("  Enter path to predictions file (.npz or .csv): ").strip()
                    if custom and os.path.isfile(custom):
                        predictions_path = custom
                    else:
                        print("  No valid predictions file. Running evaluation instead.")
                        run_inference = True

        # ── Execute chosen path ──────────────────────────────────────────── #
        if run_inference:
            # Full inference pipeline
            self.load_model()
            self.load_test_data(test_csv_path, max_length=max_length, batch_size=batch_size)
            y_true, y_pred, y_proba, languages = self.evaluate_model()
        else:
            # Load from file
            y_true, y_pred, y_proba, languages = self.load_predictions_from_file(predictions_path)

        # 4. Compute metrics
        metrics = self.compute_metrics(y_true, y_pred, y_proba)
        per_lang_metrics = self.compute_per_language_metrics(y_true, y_pred, languages)

        # 5. Print results
        self.print_results(metrics, per_lang_metrics)

        # 6. Load training history if available
        history = None
        if history_json_path and os.path.isfile(history_json_path):
            with open(history_json_path, "r") as f:
                history = json.load(f)
            print(f"  Training history loaded from: {history_json_path}")
        else:
            # Try to find history from latest run
            runs_root = "outputs/runs"
            if os.path.isdir(runs_root):
                import glob
                run_dirs = sorted(
                    glob.glob(os.path.join(runs_root, f"{self.model_type}_*")),
                    key=os.path.getmtime, reverse=True,
                )
                for rd in run_dirs:
                    hp = os.path.join(rd, "history.json")
                    if os.path.isfile(hp):
                        with open(hp, "r") as f:
                            history = json.load(f)
                        print(f"  Training history auto-loaded from: {hp}")
                        break

        # 7. Generate ALL visualizations
        self.generate_all_visualizations(
            y_true, y_pred, y_proba, languages,
            metrics, per_lang_metrics, history,
        )

        # 8. Save artifacts
        self.save_artifacts(y_true, y_pred, y_proba, languages, metrics, per_lang_metrics)

        print("\n" + "=" * 70)
        print(" EVALUATION COMPLETE! ".center(70, "="))
        print("=" * 70)
        print(f"  All results saved to: {self.output_dir}")
        print("=" * 70 + "\n")

        return metrics, per_lang_metrics


# ════════════════════════════════════════════════════════════════════════════ #
#  CLI entry point                                                             #
# ════════════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained fake news detection model and generate all visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (will ask whether to evaluate or skip):
  python utils/eval_and_vis.py -c models/checkpoints/muril_best.pt

  # Skip evaluation, load saved predictions:
  python utils/eval_and_vis.py -c models/checkpoints/muril_best.pt --predictions outputs/runs/muril_xxx/predictions.npz

  # Force evaluation (no prompt):
  python utils/eval_and_vis.py -c models/checkpoints/ensemble_best.pt -m ensemble --eval

  # Force skip (auto-search for predictions, no prompt):
  python utils/eval_and_vis.py -c models/checkpoints/muril_best.pt --skip-eval
        """,
    )
    parser.add_argument(
        "--checkpoint", "-c", type=str, required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--model-type", "-m", type=str, default=None,
        help="Model type: xlm-roberta | muril | ensemble (auto-detected if omitted)",
    )
    parser.add_argument(
        "--test-csv", type=str, default="data/processed/test.csv",
        help="Path to test CSV file (default: data/processed/test.csv)",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=None,
        help="Output directory (default: outputs/evaluation/<type>_<timestamp>)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Evaluation batch size (default: 16)",
    )
    parser.add_argument(
        "--max-length", type=int, default=512,
        help="Tokenisation max length (default: 512)",
    )
    parser.add_argument(
        "--history", type=str, default=None,
        help="Path to training history.json for training history plots",
    )
    parser.add_argument(
        "--predictions", "-p", type=str, default=None,
        help="Path to pre-saved predictions (.npz or .csv). Skips model inference.",
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip evaluation and auto-search for existing prediction files.",
    )
    parser.add_argument(
        "--eval", action="store_true", dest="force_eval",
        help="Force evaluation (no interactive prompt).",
    )
    args = parser.parse_args()

    evaluator = ModelEvaluatorVisualizer(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        output_dir=args.output_dir,
    )

    # Determine skip_eval: --eval flag overrides --skip-eval
    skip_eval = args.skip_eval
    predictions_path = args.predictions
    if args.force_eval:
        skip_eval = False
        predictions_path = None  # force fresh evaluation

    evaluator.run_full_evaluation(
        test_csv_path=args.test_csv,
        batch_size=args.batch_size,
        max_length=args.max_length,
        history_json_path=args.history,
        predictions_path=predictions_path,
        skip_eval=skip_eval,
    )
