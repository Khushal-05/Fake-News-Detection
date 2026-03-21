"""
Training script for fake news detection.

Supports three model modes:
    model_type = 'xlm-roberta'  →  XLMRobertaFakeNewsClassifier (standalone)
    model_type = 'muril'        →  MuRILFakeNewsClassifier       (standalone)
    model_type = 'ensemble'     →  EnsembleFakeNewsClassifier    (both combined)

Key fixes from original:
    - Flat single LR replaced with differential LRs via model-provided
      build_optimizer_and_scheduler() — critical for fine-tuning transformers
    - token_type_ids forwarded to models that need them (MuRIL / ensemble)
    - Ensemble uses NLLLoss for weighted_avg/max and CrossEntropyLoss for learned
    - save_model() stores full training context (epoch, optimizer, scheduler,
      config, metrics) so training can be resumed
    - Checkpoint loading in __main__ is robust: handles model_state_dict,
      state_dict, raw dicts, and strips 'module.' prefix (DDP)
    - model_type normalised (handles 'xlm_roberta', 'xlmroberta', etc.)
    - Ensemble wired end-to-end: sub-models built, optional pre-trained
      checkpoints loaded, then wrapped in EnsembleFakeNewsClassifier
    - Run outputs (history, predictions, config) saved to timestamped run dir
    - pin_memory only activated when CUDA is available (avoids CPU warning)
    - Best checkpoint tracked by val F1 (more robust than accuracy for
      imbalanced multilingual datasets)
"""

import os
import time
import json
import shutil
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data.dataset import MultilingualFakeNewsDataset
from models.xlm_roberta_model import (
    XLMRobertaFakeNewsClassifier,
    build_optimizer_and_scheduler as xlmr_build_opt,
)
from models.muril_model import (
    MuRILFakeNewsClassifier,
    build_optimizer_and_scheduler as muril_build_opt,
)
from models.ensemble_model import (
    EnsembleFakeNewsClassifier,
    build_optimizer_and_scheduler as ensemble_build_opt,
)


# ── Model-type helpers ───────────────────────────────────────────────────── #

def _normalise_model_type(raw: str) -> str:
    """Normalise model_type to one of: 'xlm-roberta', 'muril', 'ensemble'."""
    key = raw.strip().lower().replace("-", "").replace("_", "")
    if key == "xlmroberta":
        return "xlm-roberta"
    if key == "muril":
        return "muril"
    if key == "ensemble":
        return "ensemble"
    raise ValueError(
        f"Unknown model_type='{raw}'. "
        "Choose from: 'xlm-roberta', 'muril', 'ensemble'."
    )


# ── Checkpoint helpers ───────────────────────────────────────────────────── #

def _extract_state_dict(ckpt) -> dict:
    """Pull model weights out of any checkpoint format."""
    if not isinstance(ckpt, dict):
        return ckpt
    for key in ("model_state_dict", "state_dict", "model"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt   # assume it is already a plain state dict


def _strip_module_prefix(state_dict: dict) -> dict:
    """Remove 'module.' prefix left by DistributedDataParallel."""
    if any(k.startswith("module.") for k in state_dict):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _safe_load(model: nn.Module, state_dict: dict) -> None:
    """Load state dict strictly; fall back to strict=False on key mismatch."""
    state_dict = _strip_module_prefix(state_dict)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"  [warn] Strict load failed ({e}). Retrying with strict=False.")
        model.load_state_dict(state_dict, strict=False)


# ════════════════════════════════════════════════════════════════════════════ #
#  FakeNewsTrainer                                                             #
# ════════════════════════════════════════════════════════════════════════════ #

class FakeNewsTrainer:
    """
    Unified trainer for XLM-RoBERTa, MuRIL, and Ensemble models.

    Changes from original:
        - Differential LRs via model-native build_optimizer_and_scheduler()
        - token_type_ids routed correctly per model type
        - Loss function selected per model type (NLLLoss vs CrossEntropyLoss)
        - Richer checkpoints (epoch, optimizer, scheduler, config, metrics)
        - pin_memory only on CUDA
        - warmup_ratio instead of raw warmup_steps
        - Best checkpoint tracked by val F1
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset,
        model_type: str = "muril",
        device: str = "cuda",
        batch_size: int = 16,
        encoder_lr: float = 2e-5,
        classifier_lr: float = 1e-4,
        ensemble_lr: float = 1e-3,
        num_epochs: int = 3,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        use_wandb: bool = False,
        config: dict = None,
    ):
        self.model      = model.to(device)
        self.model_type = _normalise_model_type(model_type)
        self.device     = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.use_wandb  = use_wandb
        self.config     = config or {}

        # ── Loss function ────────────────────────────────────────────────── #
        # Ensemble weighted_avg/max paths output log-probabilities → NLLLoss.
        # All other paths output raw logits → CrossEntropyLoss.
        if (
            self.model_type == "ensemble"
            and isinstance(model, EnsembleFakeNewsClassifier)
            and model.ensemble_method in ("weighted_avg", "max")
        ):
            self.criterion = nn.NLLLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        # ── DataLoaders ──────────────────────────────────────────────────── #
        pin = (device == "cuda" and torch.cuda.is_available())

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=pin,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=pin,
        )

        # ── Optimizer + scheduler with differential LRs ──────────────────── #
        num_training_steps = len(self.train_loader) * num_epochs
        opt_kwargs = dict(
            num_training_steps=num_training_steps,
            encoder_lr=encoder_lr,
            classifier_lr=classifier_lr,
            warmup_ratio=warmup_ratio,
        )

        if self.model_type == "xlm-roberta":
            self.optimizer, self.scheduler = xlmr_build_opt(model, **opt_kwargs)
        elif self.model_type == "muril":
            self.optimizer, self.scheduler = muril_build_opt(model, **opt_kwargs)
        else:  # ensemble
            self.optimizer, self.scheduler = ensemble_build_opt(
                model, ensemble_lr=ensemble_lr, **opt_kwargs
            )

        # ── State tracking ────────────────────────────────────────────────── #
        self.best_val_f1 = 0.0
        self.best_epoch  = 0
        self.history = {
            "train_loss": [], "train_accuracy": [], "train_f1": [],
            "val_loss":   [], "val_accuracy":   [], "val_f1":   [],
            "val_precision": [], "val_recall": [],
        }

    # ── Batch routing helpers ─────────────────────────────────────────────── #

    def _unpack_batch(self, batch: dict):
        """
        Extract tensors from a dataloader batch.
        token_type_ids is forwarded only for MuRIL and ensemble
        (XLM-RoBERTa does not use segment embeddings).
        """
        input_ids      = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels         = batch["label"].to(self.device)

        token_type_ids = None
        if self.model_type in ("muril", "ensemble"):
            raw_tti = batch.get("token_type_ids")
            if isinstance(raw_tti, torch.Tensor):
                token_type_ids = raw_tti.to(self.device)

        return input_ids, attention_mask, token_type_ids, labels

    def _forward(self, input_ids, attention_mask, token_type_ids):
        """Route inputs to model respecting each model's forward() signature."""
        if self.model_type == "xlm-roberta":
            return self.model(input_ids, attention_mask)
        return self.model(input_ids, attention_mask, token_type_ids)

    # ── train_epoch ──────────────────────────────────────────────────────── #

    def train_epoch(self) -> dict:
        """Run one full training epoch. Returns dict of train metrics."""
        self.model.train()
        total_loss      = 0.0
        all_predictions = []
        all_labels      = []

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            ids, mask, tti, labels = self._unpack_batch(batch)

            self.optimizer.zero_grad()
            logits = self._forward(ids, mask, tti)
            loss   = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        _, _, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average="weighted", zero_division=0
        )
        return {"loss": avg_loss, "accuracy": accuracy, "f1": f1}

    # ── validate ─────────────────────────────────────────────────────────── #

    def validate(self) -> dict:
        """Evaluate on validation set. Returns dict of val metrics."""
        self.model.eval()
        total_loss      = 0.0
        all_predictions = []
        all_labels      = []
        all_languages   = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                ids, mask, tti, labels = self._unpack_batch(batch)

                logits = self._forward(ids, mask, tti)
                loss   = self.criterion(logits, labels)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                lang_batch = batch.get("language")
                if lang_batch is not None:
                    all_languages.extend(
                        lang_batch.cpu().numpy().tolist()
                        if isinstance(lang_batch, torch.Tensor)
                        else list(lang_batch)
                    )

        avg_loss  = total_loss / len(self.val_loader)
        accuracy  = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average="weighted", zero_division=0
        )

        per_lang_acc = {}
        if all_languages:
            df = pd.DataFrame({
                "label":      all_labels,
                "prediction": all_predictions,
                "language":   all_languages,
            })
            for lang in df["language"].unique():
                ldf = df[df["language"] == lang]
                per_lang_acc[lang] = accuracy_score(ldf["label"], ldf["prediction"])

        return {
            "loss":                  avg_loss,
            "accuracy":              accuracy,
            "precision":             precision,
            "recall":                recall,
            "f1":                    f1,
            "per_language_accuracy": per_lang_acc,
        }

    # ── save_model ───────────────────────────────────────────────────────── #

    def save_model(self, save_path: str, epoch: int = 0, val_metrics: dict = None) -> None:
        """Save full training checkpoint (weights + optimizer + scheduler + context)."""
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch":                epoch,
                "best_val_f1":          self.best_val_f1,
                "val_metrics":          val_metrics or {},
                "model_type":           self.model_type,
                "config":               self.config,
            },
            save_path,
        )
        print(f"  Checkpoint saved → {save_path}")

    # ── full training loop ────────────────────────────────────────────────── #

    def train(self, save_dir: str = "models/checkpoints") -> nn.Module:
        """
        Full training loop. Saves the best checkpoint (by val F1).
        Reloads best weights into the model before returning.
        """
        print(f"\nStarting training — {self.num_epochs} epoch(s), model={self.model_type}")
        best_path = os.path.join(save_dir, f"{self.model_type}_best.pt")

        for epoch in range(self.num_epochs):
            print(f"\n{'='*55}")
            print(f"Epoch {epoch + 1} / {self.num_epochs}")
            print(f"{'='*55}")

            train_m = self.train_epoch()
            val_m   = self.validate()

            # Record history
            self.history["train_loss"].append(train_m["loss"])
            self.history["train_accuracy"].append(train_m["accuracy"])
            self.history["train_f1"].append(train_m["f1"])
            self.history["val_loss"].append(val_m["loss"])
            self.history["val_accuracy"].append(val_m["accuracy"])
            self.history["val_precision"].append(val_m["precision"])
            self.history["val_recall"].append(val_m["recall"])
            self.history["val_f1"].append(val_m["f1"])

            print(
                f"\n  Train  loss={train_m['loss']:.4f}  "
                f"acc={train_m['accuracy']:.4f}  f1={train_m['f1']:.4f}"
            )
            print(
                f"  Val    loss={val_m['loss']:.4f}  "
                f"acc={val_m['accuracy']:.4f}  f1={val_m['f1']:.4f}  "
                f"prec={val_m['precision']:.4f}  rec={val_m['recall']:.4f}"
            )

            if val_m["per_language_accuracy"]:
                print("  Per-language val accuracy:")
                for lang, acc in sorted(
                    val_m["per_language_accuracy"].items(),
                    key=lambda x: x[1], reverse=True,
                ):
                    print(f"    {lang:>8}: {acc:.4f}")

            if self.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "epoch":          epoch + 1,
                        "train/loss":     train_m["loss"],
                        "train/accuracy": train_m["accuracy"],
                        "train/f1":       train_m["f1"],
                        "val/loss":       val_m["loss"],
                        "val/accuracy":   val_m["accuracy"],
                        "val/precision":  val_m["precision"],
                        "val/recall":     val_m["recall"],
                        "val/f1":         val_m["f1"],
                    })
                except Exception as e:
                    print(f"  [wandb] logging failed: {e}")

            if val_m["f1"] > self.best_val_f1:
                self.best_val_f1 = val_m["f1"]
                self.best_epoch  = epoch + 1
                self.save_model(best_path, epoch=epoch + 1, val_metrics=val_m)
                print(f"  ★ New best val F1={self.best_val_f1:.4f} (epoch {self.best_epoch})")

        print(f"\n{'='*55}")
        print(f"Training complete.")
        print(f"Best val F1 = {self.best_val_f1:.4f} at epoch {self.best_epoch}")
        print(f"Best checkpoint: {best_path}")

        # Reload best weights before returning
        ckpt = torch.load(best_path, map_location=self.device)
        _safe_load(self.model, _extract_state_dict(ckpt))
        return self.model


# ════════════════════════════════════════════════════════════════════════════ #
#  CONFIG (module-level so test.py / inference.py can import it)              #
# ════════════════════════════════════════════════════════════════════════════ #

CONFIG = {
    "model_type":       "muril",              # 'xlm-roberta' | 'muril' | 'ensemble'
    "model_name":       "google/muril-base-cased",
    # Ensemble-specific (ignored for standalone models):
    # "xlmr_model_name":   "xlm-roberta-base",
    # "xlmr_checkpoint":   "",   # path to pre-fine-tuned XLM-R weights
    # "muril_checkpoint":  "",   # path to pre-fine-tuned MuRIL weights
    # "ensemble_method":   "weighted_avg",   # 'weighted_avg'|'max'|'learned'
    # "ensemble_weights":  [0.4, 0.6],       # only for weighted_avg
    "batch_size":       16,
    "encoder_lr":       2e-5,
    "classifier_lr":    1e-4,
    "ensemble_lr":      1e-3,
    "num_epochs":       3,
    "warmup_ratio":     0.1,
    "max_length":       512,
    "dropout":          0.3,
    "freeze_layers":    0,
    "use_wandb":        False,
    "device":           "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir":         "models/checkpoints",
}


# ════════════════════════════════════════════════════════════════════════════ #
#  Entry point                                                                 #
# ════════════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    model_type = _normalise_model_type(CONFIG["model_type"])
    device     = CONFIG["device"]
    print(f"Device: {device}  |  Model type: {model_type}")

    # ── Tokenizer ─────────────────────────────────────────────────────────── #
    TOKENIZER_MAP = {
        "xlm-roberta": "xlm-roberta-base",
        "muril":       "google/muril-base-cased",
        # Ensemble: tokenise with the primary model (MuRIL favoured for Indian data)
        "ensemble":    "google/muril-base-cased",
    }
    tokenizer_name = TOKENIZER_MAP[model_type]

    # ── Data ──────────────────────────────────────────────────────────────── #
    print("Loading data...")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df   = pd.read_csv("data/processed/val.csv")

    print("Train_Length: ",len(train_df))
    print("Val_Length: ",len(val_df))
    print("Train_Shape: ",train_df.shape)

    train_dataset = MultilingualFakeNewsDataset(
        texts=train_df["cleaned_text"].values,
        labels=train_df["label"].values,
        languages=train_df["language"].values,
        tokenizer_name=tokenizer_name,
        max_length=CONFIG["max_length"],
    )
    val_dataset = MultilingualFakeNewsDataset(
        texts=val_df["cleaned_text"].values,
        labels=val_df["label"].values,
        languages=val_df["language"].values,
        tokenizer_name=tokenizer_name,
        max_length=CONFIG["max_length"],
    )

    # ── Model ─────────────────────────────────────────────────────────────── #
    print(f"Building {model_type} model...")
    common_kwargs = dict(
        num_classes=2,
        dropout=CONFIG["dropout"],
        freeze_layers=CONFIG["freeze_layers"],
    )

    if model_type == "xlm-roberta":
        model = XLMRobertaFakeNewsClassifier(
            model_name=CONFIG.get("model_name", "xlm-roberta-base"),
            **common_kwargs,
        )

    elif model_type == "muril":
        model = MuRILFakeNewsClassifier(
            model_name=CONFIG.get("model_name", "google/muril-base-cased"),
            **common_kwargs,
        )

    else:  # ensemble
        xlmr = XLMRobertaFakeNewsClassifier(
            model_name=CONFIG.get("xlmr_model_name", "xlm-roberta-base"),
            **common_kwargs,
        )
        muril = MuRILFakeNewsClassifier(
            model_name=CONFIG.get("model_name", "google/muril-base-cased"),
            **common_kwargs,
        )
        # Load optional pre-fine-tuned sub-model weights
        for sub_model, ckpt_key, label in [
            (xlmr,  "xlmr_checkpoint",  "XLM-R"),
            (muril, "muril_checkpoint", "MuRIL"),
        ]:
            ckpt_path = CONFIG.get(ckpt_key, "").strip()
            if ckpt_path and os.path.isfile(ckpt_path):
                print(f"  Loading {label} checkpoint: {ckpt_path}")
                _safe_load(sub_model, _extract_state_dict(
                    torch.load(ckpt_path, map_location=device)
                ))

        model = EnsembleFakeNewsClassifier(
            xlmr_model=xlmr,
            muril_model=muril,
            num_classes=2,
            ensemble_method=CONFIG.get("ensemble_method", "weighted_avg"),
            weights=CONFIG.get("ensemble_weights"),
            freeze_base_models=False,
        )

    param_info = model.count_parameters()
    print(
        f"  Trainable: {param_info['trainable']:,} | "
        f"Frozen: {param_info['frozen']:,} | "
        f"Total: {param_info['total']:,}"
    )

    # ── wandb ─────────────────────────────────────────────────────────────── #
    if CONFIG["use_wandb"]:
        try:
            import wandb
            wandb.init(project="indian-fake-news-detection", config=CONFIG)
        except Exception as e:
            print(f"[wandb] init failed: {e}. Continuing without wandb.")
            CONFIG["use_wandb"] = False

    # ── Trainer ───────────────────────────────────────────────────────────── #
    trainer = FakeNewsTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_type=model_type,
        device=device,
        batch_size=CONFIG["batch_size"],
        encoder_lr=CONFIG["encoder_lr"],
        classifier_lr=CONFIG["classifier_lr"],
        ensemble_lr=CONFIG.get("ensemble_lr", 1e-3),
        num_epochs=CONFIG["num_epochs"],
        warmup_ratio=CONFIG["warmup_ratio"],
        use_wandb=CONFIG["use_wandb"],
        config=CONFIG,
    )

    trained_model = trainer.train(save_dir=CONFIG["save_dir"])

    # ── Save run outputs ──────────────────────────────────────────────────── #
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join("outputs", "runs", f"{model_type}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    best_ckpt = os.path.join(CONFIG["save_dir"], f"{model_type}_best.pt")
    if os.path.exists(best_ckpt):
        shutil.copy2(best_ckpt, os.path.join(run_dir, "checkpoint.pt"))

    with open(os.path.join(run_dir, "history.json"), "w") as fh:
        json.dump(trainer.history, fh, indent=2)

    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        safe_cfg = {
            k: str(v) if not isinstance(v, (str, int, float, bool, list, type(None))) else v
            for k, v in CONFIG.items()
        }
        json.dump(safe_cfg, fh, indent=2)

    # Val-set predictions
    trained_model.eval()
    y_true, y_pred, probs_list, languages = [], [], [], []

    with torch.no_grad():
        for batch in trainer.val_loader:
            ids      = batch["input_ids"].to(device)
            mask     = batch["attention_mask"].to(device)
            labels   = batch["label"]
            tti      = None
            if model_type in ("muril", "ensemble"):
                raw_tti = batch.get("token_type_ids")
                if isinstance(raw_tti, torch.Tensor):
                    tti = raw_tti.to(device)

            if model_type == "xlm-roberta":
                logits = trained_model(ids, mask)
            else:
                logits = trained_model(ids, mask, tti)

            # Convert to probabilities (handle log-prob output for ensemble)
            if (
                model_type == "ensemble"
                and isinstance(trained_model, EnsembleFakeNewsClassifier)
                and trained_model.ensemble_method in ("weighted_avg", "max")
            ):
                probs = torch.exp(logits).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            y_pred.extend(probs.argmax(axis=1).tolist())
            y_true.extend(labels.numpy().tolist())
            probs_list.append(probs)

            lang_batch = batch.get("language")
            if lang_batch is not None:
                languages.extend(
                    lang_batch.cpu().numpy().tolist()
                    if isinstance(lang_batch, torch.Tensor)
                    else list(lang_batch)
                )

    y_proba_full  = np.vstack(probs_list) if probs_list else np.zeros((len(y_pred), 2))
    y_true_arr    = np.array(y_true,    dtype=object)
    y_pred_arr    = np.array(y_pred,    dtype=object)
    languages_arr = np.array(languages, dtype=object)

    np.savez_compressed(
        os.path.join(run_dir, "predictions.npz"),
        y_true=y_true_arr, y_pred=y_pred_arr,
        y_proba=y_proba_full, languages=languages_arr,
    )

    scalar_probs = (
        y_proba_full[:, 1]
        if y_proba_full.ndim == 2 and y_proba_full.shape[1] == 2
        else y_proba_full.max(axis=1)
    )
    with open(os.path.join(run_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        # Column name must be "y_proba" — visualisation.py reads r["y_proba"] from CSV
        writer.writerow(["y_true", "y_pred", "y_proba", "language"])
        for row in zip(
            y_true_arr.tolist(), y_pred_arr.tolist(),
            scalar_probs.tolist(), languages_arr.tolist(),
        ):
            writer.writerow(row)

    print(f"\nRun outputs saved to: {run_dir}")

    if CONFIG["use_wandb"]:
        try:
            wandb.finish()
        except Exception:
            pass
