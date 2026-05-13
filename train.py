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

Resumption-dip fixes (v2):
    FIX-1 — Scheduler curve mismatch (primary cause of the dip):
        The resume checkpoint now stores total_steps and completed_steps.
        On resume, the scheduler is rebuilt with the SAME total_steps as the
        original run, not a fresh N-epoch total. This preserves the exact
        shape of the LR decay curve so the LR at step K in session 2 is
        identical to what it would have been in session 1.
        Implementation: _peek_resume_epoch() reads epoch/total_steps from
        the checkpoint BEFORE __init__ builds the optimizer, so
        num_training_steps is always consistent across sessions.

    FIX-2 — DataLoader shuffle seed:
        A seeded torch.Generator is used for the train DataLoader so batch
        ordering is deterministic and reproducible across sessions.
        The seed advances by 1 each epoch (via _set_epoch_seed()) so
        batches still differ between epochs as intended.

    FIX-3 — start_epoch extracted before optimizer build:
        _peek_resume_epoch() is a lightweight static helper that reads only
        the epoch and total_steps fields from the checkpoint dict without
        touching model weights. This value is used in __init__ to correctly
        calculate num_training_steps for the scheduler before
        _load_resume_checkpoint() runs the full state restoration.

Model checkpointing + early stopping (v3):
    - Best checkpoint now saved by val_loss improvement, not val_f1.
      val_loss is a more direct signal of generalisation — the model with
      the lowest val_loss is saved as _best.pt regardless of whether F1
      also improved that epoch.  val_f1 tracking is kept alongside for
      reporting and history purposes.
    - Early stopping: if val_loss does not improve for `early_stopping_patience`
      consecutive epochs the training loop breaks automatically.  The
      patience counter is saved in both _resume.pt and _best.pt so it is
      preserved correctly across Kaggle sessions.
    - To disable early stopping, set early_stopping_patience=0 (or omit it
      from CONFIG — the default is 0, meaning no early stopping).
    - Recommended workflow: set num_epochs=15 (or any safely high ceiling)
      and early_stopping_patience=3.  Training will stop as soon as the
      model stops improving, typically well before epoch 15.
"""

import os
import sys
import time
import json
import shutil
import csv
import signal
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ── Robust imports for both flat and package layouts ─────────────────────── #
_THIS_DIR   = os.path.abspath(os.path.dirname(__file__))
_PARENT_DIR = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
for _p in (_THIS_DIR, _PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
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
except ModuleNotFoundError:
    # Flat layout: all modules live at the project root
    from dataset import MultilingualFakeNewsDataset                         # noqa: E402
    from xlm_roberta_model import (                                         # noqa: E402
        XLMRobertaFakeNewsClassifier,
        build_optimizer_and_scheduler as xlmr_build_opt,
    )
    from muril_model import (                                               # noqa: E402
        MuRILFakeNewsClassifier,
        build_optimizer_and_scheduler as muril_build_opt,
    )
    from ensemble_model import (                                            # noqa: E402
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


# ── Checkpoint helpers — shared via utils.checkpoint ────────────────────── #
try:
    from utils.checkpoint import extract_state_dict as _extract_state_dict
    from utils.checkpoint import strip_module_prefix as _strip_module_prefix
    from utils.checkpoint import safe_load as _safe_load
except ImportError:
    # Flat layout fallback (utils/ not on path yet — define inline)
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

    def _safe_load(model, state_dict: dict, verbose: bool = True) -> None:
        state_dict = _strip_module_prefix(state_dict)
        try:
            model.load_state_dict(state_dict)
            if verbose:
                print("  Weights loaded (strict=True).")
        except RuntimeError as e:
            if verbose:
                print(f"  Strict load failed ({e}). Retrying with strict=False.")
            model.load_state_dict(state_dict, strict=False)


# ── FIX-1 / FIX-3: lightweight peek helper ──────────────────────────────── #

def _peek_resume_epoch(resume_checkpoint: dict | None) -> tuple[int, int | None]:
    """
    Read epoch and total_steps from a resume checkpoint WITHOUT loading any
    tensors or model weights.  Called before __init__ builds the optimizer so
    that num_training_steps can be set correctly.

    Args:
        resume_checkpoint: The raw checkpoint dict, or None for a fresh run.

    Returns:
        (start_epoch, total_steps)
            start_epoch  — number of epochs already completed (0 if fresh).
            total_steps  — the total_steps value stored when the checkpoint
                           was saved, or None if the checkpoint pre-dates
                           this fix (safe fallback: caller recomputes it).
    """
    if resume_checkpoint is None:
        return 0, None
    start_epoch = int(resume_checkpoint.get("epoch", 0))
    total_steps = resume_checkpoint.get("total_steps", None)
    if total_steps is not None:
        total_steps = int(total_steps)
    return start_epoch, total_steps


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

    Resumption-dip fixes (v2):
        - Scheduler rebuilt with the same total_steps as the original session
          so the LR decay curve shape is preserved exactly across sessions.
        - Train DataLoader uses a seeded Generator for reproducible shuffling.
        - start_epoch is extracted from the checkpoint BEFORE the optimizer
          is built so num_training_steps is always consistent.

    Model checkpointing + early stopping (v3):
        - _best.pt saved whenever val_loss hits a new low (not val_f1).
        - Early stopping breaks the loop after `early_stopping_patience`
          epochs without val_loss improvement (0 = disabled).
        - Both patience counter and best_val_loss are persisted in checkpoints
          so early stopping state survives Kaggle session boundaries.
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
        num_epochs: int = 15,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        use_wandb: bool = False,
        config: dict = None,
        resume_checkpoint: dict = None,
        log_dir: str = "outputs/logs",
        save_resume_every_n_epochs: int = 1,
        dataloader_seed: int = 42,
        early_stopping_patience: int = 0,   # 0 = disabled; 3 is recommended
        checkpoint_metric: str = "val_f1",  # 'val_f1' (recommended) or 'val_loss'
    ):
        self.model      = model.to(device)
        self.model_type = _normalise_model_type(model_type)
        self.device     = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.use_wandb  = use_wandb
        self.config     = config or {}
        self.dataloader_seed = dataloader_seed  # FIX-2

        # Checkpoint metric: which validation metric triggers saving _best.pt
        # 'val_f1'   — save when weighted F1 improves (recommended for imbalanced multilingual data)
        # 'val_loss' — save when validation loss decreases (smoother signal, less directly meaningful)
        if checkpoint_metric not in ("val_f1", "val_loss"):
            raise ValueError(
                f"checkpoint_metric='{checkpoint_metric}' is invalid. "
                "Choose 'val_f1' (recommended) or 'val_loss'."
            )
        self.checkpoint_metric = checkpoint_metric

        # Early stopping patience is applied to the same metric as checkpoint_metric.
        # patience=0 means disabled — the loop always runs all num_epochs.
        # patience=N means: stop if the chosen metric has not improved for N consecutive epochs.
        self.early_stopping_patience = early_stopping_patience

        # Resumable training state
        self.log_dir = log_dir
        self.save_resume_every_n_epochs = save_resume_every_n_epochs
        self.interrupted = False

        # ── FIX-3: peek at start_epoch BEFORE building optimizer ─────────── #
        # This must happen first so num_training_steps below is correct.
        self.start_epoch, _stored_total_steps = _peek_resume_epoch(resume_checkpoint)

        # ── Loss function ────────────────────────────────────────────────── #
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

        # FIX-2: seeded Generator so batch order is reproducible across sessions.
        # The seed advances per epoch inside _set_epoch_seed(), so epochs still
        # see different orderings from each other.
        self._train_generator = torch.Generator()
        self._train_generator.manual_seed(dataloader_seed)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=pin,
            generator=self._train_generator,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=pin,
        )

        # ── FIX-1: compute num_training_steps consistently across sessions ─ #
        #
        # Fresh run:  total_steps = steps_per_epoch * num_epochs   (unchanged)
        #
        # Resumed run with stored total_steps (new checkpoints):
        #   Reuse the EXACT same total_steps that the original session used.
        #   This preserves the LR decay curve shape — at step K the LR is
        #   the same value it would have been had training never been interrupted.
        #
        # Resumed run WITHOUT stored total_steps (old checkpoints, backward compat):
        #   Fall back to recomputing from num_epochs so old checkpoints still work.
        #   The dip may still occur for these old checkpoints, but new ones are safe.
        #
        steps_per_epoch = len(self.train_loader)
        if _stored_total_steps is not None:
            # New-format checkpoint: preserve the original curve exactly.
            num_training_steps = _stored_total_steps
        else:
            # Fresh run OR old checkpoint without total_steps stored.
            num_training_steps = steps_per_epoch * num_epochs

        self._num_training_steps = num_training_steps   # stored for saving later

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
        self.best_val_f1   = 0.0
        self.best_val_loss = float("inf")
        self.best_epoch    = 0
        self._es_counter   = 0              # epochs without improvement on checkpoint_metric
        self._last_val_metrics: dict = {}   # most recent val metrics for emergency resume saves
        self.history = {
            "train_loss": [], "train_accuracy": [], "train_f1": [],
            "val_loss":   [], "val_accuracy":   [], "val_f1":   [],
            "val_precision": [], "val_recall": [],
        }

        # Setup persistent CSV logging
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_log_path = os.path.join(log_dir, f"{self.model_type}_metrics.csv")
        if not os.path.exists(self.metrics_log_path):
            with open(self.metrics_log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'epoch', 'train_loss', 'train_accuracy', 'train_f1',
                    'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1',
                    'best_val_f1', 'per_lang_acc_json'
                ])
                writer.writeheader()

        # Resume from checkpoint if provided.
        # Optimizer and scheduler are already built above with the correct
        # total_steps; _load_resume_checkpoint() restores their saved states
        # (momentum buffers, last_epoch counter, etc.) on top of that.
        if resume_checkpoint is not None:
            self._load_resume_checkpoint(resume_checkpoint)

        # Setup graceful interrupt handler
        signal.signal(signal.SIGINT, self._interrupt_handler)
        signal.signal(signal.SIGTERM, self._interrupt_handler)

    # ── FIX-2: per-epoch seed helper ─────────────────────────────────────── #

    def _set_epoch_seed(self, epoch: int) -> None:
        """
        Advance the DataLoader Generator seed to a deterministic value for
        this epoch.  Ensures:
            - Same session vs. resumed session → identical batch order.
            - Different epochs → different batch orderings (seed varies by epoch).

        Called at the top of each epoch iteration in train().
        """
        self._train_generator.manual_seed(self.dataloader_seed + epoch)

    # ── Batch routing helpers ─────────────────────────────────────────────── #

    def _unpack_batch(self, batch: dict) -> tuple:
        """Unpack batch into model-specific inputs and labels.

        Returns:
            (inputs, labels) where inputs is a tuple of tensors specific to
            the model type.
        """
        labels = batch["label"].to(self.device)

        if self.model_type == "ensemble":
            xlmr_ids  = batch["xlmr_ids"].to(self.device)
            xlmr_mask = batch["xlmr_mask"].to(self.device)
            muril_ids = batch["muril_ids"].to(self.device)
            muril_mask = batch["muril_mask"].to(self.device)
            muril_tti = batch.get("muril_tti")
            if muril_tti is not None:
                muril_tti = muril_tti.to(self.device)
            return (xlmr_ids, xlmr_mask, muril_ids, muril_mask, muril_tti), labels
        else:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            return (input_ids, attention_mask, token_type_ids), labels

    def _forward(self, inputs: tuple) -> torch.Tensor:
        """Route inputs to the appropriate model forward pass.

        FIX (BUG-12): removed redundant `model_type` parameter — callers
        always passed `self.model_type`, so it is used directly now.

        Args:
            inputs: Tuple of tensors specific to self.model_type.

        Returns:
            logits: Model output logits [batch_size, num_classes].
        """
        if self.model_type == "ensemble":
            xlmr_ids, xlmr_mask, muril_ids, muril_mask, muril_tti = inputs
            return self.model(xlmr_ids, xlmr_mask, muril_ids, muril_mask, muril_tti)
        elif self.model_type == "xlm-roberta":
            input_ids, attention_mask, _ = inputs
            return self.model(input_ids, attention_mask)
        else:  # muril
            input_ids, attention_mask, token_type_ids = inputs
            return self.model(input_ids, attention_mask, token_type_ids)

    # ── train_epoch ──────────────────────────────────────────────────────── #

    def train_epoch(self) -> dict:
        """Run one full training epoch.

        Returns:
            dict with keys: 'loss', 'accuracy', 'f1'
        """
        self.model.train()
        total_loss      = 0.0
        all_predictions = []
        all_labels      = []

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in pbar:
            inputs, labels = self._unpack_batch(batch)

            self.optimizer.zero_grad()
            logits = self._forward(inputs)
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
        """Evaluate on validation set.

        Returns:
            dict with keys: 'loss', 'accuracy', 'precision', 'recall', 'f1',
                            'per_language_accuracy'
        """
        # FIX (BUG-6): remember training state so we can restore it after
        # validation, preventing the model from being stuck in eval mode
        # when validate() is called standalone (e.g. from a notebook cell).
        was_training = self.model.training
        self.model.eval()
        total_loss      = 0.0
        all_predictions = []
        all_labels      = []
        all_languages   = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                inputs, labels = self._unpack_batch(batch)
                logits = self._forward(inputs)
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

        if was_training:
            self.model.train()

        return {
            "loss":                  avg_loss,
            "accuracy":              accuracy,
            "precision":             precision,
            "recall":                recall,
            "f1":                    f1,
            "per_language_accuracy": per_lang_acc,
        }

    # ── Resumable training helpers ────────────────────────────────────────── #

    def _interrupt_handler(self, signum, frame):
        """Handle Ctrl+C / system termination gracefully."""
        print("\n" + "=" * 70)
        print("INTERRUPT SIGNAL RECEIVED")
        print("=" * 70)
        print("Saving resume checkpoint before exit...")
        self.interrupted = True

    def _load_resume_checkpoint(self, checkpoint: dict) -> None:
        """
        Restore full training state from a resume checkpoint.

        The optimizer and scheduler were already built with the correct
        total_steps (via _peek_resume_epoch + FIX-1).  This method restores:
          1. Model weights  ← CRITICAL: without this the model starts from
             pretrained/random weights while the optimizer/scheduler resume
             from a later point in training — the primary cause of the dip.
          2. Optimizer + scheduler saved states (momentum buffers, last_epoch).
          3. Scalar tracking state: best metrics, history, early stopping counter.
        """
        print(f"\n  Loading resume state...")
        # start_epoch and total_steps already set in __init__ via _peek_resume_epoch

        # ── 1. Model weights — must happen FIRST ─────────────────────────── #
        if 'model_state_dict' in checkpoint:
            try:
                _safe_load(self.model, checkpoint['model_state_dict'])
                print("  ✓ Model weights restored")
            except Exception as e:
                print(f"  ⚠ Could not restore model weights: {e} — training may dip!")
        else:
            print("  ⚠ No model_state_dict in checkpoint — starting from current weights.")

        self.best_val_f1   = checkpoint.get('best_val_f1', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch    = checkpoint.get('best_epoch', 0)
        self._es_counter   = checkpoint.get('es_counter', 0)
        # Restore checkpoint_metric if saved; warn if mismatch with current setting
        saved_metric = checkpoint.get('checkpoint_metric')
        if saved_metric is not None and saved_metric != self.checkpoint_metric:
            print(
                f"  [warn] checkpoint_metric changed: saved='{saved_metric}' "
                f"current='{self.checkpoint_metric}'. "
                f"Using current setting '{self.checkpoint_metric}'."
            )
        if 'history' in checkpoint:
            self.history = checkpoint['history']

        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("  ✓ Optimizer state restored")
            except Exception as e:
                print(f"  ⚠ Could not restore optimizer: {e}")

        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("  ✓ Scheduler state restored")
            except Exception as e:
                print(f"  ⚠ Could not restore scheduler: {e}")

        # FIX-2: restore the DataLoader Generator to the correct per-epoch seed
        # so batch ordering in the resumed session matches what it would have been.
        self._set_epoch_seed(self.start_epoch)
        print("  ✓ DataLoader seed aligned to resumed epoch")

        print(f"  Resuming from epoch {self.start_epoch + 1}")
        print(f"  Checkpoint metric:    {self.checkpoint_metric}")
        print(f"  Best val loss so far: {self.best_val_loss:.4f} (epoch {self.best_epoch})")
        print(f"  Best val F1 so far:   {self.best_val_f1:.4f}")
        if self.early_stopping_patience > 0:
            print(f"  Early stopping ({self.checkpoint_metric}): {self._es_counter} / {self.early_stopping_patience}")

    def _log_metrics_to_csv(self, epoch: int, train_m: dict, val_m: dict) -> None:
        """Append per-epoch metrics to the persistent CSV log."""
        row = {
            'timestamp':        datetime.now().isoformat(),
            'epoch':            epoch,
            'train_loss':       train_m['loss'],
            'train_accuracy':   train_m['accuracy'],
            'train_f1':         train_m['f1'],
            'val_loss':         val_m['loss'],
            'val_accuracy':     val_m['accuracy'],
            'val_precision':    val_m['precision'],
            'val_recall':       val_m['recall'],
            'val_f1':           val_m['f1'],
            'best_val_f1':      self.best_val_f1,
            'per_lang_acc_json': json.dumps(val_m.get('per_language_accuracy', {})),
        }
        with open(self.metrics_log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

    def _save_resume_checkpoint(self, save_dir: str, epoch: int, val_metrics: dict) -> None:
        """
        Save full training state so the next session can resume seamlessly.

        FIX-1: total_steps is stored alongside epoch so the next session can
        rebuild the scheduler with the same curve shape without recomputing it
        from num_epochs (which might differ if the user changes the config).
        Early stopping state (best_val_loss, es_counter) is also persisted so
        patience counting is continuous across Kaggle session boundaries.
        """
        resume_path = os.path.join(save_dir, f"{self.model_type}_resume.pt")
        torch.save({
            "model_state_dict":      self.model.state_dict(),
            "optimizer_state_dict":  self.optimizer.state_dict(),
            "scheduler_state_dict":  self.scheduler.state_dict(),
            "epoch":                 epoch,
            "total_steps":           self._num_training_steps,  # FIX-1
            "best_val_f1":           self.best_val_f1,
            "best_val_loss":         self.best_val_loss,
            "best_epoch":            self.best_epoch,
            "es_counter":            self._es_counter,          # early stopping
            "checkpoint_metric":     self.checkpoint_metric,    # which metric triggers _best.pt
            "history":               self.history,
            "val_metrics":           val_metrics,
            "model_type":            self.model_type,
            "config":                self.config,
            "dataloader_seed":       self.dataloader_seed,      # FIX-2
            "timestamp":             datetime.now().isoformat(),
        }, resume_path)
        print(f"  Resume checkpoint saved → {resume_path}")

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
                "total_steps":          self._num_training_steps,  # FIX-1
                "best_val_f1":          self.best_val_f1,
                "best_val_loss":        self.best_val_loss,
                "es_counter":           self._es_counter,           # early stopping
                "checkpoint_metric":    self.checkpoint_metric,     # which metric triggers _best.pt
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
        Full training loop with val_loss checkpointing, early stopping,
        resumable checkpoints, and persistent logging.

        Checkpointing: _best.pt is saved whenever val_loss hits a new low.
        Early stopping: loop breaks after `early_stopping_patience` consecutive
                        epochs without a new val_loss low (0 = disabled).
        Resumption: starts from self.start_epoch; all early stopping state is
                    restored from checkpoint so patience counting is continuous
                    across Kaggle session boundaries.
        """
        print(f"\nStarting training — {self.num_epochs} epoch(s), model={self.model_type}")
        if self.start_epoch > 0:
            print(f"  (Resuming from epoch {self.start_epoch + 1})")
        print(f"  Scheduler total_steps={self._num_training_steps} "
              f"(steps_per_epoch={len(self.train_loader)})")
        if self.early_stopping_patience > 0:
            print(f"  Early stopping patience: {self.early_stopping_patience} epochs")
            print(f"  Early stopping counter on entry: {self._es_counter}")

        best_path = os.path.join(save_dir, f"{self.model_type}_best.pt")

        try:
            for epoch in range(self.start_epoch, self.num_epochs):
                if self.interrupted:
                    print("\nInterrupted. Saving state...")
                    # `epoch` here is 0-indexed and this iteration was
                    # interrupted before completing, so save bare `epoch`
                    # (not epoch+1) so the next session re-runs this epoch.
                    self._save_resume_checkpoint(save_dir, epoch, self._last_val_metrics)
                    print("Resume checkpoint saved. Exiting gracefully.")
                    break

                # FIX-2: set deterministic seed for this epoch's shuffle
                self._set_epoch_seed(epoch)

                print(f"\n{'=' * 55}")
                print(f"Epoch {epoch + 1} / {self.num_epochs}")
                print(f"{'=' * 55}")

                train_m = self.train_epoch()
                val_m   = self.validate()
                self._last_val_metrics = val_m  # keep for emergency resume saves

                # Record history
                self.history["train_loss"].append(train_m["loss"])
                self.history["train_accuracy"].append(train_m["accuracy"])
                self.history["train_f1"].append(train_m["f1"])
                self.history["val_loss"].append(val_m["loss"])
                self.history["val_accuracy"].append(val_m["accuracy"])
                self.history["val_precision"].append(val_m["precision"])
                self.history["val_recall"].append(val_m["recall"])
                self.history["val_f1"].append(val_m["f1"])

                self._log_metrics_to_csv(epoch + 1, train_m, val_m)

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

                # ── Model checkpointing + early stopping ─────────────────── #
                # _best.pt is saved whenever the chosen checkpoint_metric
                # hits a new high (val_f1) or new low (val_loss).
                # Early stopping patience is counted on the same metric.
                current_f1   = val_m["f1"]
                current_loss = val_m["loss"]

                # Always keep best_val_f1 and best_val_loss current for logging
                # regardless of which metric is used for checkpointing.
                improved = False
                if self.checkpoint_metric == "val_f1":
                    if current_f1 > self.best_val_f1:
                        self.best_val_f1   = current_f1
                        self.best_val_loss = current_loss   # update companion metric too
                        self.best_epoch    = epoch + 1
                        self._es_counter   = 0
                        improved = True
                else:  # val_loss
                    if current_loss < self.best_val_loss:
                        self.best_val_loss = current_loss
                        self.best_val_f1   = current_f1     # update companion metric too
                        self.best_epoch    = epoch + 1
                        self._es_counter   = 0
                        improved = True

                if improved:
                    self.save_model(best_path, epoch=epoch + 1, val_metrics=val_m)
                    print(
                        f"  ★ New best ({self.checkpoint_metric}): "
                        f"val_loss={self.best_val_loss:.4f}  "
                        f"val_f1={self.best_val_f1:.4f}  "
                        f"(epoch {self.best_epoch})"
                    )
                else:
                    self._es_counter += 1
                    metric_val = current_f1 if self.checkpoint_metric == "val_f1" else current_loss
                    best_val   = self.best_val_f1 if self.checkpoint_metric == "val_f1" else self.best_val_loss
                    direction  = "high" if self.checkpoint_metric == "val_f1" else "low"
                    print(
                        f"  No {self.checkpoint_metric} improvement "
                        f"(best={best_val:.4f} [{direction}], current={metric_val:.4f})"
                    )
                    if self.early_stopping_patience > 0:
                        print(
                            f"  Early stopping: {self._es_counter} / "
                            f"{self.early_stopping_patience}"
                        )

                # Save resume checkpoint periodically
                if (epoch + 1) % self.save_resume_every_n_epochs == 0:
                    self._save_resume_checkpoint(save_dir, epoch + 1, val_m)

                # ── Early stopping check ──────────────────────────────────── #
                # Placed AFTER the periodic resume save so the checkpoint always
                # reflects the latest state before the loop exits.
                if (
                    self.early_stopping_patience > 0
                    and self._es_counter >= self.early_stopping_patience
                ):
                    print(
                        f"\n  Early stopping triggered: {self.checkpoint_metric} has not improved "
                        f"for {self._es_counter} consecutive epochs."
                    )
                    print(f"  Best epoch was {self.best_epoch} "
                          f"(val_loss={self.best_val_loss:.4f}, val_f1={self.best_val_f1:.4f}).")
                    break

        except KeyboardInterrupt:
            print("\n\nKeyboardInterrupt caught. Saving state...")
            # `epoch` is the 0-indexed loop variable for the epoch that was
            # interrupted mid-run. Save it as-is so start_epoch on resume
            # re-runs this epoch from scratch instead of skipping it.
            # Periodic saves use `epoch + 1` (completed epochs); here the
            # epoch was NOT completed, so we use bare `epoch`.
            current_epoch = epoch if 'epoch' in locals() else self.start_epoch
            self._save_resume_checkpoint(save_dir, current_epoch, self._last_val_metrics)
            print("Resume checkpoint saved. You can restart training to continue.")
            raise

        print(f"\n{'=' * 55}")
        print(f"Training complete.")
        print(f"Checkpoint metric:  {self.checkpoint_metric}")
        print(f"Best val loss = {self.best_val_loss:.4f} at epoch {self.best_epoch}")
        print(f"Best val F1   = {self.best_val_f1:.4f}")
        print(f"Best checkpoint: {best_path}")
        print(f"Metrics log: {self.metrics_log_path}")

        # Reload best weights before returning.
        # FIX (BUG-3): guard with isfile() so that a resume where start_epoch >= num_epochs
        # (training already complete) does not crash with FileNotFoundError.
        if os.path.isfile(best_path):
            ckpt = torch.load(best_path, map_location=self.device)
            _safe_load(self.model, _extract_state_dict(ckpt))
            print(f"  Best weights reloaded from: {best_path}")
        else:
            print(
                f"  [warn] Best checkpoint not found at '{best_path}'. "
                "Returning current weights (training may have already been complete)."
            )
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
    # Set high enough that the model peaks before hitting the limit.
    # Early stopping will exit well before this ceiling if val_loss plateaus.
    "num_epochs":             15,
    # Stop training if the checkpoint metric has not improved for this many
    # consecutive epochs. Set to 0 to disable early stopping entirely.
    "early_stopping_patience": 3,
    # Which validation metric triggers saving _best.pt and early stopping.
    # 'val_f1'   — recommended for imbalanced multilingual classification
    # 'val_loss' — smoother signal; use if F1 is very noisy on your data
    "checkpoint_metric":       "val_f1",
    "warmup_ratio":     0.1,
    "max_length":       512,
    "dropout":          0.3,
    "freeze_layers":    6,
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
        "ensemble":    "ensemble",
    }
    tokenizer_name = TOKENIZER_MAP[model_type]

    # ── Data ──────────────────────────────────────────────────────────────── #
    print("Loading data...")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df   = pd.read_csv("data/processed/val.csv")

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
            model_name=CONFIG.get("muril_model_name", "google/muril-base-cased"),
            **common_kwargs,
        )
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
            freeze_base_models=True,
        )

    param_info = model.count_parameters()
    # count_parameters() key names differ by model type:
    #   EnsembleFakeNewsClassifier  → 'total_trainable', 'total_frozen'
    #   XLM-R / MuRIL               → 'trainable', 'frozen'
    t_key = "total_trainable" if "total_trainable" in param_info else "trainable"
    f_key = "total_frozen"    if "total_frozen"    in param_info else "frozen"
    print(
        f"  Trainable: {param_info[t_key]:,} | "
        f"Frozen: {param_info[f_key]:,} | "
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
        early_stopping_patience=CONFIG.get("early_stopping_patience", 0),
        checkpoint_metric=CONFIG.get("checkpoint_metric", "val_f1"),
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

    # Val-set predictions — reuse _unpack_batch and _forward for consistency
    trained_model.eval()
    y_true, y_pred, probs_list, languages = [], [], [], []

    with torch.no_grad():
        for batch in trainer.val_loader:
            labels = batch["label"]

            if model_type == "ensemble":
                muril_tti = batch.get("muril_tti")
                logits = trained_model(
                    batch["xlmr_ids"].to(device),
                    batch["xlmr_mask"].to(device),
                    batch["muril_ids"].to(device),
                    batch["muril_mask"].to(device),
                    muril_tti.to(device) if isinstance(muril_tti, torch.Tensor) else muril_tti,
                )
            elif model_type == "xlm-roberta":
                logits = trained_model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
            else:  # muril
                tti = batch.get("token_type_ids")
                if isinstance(tti, torch.Tensor):
                    tti = tti.to(device)
                logits = trained_model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    tti,
                )

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
