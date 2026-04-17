import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


class EnsembleFakeNewsClassifier(nn.Module):
    """
    Ensemble model combining XLM-RoBERTa and MuRIL for fake news classification.

    XLM-RoBERTa: stronger cross-lingual transfer across 100 languages.
    MuRIL:       stronger on Indian languages and their transliterated forms.

    Three ensemble strategies:
        'weighted_avg' — static weighted sum of softmax probabilities
                         (operating on probabilities, not raw logits — see FIX notes)
        'max'          — element-wise max of softmax probabilities per class
        'learned'      — trainable FC layer fuses both models' logits

    IMPORTANT — forward() signatures differ between models:
        XLM-RoBERTa:  forward(input_ids, attention_mask)
                          NO token_type_ids (RoBERTa-style pretraining)
        MuRIL:        forward(input_ids, attention_mask, token_type_ids=None)
                          token_type_ids optional but supported (BERT-style)

    Example:
        from xlm_roberta_model import XLMRobertaFakeNewsClassifier
        from muril_model import MuRILFakeNewsClassifier

        xlmr  = XLMRobertaFakeNewsClassifier(num_classes=2)
        muril = MuRILFakeNewsClassifier(num_classes=2)

        ensemble = EnsembleFakeNewsClassifier(
            xlmr_model=xlmr,
            muril_model=muril,
            num_classes=2,
            ensemble_method='weighted_avg',
            weights=[0.4, 0.6],   # favour MuRIL for Indian-language data
        )

        logits = ensemble(input_ids, attention_mask, token_type_ids)
    """

    VALID_METHODS = {'weighted_avg', 'max', 'learned'}

    def __init__(
        self,
        xlmr_model: nn.Module,
        muril_model: nn.Module,
        num_classes: int = 2,
        ensemble_method: str = 'weighted_avg',
        weights: list[float] | None = None,
        freeze_base_models: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        """
        Args:
            xlmr_model:               Pretrained XLMRobertaFakeNewsClassifier.
            muril_model:              Pretrained MuRILFakeNewsClassifier.
            num_classes:              Number of output classes. Must match both
                                      sub-models. (default: 2)
            ensemble_method:          One of 'weighted_avg', 'max', 'learned'.
            weights:                  [xlmr_weight, muril_weight] for
                                      'weighted_avg'. Must sum to 1.0.
                                      Defaults to [0.5, 0.5].
            freeze_base_models:       If True, freeze both sub-model parameters
                                      so only the ensemble FC layer (if 'learned')
                                      is trained. Useful for inference-only or
                                      when the base models are already fine-tuned.
            use_gradient_checkpointing: Enable gradient checkpointing on both
                                      sub-models to save GPU memory.
        """
        super(EnsembleFakeNewsClassifier, self).__init__()

        # ------------------------------------------------------------------ #
        #  Validate inputs                                                    #
        # ------------------------------------------------------------------ #
        if ensemble_method not in self.VALID_METHODS:
            raise ValueError(
                f"ensemble_method='{ensemble_method}' is not valid. "
                f"Choose from: {self.VALID_METHODS}"
            )

        self.ensemble_method = ensemble_method
        self.num_classes = num_classes

        # Cross-check num_classes matches both sub-models if they expose it
        for name, m in [("xlmr_model", xlmr_model), ("muril_model", muril_model)]:
            sub_nc = getattr(m, "num_classes", None)
            if sub_nc is not None and sub_nc != num_classes:
                raise ValueError(
                    f"{name}.num_classes={sub_nc} does not match "
                    f"ensemble num_classes={num_classes}. "
                    f"All models must output the same number of classes."
                )

        # ------------------------------------------------------------------ #
        #  Sub-models                                                         #
        # ------------------------------------------------------------------ #
        self.xlmr_model  = xlmr_model
        self.muril_model = muril_model

        if use_gradient_checkpointing:
            self.xlmr_model.xlm_roberta.gradient_checkpointing_enable()
            self.muril_model.muril.gradient_checkpointing_enable()

        if freeze_base_models:
            for param in self.xlmr_model.parameters():
                param.requires_grad = False
            for param in self.muril_model.parameters():
                param.requires_grad = False

        # ------------------------------------------------------------------ #
        #  Ensemble-method–specific setup                                     #
        # ------------------------------------------------------------------ #
        if ensemble_method == 'weighted_avg':
            if weights is None:
                weights = [0.5, 0.5]

            weights = list(weights)

            # FIX: validate weights sum to ~1.0
            if len(weights) != 2:
                raise ValueError(
                    f"weights must have exactly 2 elements "
                    f"[xlmr_weight, muril_weight], got {len(weights)}."
                )
            if abs(sum(weights) - 1.0) > 1e-4:
                raise ValueError(
                    f"weights must sum to 1.0, got sum={sum(weights):.4f}. "
                    f"Normalise your weights before passing them in."
                )

            # FIX: register as a buffer so it moves with .to(device) automatically
            # and is included in state_dict() for checkpointing.
            # Original used a plain tensor attribute which did NOT move with the
            # model and was NOT saved in checkpoints.
            self.register_buffer(
                'weights',
                torch.tensor(weights, dtype=torch.float32)
            )

        elif ensemble_method == 'learned':
            # FIX: use num_classes * 2 as input size so it works for any
            # num_classes, not just hardcoded 4 (which broke for num_classes != 2)
            self.ensemble_fc = nn.Linear(num_classes * 2, num_classes)
            nn.init.xavier_uniform_(self.ensemble_fc.weight)
            nn.init.zeros_(self.ensemble_fc.bias)

    # ---------------------------------------------------------------------- #
    #  Forward pass                                                           #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        xlmr_ids: torch.Tensor,
        xlmr_mask: torch.Tensor,
        muril_ids: torch.Tensor,
        muril_mask: torch.Tensor,
        muril_tti: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:       Token IDs              [batch_size, seq_length]
            attention_mask:  Padding mask            [batch_size, seq_length]
            token_type_ids:  Segment IDs (optional)  [batch_size, seq_length]
                             Forwarded to MuRIL only. XLM-RoBERTa does not
                             use segment embeddings and ignores this argument.

        Returns:
            logits: Ensembled classification scores [batch_size, num_classes]

            Output space by method:
                'weighted_avg' → log-probabilities  (use NLLLoss for training)
                'max'          → log-probabilities  (use NLLLoss for training)
                'learned'      → raw logits         (use CrossEntropyLoss for training)
            The training utilities in this file handle the correct loss automatically.
        """
        # -- Sub-model forward passes -------------------------------------- #
        # XLM-RoBERTa: no token_type_ids
        xlmr_logits  = self.xlmr_model(xlmr_ids, xlmr_mask)

        # MuRIL: token_type_ids optional
        muril_logits = self.muril_model(muril_ids, muril_mask, muril_tti)

        # -- Ensemble ------------------------------------------------------ #
        if self.ensemble_method == 'weighted_avg':
            # FIX: average over probabilities, not raw logits.
            # Averaging raw logits treats both models' scales as equivalent,
            # but logit scales vary between architectures/training runs.
            # Softmax-normalising first makes the weighted average meaningful.
            xlmr_probs  = F.softmax(xlmr_logits,  dim=-1)   # [B, C]
            muril_probs = F.softmax(muril_logits, dim=-1)    # [B, C]

            # weights is a registered buffer: [xlmr_w, muril_w]
            combined_probs = (
                self.weights[0] * xlmr_probs +
                self.weights[1] * muril_probs
            )                                                 # [B, C]

            # Convert back to logit space so the output is consistent with
            # cross-entropy loss expectations (log of probabilities)
            logits = torch.log(combined_probs.clamp(min=1e-9))

        elif self.ensemble_method == 'max':
            # FIX: torch.max(tensor_a, tensor_b) is element-wise max and
            # returns a tensor directly (not a named tuple) — this is correct.
            # But again we should operate on probabilities, not raw logits,
            # for the same scale-consistency reasons as above.
            xlmr_probs  = F.softmax(xlmr_logits,  dim=-1)
            muril_probs = F.softmax(muril_logits, dim=-1)

            combined_probs = torch.max(xlmr_probs, muril_probs)  # [B, C]
            logits = torch.log(combined_probs.clamp(min=1e-9))

        elif self.ensemble_method == 'learned':
            # Concatenate raw logits — the FC layer learns its own scaling
            combined = torch.cat([xlmr_logits, muril_logits], dim=-1)  # [B, 2C]
            logits   = self.ensemble_fc(combined)                       # [B, C]
        
        return logits

    # ---------------------------------------------------------------------- #
    #  Convenience: inference                                                 #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def predict(
        self,
        xlmr_ids: torch.Tensor,
        xlmr_mask: torch.Tensor,
        muril_ids: torch.Tensor,
        muril_mask: torch.Tensor,
        muril_tti: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference with dropout disabled across all sub-models.
        
        This method must match the forward() signature for ensemble models.
        It takes separate tokenizations for XLM-RoBERTa and MuRIL vocabularies.

        Args:
            xlmr_ids:    XLM-RoBERTa input token IDs [batch_size, seq_length]
            xlmr_mask:   XLM-RoBERTa attention mask [batch_size, seq_length]
            muril_ids:   MuRIL input token IDs [batch_size, seq_length]
            muril_mask:  MuRIL attention mask [batch_size, seq_length]
            muril_tti:   MuRIL token type IDs [batch_size, seq_length], optional

        Returns:
            probs: Softmax probabilities [batch_size, num_classes]
            preds: Predicted class indices [batch_size]
        """
        was_training = self.training
        self.eval()

        # Forward pass with ensemble inputs (5 tensors)
        logits = self.forward(xlmr_ids, xlmr_mask, muril_ids, muril_mask, muril_tti)

        # Convert logits to probabilities based on ensemble method:
        #   weighted_avg / max → log-probabilities: use exp() to get probs
        #   learned            → raw logits: use softmax to get probs
        if self.ensemble_method == 'learned':
            probs = F.softmax(logits, dim=-1)
        else:
            probs = torch.exp(logits)          # inverse of log()

        preds = torch.argmax(probs, dim=-1)

        if was_training:
            self.train()

        return probs, preds

    # ---------------------------------------------------------------------- #
    #  Convenience: parameter counts                                          #
    # ---------------------------------------------------------------------- #

    def count_parameters(self) -> dict[str, int]:
        """
        Returns trainable/frozen/total parameter counts broken down by
        sub-model for easy debugging of freeze configs.
        """
        def _count(module):
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            frozen    = sum(p.numel() for p in module.parameters() if not p.requires_grad)
            return trainable, frozen

        xlmr_t,  xlmr_f  = _count(self.xlmr_model)
        muril_t, muril_f = _count(self.muril_model)

        ensemble_params = {}
        ensemble_fc_t = 0
        if self.ensemble_method == 'learned':
            ensemble_fc_t, ensemble_fc_f = _count(self.ensemble_fc)
            ensemble_params = {
                "ensemble_fc_trainable": ensemble_fc_t,
                "ensemble_fc_frozen":    ensemble_fc_f,
            }

        # FIX: was double-counting ensemble_fc by iterating parameters() inline
        total_t = xlmr_t  + muril_t  + ensemble_fc_t
        total_f = xlmr_f  + muril_f

        return {
            "xlmr_trainable":  xlmr_t,
            "xlmr_frozen":     xlmr_f,
            "muril_trainable": muril_t,
            "muril_frozen":    muril_f,
            **ensemble_params,
            "total_trainable": total_t,
            "total_frozen":    total_f,
            "total":           total_t + total_f,
        }

    # ---------------------------------------------------------------------- #
    #  Convenience: save / load                                               #
    # ---------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """Save full ensemble state dict (includes sub-models + buffers)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"[Ensemble] Model saved → {path}")

    def load(self, path: str, map_location: str | torch.device = "cpu") -> None:
        """Load ensemble state dict from checkpoint."""
        self.load_state_dict(torch.load(path, map_location=map_location))
        print(f"[Ensemble] Weights loaded ← {path}")


# ========================================================================== #
#  Training utilities                                                         #
# ========================================================================== #

def build_optimizer_and_scheduler(
    model: EnsembleFakeNewsClassifier,
    num_training_steps: int,
    encoder_lr: float = 2e-5,
    classifier_lr: float = 1e-4,
    ensemble_lr: float = 1e-3,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
) -> tuple[AdamW, object]:
    """
    Three-tier differential learning rates:
        - encoder_lr   for the transformer backbone layers (lowest)
        - classifier_lr for each model's classification head
        - ensemble_lr   for the learned FC layer, if used (highest)

    Args:
        model:               EnsembleFakeNewsClassifier instance.
        num_training_steps:  epochs × steps_per_epoch.
        encoder_lr:          LR for backbone encoder layers (default 2e-5).
        classifier_lr:       LR for per-model classifiers (default 1e-4).
        ensemble_lr:         LR for ensemble FC layer (default 1e-3).
        weight_decay:        L2 penalty, skipped for biases/LayerNorm.
        warmup_ratio:        Fraction of steps used for warm-up.

    Returns:
        optimizer, scheduler
    """
    no_decay = {"bias", "LayerNorm.weight"}

    def _param_groups(backbone, head, backbone_lr, head_lr):
        """Helper to produce decayed/non-decayed groups for one sub-model."""
        return [
            {
                "params": [
                    p for n, p in backbone.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "lr": backbone_lr, "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in backbone.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "lr": backbone_lr, "weight_decay": 0.0,
            },
            {
                "params": [
                    p for n, p in head.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "lr": head_lr, "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in head.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "lr": head_lr, "weight_decay": 0.0,
            },
        ]

    param_groups = (
        _param_groups(
            model.xlmr_model.xlm_roberta,
            model.xlmr_model.classifier,
            encoder_lr, classifier_lr,
        ) +
        _param_groups(
            model.muril_model.muril,
            model.muril_model.classifier,
            encoder_lr, classifier_lr,
        )
    )

    # Add ensemble FC layer params if using 'learned' method
    if model.ensemble_method == 'learned':
        param_groups += [
            {
                "params": [
                    p for n, p in model.ensemble_fc.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "lr": ensemble_lr, "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in model.ensemble_fc.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "lr": ensemble_lr, "weight_decay": 0.0,
            },
        ]

    optimizer = AdamW(param_groups)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * warmup_ratio),
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


def train_one_epoch(
    model: EnsembleFakeNewsClassifier,
    dataloader: torch.utils.data.DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """
    Run one full training epoch over the ensemble.

    Args:
        model:          EnsembleFakeNewsClassifier (set to train mode internally).
        dataloader:     Yields dicts with keys:
                            'input_ids', 'attention_mask', 'labels'
                        and optionally 'token_type_ids' (used by MuRIL only).
        optimizer:      From build_optimizer_and_scheduler().
        scheduler:      LR scheduler.
        device:         torch.device.
        max_grad_norm:  Gradient clipping threshold.

    Returns:
        dict with 'loss' and 'accuracy'.
    """
    model.train()

    # FIX: loss function must match the output space of forward():
    #   weighted_avg / max  → log-probabilities → NLLLoss
    #   learned             → raw logits        → CrossEntropyLoss
    if model.ensemble_method == 'learned':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.NLLLoss()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        preds          = torch.argmax(logits, dim=-1)
        total_loss    += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return {
        "loss":     total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: EnsembleFakeNewsClassifier,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate the ensemble on a validation or test dataloader.

    Args:
        model:      EnsembleFakeNewsClassifier (set to eval mode internally).
        dataloader: Same format as train_one_epoch.
        device:     torch.device.

    Returns:
        dict with 'loss' and 'accuracy'.
    """
    model.eval()

    # FIX: match loss to output space (same logic as train_one_epoch)
    if model.ensemble_method == 'learned':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.NLLLoss()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss   = criterion(logits, labels)

        preds          = torch.argmax(logits, dim=-1)
        total_loss    += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return {
        "loss":     total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


# ========================================================================== #
#  Quick-start smoke test                                                     #
# ========================================================================== #

if __name__ == "__main__":
    """
    Minimal usage demo.  Run with:  python ensemble_model.py

    In practice, pass already fine-tuned sub-models here.
    Dummy un-pretrained models are used below only to test shapes/logic.
    """
    from unittest.mock import MagicMock  # noqa: F401 — kept for user extension

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Build dummy sub-models for smoke testing ─────────────────────────── #
    # Replace with your actual fine-tuned XLMRobertaFakeNewsClassifier /
    # MuRILFakeNewsClassifier instances in real use.

    class _DummyModel(nn.Module):
        """Minimal stand-in for smoke tests without downloading weights."""
        def __init__(self, num_classes=2, use_token_type=False):
            super().__init__()
            self.use_token_type = use_token_type
            self.fc = nn.Linear(128, num_classes)
        def forward(self, input_ids, attention_mask, token_type_ids=None):
            return self.fc(input_ids.float()[:, :128])

    xlmr_dummy  = _DummyModel(num_classes=2, use_token_type=False).to(device)
    muril_dummy = _DummyModel(num_classes=2, use_token_type=True).to(device)

    for method in ['weighted_avg', 'max', 'learned']:
        print(f"\n── ensemble_method='{method}' ──")

        ensemble = EnsembleFakeNewsClassifier(
            xlmr_model=xlmr_dummy,
            muril_model=muril_dummy,
            num_classes=2,
            ensemble_method=method,
            weights=[0.4, 0.6] if method == 'weighted_avg' else None,
        ).to(device)

        B, L = 4, 128
        ids   = torch.randint(0, 100, (B, L)).to(device)
        mask  = torch.ones(B, L, dtype=torch.long).to(device)
        types = torch.zeros(B, L, dtype=torch.long).to(device)

        logits = ensemble(ids, mask, types)
        print(f"  Logits shape:  {logits.shape}")          # [4, 2]

        probs, preds = ensemble.predict(ids, mask, types)
        print(f"  Predictions:   {preds.tolist()}")
        print(f"  Confidences:   {probs.max(dim=-1).values.tolist()}")

        info = ensemble.count_parameters()
        print(f"  Trainable params: {info['total_trainable']:,}")

    # ── Weights-sum validation ────────────────────────────────────────────── #
    print("\n── Testing weight validation ──")
    try:
        EnsembleFakeNewsClassifier(
            xlmr_model=xlmr_dummy,
            muril_model=muril_dummy,
            ensemble_method='weighted_avg',
            weights=[0.6, 0.6],            # bad: sums to 1.2
        )
    except ValueError as e:
        print(f"  Caught expected error: {e}")
