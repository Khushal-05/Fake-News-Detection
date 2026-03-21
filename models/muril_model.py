import os
import torch
import torch.nn as nn
from transformers import BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW


class MuRILFakeNewsClassifier(nn.Module):
    """
    MuRIL (Multilingual Representations for Indian Languages) based classifier.
    Optimized for 17 Indian languages and their transliterated forms.

    Architecture:
        - Pretrained MuRIL (BERT) encoder
        - Optional mean pooling or CLS pooler output
        - Dropout + Linear classification head

    Example usage:
        model = MuRILFakeNewsClassifier(num_classes=2)
        logits = model(input_ids, attention_mask, token_type_ids)
    """

    def __init__(
        self,
        model_name: str = 'google/muril-base-cased',
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_bert: bool = False,
        freeze_layers: int = 0,
        use_pooler: bool = False,           # FIX: default changed to False (mean pooling
                                            # outperforms tanh-gated CLS pooler on
                                            # downstream classification tasks)
        use_gradient_checkpointing: bool = False,
    ):
        """
        Args:
            model_name:                   HuggingFace model identifier for MuRIL.
            num_classes:                  Number of output classes (2 for binary).
            dropout:                      Dropout probability before the classifier.
            freeze_bert:                  If True, freeze ALL MuRIL parameters
                                          (only classifier head is trained).
            freeze_layers:                Freeze embeddings + first N encoder layers.
                                          Ignored when freeze_bert=True.
            use_pooler:                   If True, use BERT's built-in pooler_output
                                          ([CLS] → Linear → Tanh).
                                          If False (recommended), use mean pooling
                                          over all non-padding tokens.
            use_gradient_checkpointing:   Trade compute for memory; useful when
                                          fine-tuning on limited GPU RAM.
        """
        super(MuRILFakeNewsClassifier, self).__init__()

        # ------------------------------------------------------------------ #
        #  Load pretrained MuRIL backbone                                     #
        # ------------------------------------------------------------------ #
        self.muril = BertModel.from_pretrained(model_name)
        self.config = self.muril.config
        self.use_pooler = use_pooler
        self.num_classes = num_classes      # stored for ensemble compatibility checks

        # Optional gradient checkpointing to reduce GPU memory usage
        if use_gradient_checkpointing:
            self.muril.gradient_checkpointing_enable()

        # ------------------------------------------------------------------ #
        #  Parameter freezing                                                 #
        # ------------------------------------------------------------------ #
        if freeze_bert:
            # Freeze every MuRIL parameter; only the classifier head trains
            for param in self.muril.parameters():
                param.requires_grad = False

        elif freeze_layers > 0:
            # Validate freeze_layers against the actual model depth
            num_encoder_layers = len(self.muril.encoder.layer)
            if freeze_layers > num_encoder_layers:
                raise ValueError(
                    f"freeze_layers={freeze_layers} exceeds the number of "
                    f"encoder layers ({num_encoder_layers}) in {model_name}."
                )

            # Freeze token/position/segment embeddings
            for param in self.muril.embeddings.parameters():
                param.requires_grad = False

            # Freeze the first `freeze_layers` transformer blocks
            for layer in self.muril.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # ------------------------------------------------------------------ #
        #  Classification head                                                #
        # ------------------------------------------------------------------ #
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

        # Initialise classifier weights
        self._init_weights()

    # ---------------------------------------------------------------------- #
    #  Weight initialisation                                                  #
    # ---------------------------------------------------------------------- #

    def _init_weights(self) -> None:
        """Xavier-uniform init for the classifier linear layer."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    # ---------------------------------------------------------------------- #
    #  Pooling helpers                                                        #
    # ---------------------------------------------------------------------- #

    def mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the mean of all non-padding token embeddings.

        Args:
            token_embeddings: [batch_size, seq_length, hidden_size]
            attention_mask:   [batch_size, seq_length]  (1 = real, 0 = padding)

        Returns:
            pooled: [batch_size, hidden_size]
        """
        # Expand mask to match embedding dimensions
        input_mask_expanded = (
            attention_mask
            .unsqueeze(-1)                          # [B, L, 1]
            .expand(token_embeddings.size())        # [B, L, H]
            .float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        # Clamp prevents division-by-zero on fully-masked (empty) sequences
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask             # [B, H]

    # ---------------------------------------------------------------------- #
    #  Forward pass                                                           #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,   # FIX: added; needed for
                                                       # sentence-pair tasks and
                                                       # correct segment embeddings
    ) -> torch.Tensor:
        """
        Args:
            input_ids:       Token IDs              [batch_size, seq_length]
            attention_mask:  Padding mask            [batch_size, seq_length]
            token_type_ids:  Segment IDs (optional)  [batch_size, seq_length]
                             Pass None for single-sentence inputs; the model
                             defaults to all-zeros internally.

        Returns:
            logits: Raw classification scores [batch_size, num_classes]
        """
        # Run MuRIL encoder
        outputs = self.muril(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,      # FIX: forwarded to backbone
            return_dict=True,
        )

        # Select pooling strategy
        if self.use_pooler:
            # Built-in BERT pooler: Linear(CLS) → Tanh
            # NOTE: this was NOT trained during MuRIL pretraining and may
            # underperform for classification; use_pooler=False is recommended
            pooled_output = outputs.pooler_output
        else:
            # Mean pool over all real (non-padding) tokens  ← recommended
            pooled_output = self.mean_pooling(
                outputs.last_hidden_state,
                attention_mask,
            )

        # Dropout → classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)         # [B, num_classes]

        return logits

    # ---------------------------------------------------------------------- #
    #  Convenience: inference                                                 #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference without gradient tracking.
        Automatically sets eval mode and restores the previous training state.

        Args:
            input_ids, attention_mask, token_type_ids: same as forward().

        Returns:
            probs:   Softmax probabilities [batch_size, num_classes]
            preds:   Predicted class indices [batch_size]
        """
        was_training = self.training
        self.eval()                                     # FIX: ensure dropout is off

        logits = self.forward(input_ids, attention_mask, token_type_ids)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        if was_training:
            self.train()                                # restore original mode

        return probs, preds

    # ---------------------------------------------------------------------- #
    #  Convenience: trainable parameter count                                 #
    # ---------------------------------------------------------------------- #

    def count_parameters(self) -> dict[str, int]:
        """
        Return counts of trainable and frozen parameters.

        Returns:
            dict with keys 'trainable', 'frozen', 'total'
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}

    # ---------------------------------------------------------------------- #
    #  Convenience: save / load                                               #
    # ---------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """Save model weights to `path` (creates parent dirs if needed)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"[MuRIL] Model saved → {path}")

    def load(self, path: str, map_location: str | torch.device = "cpu") -> None:
        """Load model weights from `path`."""
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state)
        print(f"[MuRIL] Weights loaded ← {path}")


# ========================================================================== #
#  Training utilities                                                         #
# ========================================================================== #

def build_optimizer_and_scheduler(
    model: MuRILFakeNewsClassifier,
    num_training_steps: int,
    encoder_lr: float = 2e-5,           # renamed from bert_lr for cross-file consistency
    classifier_lr: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
) -> tuple[AdamW, object]:
    """
    Build an AdamW optimiser with differential learning rates and a linear
    warmup + decay schedule.

    Differential LRs are important: the pretrained MuRIL encoder needs a
    much smaller LR than the randomly-initialised classifier head.

    Args:
        model:               The MuRILFakeNewsClassifier instance.
        num_training_steps:  Total optimiser steps (epochs × steps_per_epoch).
        encoder_lr:          Learning rate for MuRIL encoder layers (default 2e-5).
        classifier_lr:       Learning rate for the classification head (default 1e-4).
        weight_decay:        L2 penalty (applied to non-bias/norm parameters).
        warmup_ratio:        Fraction of steps used for linear LR warm-up.

    Returns:
        optimizer:  Configured AdamW instance.
        scheduler:  Linear warmup + decay scheduler.

    Example:
        optimizer, scheduler = build_optimizer_and_scheduler(
            model, num_training_steps=1000
        )
        # Inside training loop:
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    """
    # Separate params that should NOT have weight decay (biases, LayerNorm)
    no_decay = {"bias", "LayerNorm.weight"}

    optimizer_grouped_parameters = [
        # MuRIL encoder — low LR, with weight decay
        {
            "params": [
                p for n, p in model.muril.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "lr": encoder_lr,
            "weight_decay": weight_decay,
        },
        # MuRIL encoder — low LR, NO weight decay
        {
            "params": [
                p for n, p in model.muril.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "lr": encoder_lr,
            "weight_decay": 0.0,
        },
        # Classifier head — high LR, with weight decay
        {
            "params": [
                p for n, p in model.classifier.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "lr": classifier_lr,
            "weight_decay": weight_decay,
        },
        # Classifier head — high LR, NO weight decay
        {
            "params": [
                p for n, p in model.classifier.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "lr": classifier_lr,
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters)

    num_warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return optimizer, scheduler


def train_one_epoch(
    model: MuRILFakeNewsClassifier,
    dataloader: torch.utils.data.DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """
    Run one full training epoch.

    Args:
        model:          The classifier (will be set to train mode).
        dataloader:     Yields dicts with keys:
                            'input_ids', 'attention_mask', 'labels'
                        and optionally 'token_type_ids'.
        optimizer:      The AdamW optimiser.
        scheduler:      The LR scheduler.
        device:         torch.device to run on.
        max_grad_norm:  Gradient clipping threshold (prevents exploding grads).

    Returns:
        dict with 'loss' and 'accuracy' for the epoch.

    Example dataloader batch format:
        {
            'input_ids':      torch.LongTensor [B, L],
            'attention_mask': torch.LongTensor [B, L],
            'token_type_ids': torch.LongTensor [B, L],   # optional
            'labels':         torch.LongTensor [B],
        }
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        # Forward
        logits = model(input_ids, attention_mask, token_type_ids)
        loss   = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        # Metrics
        preds = torch.argmax(logits, dim=-1)
        total_loss    += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return {
        "loss":     total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


@torch.no_grad()
def evaluate(
    model: MuRILFakeNewsClassifier,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate the model on a validation or test dataloader.

    Args:
        model:      The classifier (will be set to eval mode).
        dataloader: Same format as train_one_epoch.
        device:     torch.device to run on.

    Returns:
        dict with 'loss' and 'accuracy'.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

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

        preds = torch.argmax(logits, dim=-1)
        total_loss    += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return {
        "loss":     total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


# ========================================================================== #
#  Quick-start example                                                        #
# ========================================================================== #

if __name__ == "__main__":
    """
    Minimal smoke-test / usage demo.
    Run with:  python muril_model.py
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Instantiate model ────────────────────────────────────────────────── #
    model = MuRILFakeNewsClassifier(
        num_classes=2,
        dropout=0.3,
        freeze_layers=4,                  # freeze embeddings + first 4 layers
        use_pooler=False,                 # mean pooling (recommended)
        use_gradient_checkpointing=False, # set True for large batches on small GPU
    ).to(device)

    param_info = model.count_parameters()
    print(
        f"Parameters — trainable: {param_info['trainable']:,} | "
        f"frozen: {param_info['frozen']:,} | "
        f"total: {param_info['total']:,}"
    )

    # ── Dummy forward pass ───────────────────────────────────────────────── #
    batch_size, seq_len = 4, 128
    dummy_ids   = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    dummy_mask  = torch.ones(batch_size, seq_len, dtype=torch.long).to(device)
    dummy_types = torch.zeros(batch_size, seq_len, dtype=torch.long).to(device)

    logits = model(dummy_ids, dummy_mask, dummy_types)
    print(f"Logits shape: {logits.shape}")  # Expected: [4, 2]

    # ── Inference convenience method ─────────────────────────────────────── #
    probs, preds = model.predict(dummy_ids, dummy_mask, dummy_types)
    print(f"Predicted classes: {preds.tolist()}")
    print(f"Confidence scores: {probs.max(dim=-1).values.tolist()}")

    # ── Optimizer + scheduler (example for 3 epochs, 100 steps/epoch) ───── #
    num_training_steps = 3 * 100
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        num_training_steps=num_training_steps,
        encoder_lr=2e-5,
        classifier_lr=1e-4,
    )
    print("Optimizer and scheduler built successfully.")
