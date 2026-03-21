import os
import torch
import torch.nn as nn
from transformers import XLMRobertaModel, get_linear_schedule_with_warmup
from torch.optim import AdamW


class XLMRobertaFakeNewsClassifier(nn.Module):
    """
    XLM-RoBERTa based fake news classifier.
    Supports 100 languages with strong cross-lingual transfer.

    Key differences from BERT/MuRIL:
        - RoBERTa has NO token_type_ids (segment embeddings were removed
          during pretraining — passing them causes silent errors or is ignored)
        - pooler_output is unreliable: XLM-RoBERTa's pooler was not trained
          for classification; mean pooling is strongly preferred
        - Attention weights are returned per-layer as a tuple; the existing
          get_attention_weights() returned the raw tuple without any guidance
          on shape, which is easy to misuse

    Architecture:
        XLM-RoBERTa encoder → mean pooling → Dropout → Linear classifier

    Example:
        model = XLMRobertaFakeNewsClassifier(num_classes=2)
        logits = model(input_ids, attention_mask)
    """

    def __init__(
        self,
        model_name: str = 'xlm-roberta-base',
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_bert: bool = False,           # kept as-is for API compatibility
        freeze_layers: int = 0,
        use_pooler: bool = False,            # FIX: default False — mean pooling
                                             # outperforms the unreliable CLS pooler
        use_gradient_checkpointing: bool = False,
    ):
        """
        Args:
            model_name:                   HuggingFace model identifier.
                                          e.g. 'xlm-roberta-base' or
                                               'xlm-roberta-large'
            num_classes:                  Number of output classes (2 = binary).
            dropout:                      Dropout probability before classifier.
            freeze_bert:                  If True, freeze ALL encoder parameters;
                                          only the classifier head is trained.
            freeze_layers:                Freeze embeddings + first N encoder
                                          layers. Ignored when freeze_bert=True.
            use_pooler:                   If True, use the built-in pooler_output
                                          (Linear([CLS]) → Tanh).
                                          If False (recommended), use mean pooling
                                          over all non-padding tokens.
            use_gradient_checkpointing:   Trade compute for GPU memory savings.
        """
        super(XLMRobertaFakeNewsClassifier, self).__init__()

        # ------------------------------------------------------------------ #
        #  Load pretrained backbone                                           #
        # ------------------------------------------------------------------ #
        self.xlm_roberta = XLMRobertaModel.from_pretrained(model_name)
        self.config = self.xlm_roberta.config
        self.use_pooler = use_pooler
        self.num_classes = num_classes      # stored for ensemble compatibility checks

        if use_gradient_checkpointing:
            self.xlm_roberta.gradient_checkpointing_enable()

        # ------------------------------------------------------------------ #
        #  Parameter freezing                                                 #
        # ------------------------------------------------------------------ #
        if freeze_bert:
            for param in self.xlm_roberta.parameters():
                param.requires_grad = False

        elif freeze_layers > 0:
            # FIX: validate against actual model depth before slicing
            num_encoder_layers = len(self.xlm_roberta.encoder.layer)
            if freeze_layers > num_encoder_layers:
                raise ValueError(
                    f"freeze_layers={freeze_layers} exceeds the number of "
                    f"encoder layers ({num_encoder_layers}) in '{model_name}'."
                )

            for param in self.xlm_roberta.embeddings.parameters():
                param.requires_grad = False

            for layer in self.xlm_roberta.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # ------------------------------------------------------------------ #
        #  Classification head                                                #
        # ------------------------------------------------------------------ #
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        self._init_weights()

    # ---------------------------------------------------------------------- #
    #  Weight initialisation                                                  #
    # ---------------------------------------------------------------------- #

    def _init_weights(self) -> None:
        """Xavier-uniform init for the linear classifier."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    # ---------------------------------------------------------------------- #
    #  Pooling helper                                                         #
    # ---------------------------------------------------------------------- #

    def mean_pooling(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Average all non-padding token embeddings.

        Args:
            token_embeddings: [batch_size, seq_length, hidden_size]
            attention_mask:   [batch_size, seq_length]

        Returns:
            pooled: [batch_size, hidden_size]
        """
        mask_expanded = (
            attention_mask
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        sum_emb  = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_emb / sum_mask

    # ---------------------------------------------------------------------- #
    #  Forward pass                                                           #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        # FIX: token_type_ids intentionally NOT accepted.
        # XLM-RoBERTa was pretrained without segment embeddings (RoBERTa-style).
        # Passing token_type_ids is a no-op at best and misleading at worst.
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      Token IDs   [batch_size, seq_length]
            attention_mask: Padding mask [batch_size, seq_length]

        Returns:
            logits: Raw classification scores [batch_size, num_classes]

        Note:
            Unlike BERT/MuRIL, XLM-RoBERTa does NOT use token_type_ids.
            Do not pass segment IDs to this model.
        """
        outputs = self.xlm_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            # token_type_ids deliberately omitted — not supported by RoBERTa
        )

        if self.use_pooler:
            # CLS-based pooler (Linear → Tanh).
            # NOTE: was NOT trained during XLM-RoBERTa pretraining;
            # use_pooler=False (mean pooling) is recommended.
            pooled_output = outputs.pooler_output
        else:
            # Mean pool over all real tokens  ← recommended default
            pooled_output = self.mean_pooling(
                outputs.last_hidden_state,
                attention_mask,
            )

        logits = self.classifier(self.dropout(pooled_output))
        return logits

    # ---------------------------------------------------------------------- #
    #  Interpretability                                                       #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """
        Return per-layer attention weights for interpretability analysis.

        FIX: original returned raw outputs.attentions with no shape guidance.
        This method validates the output and documents the tensor shapes.

        Args:
            input_ids:      [batch_size, seq_length]
            attention_mask: [batch_size, seq_length]

        Returns:
            Tuple of length num_layers, each tensor:
                [batch_size, num_heads, seq_length, seq_length]

            Access example — last-layer, first head, first sample:
                weights = model.get_attention_weights(ids, mask)
                last_layer = weights[-1]          # [B, H, L, L]
                head_0     = last_layer[0, 0]     # [L, L]
        """
        was_training = self.training
        self.eval()

        outputs = self.xlm_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )

        if was_training:
            self.train()

        # outputs.attentions is a tuple of tensors, one per encoder layer
        # Shape per tensor: [batch_size, num_heads, seq_length, seq_length]
        return outputs.attentions

    # ---------------------------------------------------------------------- #
    #  Convenience: inference                                                 #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference with dropout disabled.
        Automatically handles eval/train mode switching.

        Args:
            input_ids, attention_mask: same as forward().

        Returns:
            probs: Softmax probabilities [batch_size, num_classes]
            preds: Predicted class indices [batch_size]
        """
        was_training = self.training
        self.eval()

        logits = self.forward(input_ids, attention_mask)
        probs  = torch.softmax(logits, dim=-1)
        preds  = torch.argmax(probs, dim=-1)

        if was_training:
            self.train()

        return probs, preds

    # ---------------------------------------------------------------------- #
    #  Convenience: parameter counts                                         #
    # ---------------------------------------------------------------------- #

    def count_parameters(self) -> dict[str, int]:
        """
        Returns:
            dict with keys 'trainable', 'frozen', 'total'.
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}

    # ---------------------------------------------------------------------- #
    #  Convenience: save / load                                               #
    # ---------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """Save model weights. Creates parent directories if needed."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"[XLM-R] Model saved → {path}")

    def load(self, path: str, map_location: str | torch.device = "cpu") -> None:
        """Load model weights from checkpoint."""
        self.load_state_dict(torch.load(path, map_location=map_location))
        print(f"[XLM-R] Weights loaded ← {path}")


# ========================================================================== #
#  Training utilities                                                         #
# ========================================================================== #

def build_optimizer_and_scheduler(
    model: XLMRobertaFakeNewsClassifier,
    num_training_steps: int,
    encoder_lr: float = 2e-5,
    classifier_lr: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
) -> tuple[AdamW, object]:
    """
    AdamW with differential learning rates + linear warmup/decay schedule.

    The pretrained encoder needs a much smaller LR than the randomly
    initialised classifier head.  Biases and LayerNorm parameters are
    excluded from weight decay as is standard practice.

    Args:
        model:               The classifier instance.
        num_training_steps:  Total optimiser steps (epochs × steps_per_epoch).
        encoder_lr:          LR for the XLM-RoBERTa encoder (default 2e-5).
        classifier_lr:       LR for the classification head (default 1e-4).
        weight_decay:        L2 regularisation (not applied to biases/norms).
        warmup_ratio:        Fraction of steps for linear LR warm-up.

    Returns:
        optimizer, scheduler

    Example:
        optimizer, scheduler = build_optimizer_and_scheduler(
            model, num_training_steps=1000
        )
        # In training loop:
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    """
    no_decay = {"bias", "LayerNorm.weight"}

    param_groups = [
        # Encoder — decayed params
        {
            "params": [
                p for n, p in model.xlm_roberta.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "lr": encoder_lr,
            "weight_decay": weight_decay,
        },
        # Encoder — non-decayed params
        {
            "params": [
                p for n, p in model.xlm_roberta.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "lr": encoder_lr,
            "weight_decay": 0.0,
        },
        # Classifier head — decayed params
        {
            "params": [
                p for n, p in model.classifier.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "lr": classifier_lr,
            "weight_decay": weight_decay,
        },
        # Classifier head — non-decayed params
        {
            "params": [
                p for n, p in model.classifier.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "lr": classifier_lr,
            "weight_decay": 0.0,
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
    model: XLMRobertaFakeNewsClassifier,
    dataloader: torch.utils.data.DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """
    Run one full training epoch.

    Args:
        model:          Classifier (set to train mode internally).
        dataloader:     Yields dicts with keys:
                            'input_ids', 'attention_mask', 'labels'
                        NOTE: do NOT include 'token_type_ids' for XLM-RoBERTa.
        optimizer:      AdamW instance from build_optimizer_and_scheduler.
        scheduler:      LR scheduler.
        device:         torch.device.
        max_grad_norm:  Gradient clipping threshold.

    Returns:
        dict with 'loss' and 'accuracy' for the epoch.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
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
    model: XLMRobertaFakeNewsClassifier,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate on a validation or test dataloader.

    Args:
        model:      Classifier (set to eval mode internally).
        dataloader: Same format as train_one_epoch.
        device:     torch.device.

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

        logits = model(input_ids, attention_mask)
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
    Minimal usage demo.  Run with:  python xlm_roberta_model.py
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Instantiate ──────────────────────────────────────────────────────── #
    model = XLMRobertaFakeNewsClassifier(
        num_classes=2,
        dropout=0.3,
        freeze_layers=4,
        use_pooler=False,                  # mean pooling (recommended)
        use_gradient_checkpointing=False,
    ).to(device)

    info = model.count_parameters()
    print(
        f"Parameters — trainable: {info['trainable']:,} | "
        f"frozen: {info['frozen']:,} | "
        f"total: {info['total']:,}"
    )

    # ── Forward pass ─────────────────────────────────────────────────────── #
    B, L = 4, 128
    ids  = torch.randint(0, 1000, (B, L)).to(device)
    mask = torch.ones(B, L, dtype=torch.long).to(device)
    # NOTE: no token_type_ids for XLM-RoBERTa

    logits = model(ids, mask)
    print(f"Logits shape: {logits.shape}")           # Expected: [4, 2]

    # ── Inference ────────────────────────────────────────────────────────── #
    probs, preds = model.predict(ids, mask)
    print(f"Predicted classes:  {preds.tolist()}")
    print(f"Confidence scores:  {probs.max(dim=-1).values.tolist()}")

    # ── Attention weights ────────────────────────────────────────────────── #
    attn = model.get_attention_weights(ids, mask)
    print(f"Num attention layers: {len(attn)}")
    print(f"Per-layer shape:      {attn[0].shape}")  # [B, num_heads, L, L]

    # ── Optimizer + scheduler ────────────────────────────────────────────── #
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        num_training_steps=3 * 100,   # 3 epochs × 100 steps
        encoder_lr=2e-5,
        classifier_lr=1e-4,
    )
    print("Optimizer and scheduler built successfully.")
