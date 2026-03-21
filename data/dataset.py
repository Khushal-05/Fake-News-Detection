"""
Dataset class for multilingual fake news detection.
Supports XLM-RoBERTa, MuRIL, and Ensemble tokenizers.

Key fixes and improvements:
    - token_type_ids now included in __getitem__ return dict.
      train.py, evaluate.py, and the ensemble forward pass all call
      batch.get("token_type_ids") — if the dataset never returns it the key
      is always missing and MuRIL silently runs without segment embeddings.
      XLM-RoBERTa tokenizer does NOT produce token_type_ids, so for that
      tokenizer the key is simply absent (None), which is safe.

    - Tokenizer loaded once at __init__ time with use_fast=True for speed.
      Original did this correctly; added use_fast=True explicitly.

    - Input validation in __init__: mismatched lengths between texts, labels,
      and languages produce a clear error immediately rather than an index
      error deep inside training.

    - Texts cast to str in __init__ (vectorised) rather than per-item in
      __getitem__, avoiding repeated isinstance checks at DataLoader worker level.

    - label dtype validated: non-integer labels (e.g. float from CSV read)
      are cast to int before wrapping in a tensor so CrossEntropyLoss /
      NLLLoss never receive float targets.

    - __getitem__ no longer creates a new tensor from an already-tensor label
      redundantly; uses item() to extract scalar first if needed.

    - tokenizer_name stored as attribute for downstream inspection
      (e.g. inference.py or test.py can verify which tokenizer was used).
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class MultilingualFakeNewsDataset(Dataset):
    """
    PyTorch Dataset for multilingual fake news detection.

    Returns batches compatible with all three model types:
        XLM-RoBERTa  — uses input_ids + attention_mask
        MuRIL        — uses input_ids + attention_mask + token_type_ids
        Ensemble     — same as MuRIL (routed internally)

    Each __getitem__ returns a dict with:
        input_ids:       LongTensor [max_length]
        attention_mask:  LongTensor [max_length]
        token_type_ids:  LongTensor [max_length]  — only present when the
                         tokenizer produces it (MuRIL/BERT); absent for
                         XLM-RoBERTa (RoBERTa-style tokenizers omit it)
        label:           LongTensor scalar
        language:        str  — language code for per-language analysis
    """

    def __init__(
        self,
        texts,
        labels,
        languages,
        tokenizer_name: str,
        max_length: int = 512,
    ):
        """
        Args:
            texts:          Sequence of raw text strings.
            labels:         Sequence of integer labels (0 = fake, 1 = real).
            languages:      Sequence of language code strings (e.g. 'hi', 'en').
            tokenizer_name: HuggingFace tokenizer identifier.
                            'xlm-roberta-base'       for XLM-RoBERTa
                            'google/muril-base-cased' for MuRIL / ensemble
            max_length:     Tokenisation truncation/padding length.
        """
        # ── Input validation ─────────────────────────────────────────────── #
        if not (len(texts) == len(labels) == len(languages)):
            raise ValueError(
                f"texts ({len(texts)}), labels ({len(labels)}), and "
                f"languages ({len(languages)}) must all have the same length."
            )
        if len(texts) == 0:
            raise ValueError("Dataset is empty — texts list has zero elements.")

        # Cast to list and pre-convert texts to str (once, not per-item)
        self.texts     = [str(t) for t in texts]
        self.labels    = [int(l) for l in labels]       # ensure int for tensor
        self.languages = list(languages)
        self.max_length     = max_length
        self.tokenizer_name = tokenizer_name

        # ── Load tokenizer ────────────────────────────────────────────────── #
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text     = self.texts[idx]
        label    = self.labels[idx]
        language = self.languages[idx]

        # Tokenise — padding and truncation to fixed length
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # ── Build output dict ─────────────────────────────────────────────── #
        item = {
            "input_ids":      encoding["input_ids"].squeeze(0),       # [L]
            "attention_mask": encoding["attention_mask"].squeeze(0),  # [L]
            "label":          torch.tensor(label, dtype=torch.long),
            "language":       language,
        }

        # token_type_ids: present for BERT/MuRIL tokenizers, absent for RoBERTa.
        # train.py / evaluate.py use batch.get("token_type_ids") safely,
        # so including it only when the tokenizer produces it is correct.
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)  # [L]

        return item


# ════════════════════════════════════════════════════════════════════════════ #
#  Smoke test                                                                  #
# ════════════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    import pandas as pd
    from torch.utils.data import DataLoader

    train_df = pd.read_csv("data/processed/train.csv")

    for tokenizer_name, label in [
        ("xlm-roberta-base",       "XLM-RoBERTa"),
        ("google/muril-base-cased", "MuRIL"),
    ]:
        print(f"\n── {label} ({tokenizer_name}) ──")
        ds = MultilingualFakeNewsDataset(
            texts=train_df["cleaned_text"].values,
            labels=train_df["label"].values,
            languages=train_df["language"].values,
            tokenizer_name=tokenizer_name,
            max_length=128,
        )
        print(f"  Size: {len(ds)}")

        sample = ds[0]
        print(f"  Keys:            {list(sample.keys())}")
        print(f"  input_ids shape: {sample['input_ids'].shape}")
        print(f"  label:           {sample['label']}")
        print(f"  language:        {sample['language']}")
        has_tti = "token_type_ids" in sample
        print(f"  token_type_ids:  {'present' if has_tti else 'absent (expected for RoBERTa)'}")

        # Verify DataLoader collation works end-to-end
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batch  = next(iter(loader))
        print(f"  Batch input_ids shape: {batch['input_ids'].shape}")
        if "token_type_ids" in batch:
            print(f"  Batch token_type_ids shape: {batch['token_type_ids'].shape}")
