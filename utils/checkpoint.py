"""
Shared checkpoint utilities used by train.py, evaluate.py, inference.py,
eval_and_vis.py, and test.py.

Centralising these three helpers means a single fix propagates everywhere
rather than requiring four separate edits.

These functions are intentionally free of module-level torch imports so that
this file can be imported even in environments where torch is not yet loaded.
The actual torch calls happen inside function bodies, which is fine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn   # only imported for type checkers, not at runtime

__all__ = ["extract_state_dict", "strip_module_prefix", "safe_load"]


def extract_state_dict(ckpt: Any) -> dict:
    """Pull model weights out of any checkpoint format.

    Handles:
        - Plain state dict (weights only)
        - Dict with 'model_state_dict', 'state_dict', or 'model' key
        - Non-dict object (e.g. raw nn.Module) — returned as-is

    Args:
        ckpt: Raw checkpoint object returned by torch.load().

    Returns:
        A plain state dict suitable for model.load_state_dict().
    """
    if not isinstance(ckpt, dict):
        return ckpt
    for key in ("model_state_dict", "state_dict", "model"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt   # assume it is already a plain state dict


def strip_module_prefix(state_dict: dict) -> dict:
    """Remove the 'module.' prefix added by DistributedDataParallel.

    If a model was saved while wrapped in nn.DataParallel or
    DistributedDataParallel, every key will start with 'module.'.
    This function strips that prefix so the state dict can be loaded
    into an unwrapped model.

    Args:
        state_dict: A plain state dict (from extract_state_dict).

    Returns:
        The same dict with 'module.' prefixes removed, or the original
        dict unchanged if no keys start with 'module.'.
    """
    if any(k.startswith("module.") for k in state_dict):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def safe_load(model: "nn.Module", state_dict: dict, verbose: bool = True) -> None:
    """Load a state dict into a model, falling back to strict=False on mismatch.

    Attempts strict loading first (all keys must match exactly). If that
    fails with a RuntimeError (e.g. missing or unexpected keys due to a
    version mismatch), retries with strict=False so partial weights are
    still loaded. A warning is printed when the fallback is used.

    Args:
        model:      The nn.Module to load weights into.
        state_dict: A plain state dict (pass through extract_state_dict first).
        verbose:    If True, print load status messages.
    """
    state_dict = strip_module_prefix(state_dict)
    try:
        model.load_state_dict(state_dict)
        if verbose:
            print("  Weights loaded (strict=True).")
    except RuntimeError as exc:
        if verbose:
            print(f"  Strict load failed ({exc}). Retrying with strict=False.")
        model.load_state_dict(state_dict, strict=False)
