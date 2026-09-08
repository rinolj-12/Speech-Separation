"""
Compatibility shim — simplified for torch>=2.11.

torch.amp.autocast and torch.amp.GradScaler are always available at the
project's minimum PyTorch version (2.11). train.py uses them directly.
This module is kept only for backward-compatibility with external scripts.
"""
import torch
from typing import Optional


def amp_autocast(device_type: str = "cuda", dtype: Optional[torch.dtype] = None, enabled: bool = True):
    """Thin wrapper around torch.amp.autocast (torch>=2.11 required)."""
    if dtype is None:
        dtype = torch.float16
    return torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=enabled)


def make_grad_scaler(device_type: str = "cuda", enabled: bool = True):
    """Thin wrapper around torch.amp.GradScaler (torch>=2.11 required)."""
    if device_type != "cuda" or not enabled:
        return None
    return torch.amp.GradScaler(device_type, enabled=enabled)
