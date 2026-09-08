"""Cross-version torch compatibility helpers for CUDA and AMP."""

import contextlib
from typing import Optional

import torch


def amp_autocast(device_type: str = "cuda", dtype: Optional[torch.dtype] = None, enabled: bool = True):
    """Return a compatible autocast context manager across PyTorch versions."""
    if dtype is None:
        dtype = torch.float16

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=enabled)

    if device_type == "cuda" and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)

    return contextlib.nullcontext()


def make_grad_scaler(device_type: str = "cuda", enabled: bool = True):
    """Return a GradScaler compatible with older and newer PyTorch releases."""
    if device_type != "cuda" or not enabled:
        return None

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device_type, enabled=enabled)

    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=enabled)

    return None
