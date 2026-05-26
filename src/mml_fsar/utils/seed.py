"""Random seed helpers."""

from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - depends on runtime environment
    torch = None
else:
    if not hasattr(torch, "manual_seed"):
        torch = None


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds when available."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if hasattr(torch, "cuda"):
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
