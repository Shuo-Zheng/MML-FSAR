"""Metric helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def aggregate_accuracy(logits: Any, labels: Any) -> float:
    """Compute classification accuracy for numpy arrays or torch tensors."""

    if hasattr(logits, "argmax") and hasattr(labels, "detach"):
        predictions = logits.argmax(dim=-1)
        return float((predictions == labels).float().mean().item())

    logits_array = np.asarray(logits)
    labels_array = np.asarray(labels)
    predictions = np.argmax(logits_array, axis=-1)
    return float(np.mean(predictions == labels_array))
