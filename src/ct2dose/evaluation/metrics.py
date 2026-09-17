"""Small, explicit evaluation utilities.

Keep final thesis metrics in one module so all models share identical definitions.
"""

from __future__ import annotations

import numpy as np


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError("prediction and target must have identical shapes")
    return float(np.mean(np.abs(pred - true)))


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError("prediction and target must have identical shapes")
    return float(np.mean((pred - true) ** 2))


def peak_normalized_absolute_error(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Pointwise absolute error normalized by the ground-truth peak."""
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError("prediction and target must have identical shapes")
    peak = float(np.max(true))
    return np.abs(pred - true) / max(peak, eps)
