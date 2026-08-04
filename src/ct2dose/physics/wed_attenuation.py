"""Minimal beam-aligned WED/attenuation helpers.

These functions assume samples are already ordered along verified beam rays.
They are suitable for phantom and unit tests, not yet a complete patient ray tracer.
"""

from __future__ import annotations

import numpy as np


def cumulative_wed(
    relative_density: np.ndarray,
    step_length: float | np.ndarray,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Compute cumulative water-equivalent depth along a beam-aligned axis."""
    density = np.asarray(relative_density, dtype=np.float64)
    if np.any(~np.isfinite(density)) or np.any(density < 0):
        raise ValueError("relative_density must be finite and non-negative")
    step = np.asarray(step_length, dtype=np.float64)
    if np.any(~np.isfinite(step)) or np.any(step <= 0):
        raise ValueError("step_length must be finite and positive")
    return np.cumsum(density * step, axis=axis)


def attenuation_from_wed(
    wed: np.ndarray,
    *,
    beta: float,
    incident_fluence: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Compute a simple exponential attenuation prior from WED."""
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and non-negative")
    wed_array = np.asarray(wed, dtype=np.float64)
    fluence = np.asarray(incident_fluence, dtype=np.float64)
    return fluence * np.exp(-beta * wed_array)
