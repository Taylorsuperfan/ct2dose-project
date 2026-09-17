"""HU-to-relative-density calibration utilities.

No default clinical calibration is provided. The calibration points must come
from an authoritative project or scanner-specific source and must be recorded
in the experiment configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PiecewiseLinearDensityCalibration:
    hu_points: tuple[float, ...]
    relative_density_points: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.hu_points) != len(self.relative_density_points):
            raise ValueError("HU and density point lists must have equal length")
        if len(self.hu_points) < 2:
            raise ValueError("At least two calibration points are required")
        if any(b <= a for a, b in zip(self.hu_points, self.hu_points[1:])):
            raise ValueError("hu_points must be strictly increasing")
        if any(value < 0 for value in self.relative_density_points):
            raise ValueError("Relative density must be non-negative")

    def __call__(self, hu: np.ndarray) -> np.ndarray:
        hu_array = np.asarray(hu, dtype=np.float64)
        return np.interp(
            hu_array,
            np.asarray(self.hu_points, dtype=np.float64),
            np.asarray(self.relative_density_points, dtype=np.float64),
        )
