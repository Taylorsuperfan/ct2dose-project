"""Beam-aligned attenuation-diffusion model specification.

The numerical solver is intentionally not implemented in the scaffold because
its discretization, source term, boundary conditions, coefficient calibration,
and physical units must first be approved in `thesis/FORMULATION.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AttenuationDiffusionConfig:
    attenuation_coefficient: float
    transverse_diffusion_coefficient: float
    depth_step: float

    def validate(self) -> None:
        if self.attenuation_coefficient < 0:
            raise ValueError("attenuation_coefficient must be non-negative")
        if self.transverse_diffusion_coefficient < 0:
            raise ValueError("transverse_diffusion_coefficient must be non-negative")
        if self.depth_step <= 0:
            raise ValueError("depth_step must be positive")


def solve_attenuation_diffusion(*, source: Any, density: Any, config: AttenuationDiffusionConfig) -> Any:
    """Placeholder pending an approved PDE and numerical discretization."""
    config.validate()
    raise NotImplementedError(
        "Define boundary conditions, units, coefficient fields, and a stable numerical scheme first."
    )
