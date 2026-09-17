"""Conditional flow path interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class ConditionalPath(Protocol):
    def sample(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(x_t, target_velocity)` for paired endpoints."""


@dataclass(frozen=True)
class LinearDosePath:
    """Linear paired path from a physical dose prior to a target dose."""

    def sample(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x0.shape != x1.shape:
            raise ValueError(f"Endpoint shape mismatch: {x0.shape} vs {x1.shape}")
        while t.ndim < x0.ndim:
            t = t.unsqueeze(-1)
        xt = (1.0 - t) * x0 + t * x1
        ut = x1 - x0
        return xt, ut
