"""Interfaces for explicit physical transport models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PhysicalTransportModel(ABC):
    """Base interface for a model that produces an explicit physical dose prior."""

    @abstractmethod
    def predict(self, *, ct: Any, beam: Any, geometry: Any) -> Any:
        """Return a physical dose prior aligned with the target dose grid."""
        raise NotImplementedError
