"""Geometry helpers for beam-aligned physical coordinates.

This module deliberately accepts physical point coordinates as input. Converting
array indices to physical coordinates must use audited spacing/origin/orientation.
"""

from __future__ import annotations

import numpy as np


def normalize_direction(direction_xyz: np.ndarray) -> np.ndarray:
    """Return a unit beam direction vector."""
    direction = np.asarray(direction_xyz, dtype=np.float64)
    if direction.shape != (3,):
        raise ValueError(f"Expected direction shape (3,), got {direction.shape}")
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Beam direction must have a finite non-zero norm")
    return direction / norm


def project_points_to_beam(
    points_xyz: np.ndarray,
    entry_point_xyz: np.ndarray,
    direction_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute beam depth and radial distance for physical 3D points.

    Parameters
    ----------
    points_xyz:
        Array with final dimension 3 containing physical coordinates.
    entry_point_xyz:
        Physical beam-entry coordinate with shape (3,).
    direction_xyz:
        Beam direction vector with shape (3,). It is normalized internally.

    Returns
    -------
    depth:
        Signed projection along the beam direction.
    radial_distance:
        Euclidean distance from the beam axis.
    """
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError("points_xyz must have final dimension 3")
    entry = np.asarray(entry_point_xyz, dtype=np.float64)
    if entry.shape != (3,):
        raise ValueError("entry_point_xyz must have shape (3,)")

    beam = normalize_direction(np.asarray(direction_xyz, dtype=np.float64))
    delta = points - entry
    depth = np.einsum("...i,i->...", delta, beam)
    radial_vector = delta - depth[..., None] * beam
    radial_distance = np.linalg.norm(radial_vector, axis=-1)
    return depth, radial_distance
