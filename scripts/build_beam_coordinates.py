#!/usr/bin/env python3
"""Build beam-depth and radial-distance maps after geometry audit."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    raise SystemExit(
        "Implement only after spacing, origin, orientation, entry point, and "
        "beam direction have been verified. Reusable math belongs in "
        "src/ct2dose/geometry/beam_coordinates.py."
    )


if __name__ == "__main__":
    main()
