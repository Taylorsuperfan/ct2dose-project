#!/usr/bin/env python3
"""Run an explicit physics-only dose baseline from a versioned YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    raise SystemExit(
        "Implement after the physical formulation and metadata audit are "
        "approved. Physics implementation belongs under src/ct2dose/physics/."
    )


if __name__ == "__main__":
    main()
