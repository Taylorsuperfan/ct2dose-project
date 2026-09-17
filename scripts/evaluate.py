#!/usr/bin/env python3
"""Unified evaluation entry point for every model family."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    raise SystemExit(
        "Evaluation orchestration will be added after the common metric schema "
        "is frozen under src/ct2dose/evaluation/."
    )


if __name__ == "__main__":
    main()
