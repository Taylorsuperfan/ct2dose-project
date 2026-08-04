#!/usr/bin/env python3
"""Unified neural-model training entry point.

The final script should load a versioned config and call reusable package code.
Do not duplicate model/loss/path implementations inside this file.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    raise SystemExit("Training orchestration will be added after baseline reproduction.")


if __name__ == "__main__":
    main()
