#!/usr/bin/env python3
"""Audit physical metadata and split integrity.

Adapt the manifest loader only after documenting the real record schema. Keep
all reusable audit logic in `src/ct2dose/data/`; this script should remain a
thin command-line entry point.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("thesis/results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raise SystemExit(
        "Manifest schema is not yet audited. Document the schema in "
        "thesis/FORMULATION.md, then implement a reusable loader under "
        "src/ct2dose/data/."
    )


if __name__ == "__main__":
    main()
