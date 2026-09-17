#!/usr/bin/env python3
"""Aggregate record-level metrics to the independent case level."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    raise SystemExit(
        "Define the record-metric CSV schema first. Aggregation logic belongs "
        "in src/ct2dose/evaluation/ and must group by case_id before statistics."
    )


if __name__ == "__main__":
    main()
