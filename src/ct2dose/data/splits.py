"""Case-level split validation utilities."""

from __future__ import annotations

from collections.abc import Iterable


def assert_disjoint_case_splits(
    train_cases: Iterable[str],
    val_cases: Iterable[str],
    test_cases: Iterable[str],
) -> None:
    """Raise ValueError if any case identifier appears in multiple splits."""
    train = set(map(str, train_cases))
    val = set(map(str, val_cases))
    test = set(map(str, test_cases))

    overlaps = {
        "train_val": sorted(train & val),
        "train_test": sorted(train & test),
        "val_test": sorted(val & test),
    }
    nonempty = {name: ids for name, ids in overlaps.items() if ids}
    if nonempty:
        raise ValueError(f"Case-level split overlap detected: {nonempty}")
