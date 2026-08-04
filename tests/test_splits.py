import pytest

from ct2dose.data.splits import assert_disjoint_case_splits


def test_disjoint_splits_pass() -> None:
    assert_disjoint_case_splits(["a", "b"], ["c"], ["d"])


def test_overlap_raises() -> None:
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_case_splits(["a", "b"], ["b"], ["d"])
