import numpy as np

from ct2dose.evaluation.metrics import mae, peak_normalized_absolute_error


def test_mae() -> None:
    assert mae(np.array([1.0, 3.0]), np.array([1.0, 1.0])) == 1.0


def test_peak_normalized_error_uses_target_peak() -> None:
    pred = np.array([0.0, 0.8])
    target = np.array([0.0, 1.0])
    error = peak_normalized_absolute_error(pred, target)
    np.testing.assert_allclose(error, [0.0, 0.2])
