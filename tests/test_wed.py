import numpy as np

from ct2dose.physics.wed_attenuation import attenuation_from_wed, cumulative_wed


def test_homogeneous_wed_is_linear() -> None:
    density = np.ones(4)
    wed = cumulative_wed(density, 2.0)
    np.testing.assert_allclose(wed, [2.0, 4.0, 6.0, 8.0])


def test_attenuation_is_nonincreasing() -> None:
    wed = np.array([0.0, 1.0, 2.0, 3.0])
    fluence = attenuation_from_wed(wed, beta=0.2)
    assert np.all(np.diff(fluence) <= 0.0)
