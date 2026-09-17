import numpy as np

from ct2dose.geometry.beam_coordinates import project_points_to_beam


def test_axis_points_have_zero_radial_distance() -> None:
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    depth, radial = project_points_to_beam(
        points,
        entry_point_xyz=np.array([0.0, 0.0, 0.0]),
        direction_xyz=np.array([1.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(depth, [0.0, 2.0])
    np.testing.assert_allclose(radial, [0.0, 0.0])


def test_off_axis_point_has_expected_radial_distance() -> None:
    points = np.array([[2.0, 3.0, 4.0]])
    depth, radial = project_points_to_beam(
        points,
        entry_point_xyz=np.zeros(3),
        direction_xyz=np.array([1.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(depth, [2.0])
    np.testing.assert_allclose(radial, [5.0])
