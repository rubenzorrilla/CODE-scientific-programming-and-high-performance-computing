import numpy as np
import pytest

from laplacian_element import calculate_element_lhs, calculate_element_rhs


def test_p1_lhs_for_unit_right_triangle():
    coordinates = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    expected = np.array(
        [
            [1.0, -0.5, -0.5],
            [-0.5, 0.5, 0.0],
            [-0.5, 0.0, 0.5],
        ]
    )
    np.testing.assert_allclose(calculate_element_lhs(coordinates), expected)


def test_q1_lhs_is_symmetric_and_has_constant_nullspace():
    # Gmsh Q1 node order: bottom-left, bottom-right, top-right, top-left.
    coordinates = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    lhs = calculate_element_lhs(coordinates)

    assert lhs.shape == (4, 4)
    np.testing.assert_allclose(lhs, lhs.T, atol=1e-12)
    np.testing.assert_allclose(lhs @ np.ones(4), 0.0, atol=1e-12)
    assert np.linalg.eigvalsh(lhs).min() >= -1e-12


def test_p1_rhs_is_mass_weighted_forcing():
    coordinates = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    forcing = np.array([1.0, 2.0, 3.0])
    area = 0.5
    mass = (area / 12.0) * (np.eye(3) + np.ones((3, 3)))

    np.testing.assert_allclose(
        calculate_element_rhs(coordinates, forcing), mass @ forcing
    )


def test_q1_constant_forcing_gives_mass_weighted_rhs():
    coordinates = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(
        calculate_element_rhs(coordinates, 1.0), np.full(4, 0.25)
    )


def test_zero_forcing_gives_zero_element_rhs():
    coordinates = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(calculate_element_rhs(coordinates), 0.0)


def test_invalid_element_node_count():
    with pytest.raises(ValueError):
        calculate_element_lhs(np.zeros((5, 2)))


def test_invalid_forcing_size():
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        calculate_element_rhs(coordinates, np.zeros(4))
