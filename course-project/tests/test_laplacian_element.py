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


@pytest.mark.parametrize(
    "coordinates,solution",
    [
        (
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            np.array([1.0, 2.0, 3.0]),
        ),
        (
            np.array(
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            ),
            np.array([1.0, 2.0, 3.0, 4.0]),
        ),
    ],
)
def test_rhs_is_negative_element_residual(coordinates, solution):
    lhs = calculate_element_lhs(coordinates)
    rhs = calculate_element_rhs(coordinates, solution)
    np.testing.assert_allclose(rhs, -(lhs @ solution))


def test_invalid_element_node_count():
    with pytest.raises(ValueError):
        calculate_element_lhs(np.zeros((5, 2)))


def test_invalid_solution_size():
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        calculate_element_rhs(coordinates, np.zeros(4))
