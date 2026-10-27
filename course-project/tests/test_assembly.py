import numpy as np

from assembly import assemble_laplacian_system


def square_triangular_mesh():
    coordinates = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    return coordinates, elements


def test_global_lhs_is_dense_symmetric_and_has_constant_nullspace():
    coordinates, elements = square_triangular_mesh()
    lhs, rhs = assemble_laplacian_system(
        coordinates, elements, np.zeros(coordinates.shape[0])
    )

    assert isinstance(lhs, np.ndarray)
    assert lhs.shape == (4, 4)
    np.testing.assert_allclose(lhs, lhs.T, atol=1e-12)
    np.testing.assert_allclose(lhs @ np.ones(4), 0.0, atol=1e-12)
    np.testing.assert_allclose(rhs, 0.0, atol=1e-12)


def test_assembled_rhs_is_negative_global_residual():
    coordinates, elements = square_triangular_mesh()
    solution = np.array([0.2, 1.0, -0.4, 0.7])
    lhs, rhs = assemble_laplacian_system(coordinates, elements, solution)
    np.testing.assert_allclose(rhs, -(lhs @ solution), atol=1e-12)
