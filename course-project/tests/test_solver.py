import numpy as np
import pytest

from solver import solve_linear_system


def test_solve_dense_linear_system():
    lhs = np.array([[4.0, 1.0], [1.0, 3.0]])
    expected = np.array([1.0, 2.0])
    rhs = lhs @ expected
    np.testing.assert_allclose(solve_linear_system(lhs, rhs), expected)


def test_reject_non_square_matrix():
    with pytest.raises(ValueError, match="square"):
        solve_linear_system(np.ones((2, 3)), np.ones(2))


def test_detect_singular_system():
    lhs = np.array([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(RuntimeError, match="could not be solved"):
        solve_linear_system(lhs, np.ones(2))
