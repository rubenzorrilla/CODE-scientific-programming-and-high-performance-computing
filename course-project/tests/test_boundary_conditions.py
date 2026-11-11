import numpy as np
import pytest

from boundary_conditions import collect_dirichlet_conditions


def test_collect_boundary_values_and_merge_corner_nodes():
    coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )
    boundaries = {
        "left": np.array([0, 3]),
        "right": np.array([1, 2]),
        "bottom": np.array([0, 1]),
        "top": np.array([3, 2]),
    }
    dofs, values = collect_dirichlet_conditions(
        coordinates,
        boundaries,
        {
            "left": 0.0,
            "right": 0.0,
            "bottom": 0.0,
            "top": lambda x, y: np.sin(np.pi * x),
        },
    )

    np.testing.assert_array_equal(dofs, [0, 1, 2, 3])
    np.testing.assert_allclose(values, 0.0, atol=1e-12)


def test_conflicting_corner_values_are_rejected():
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0]])
    boundaries = {"left": np.array([0]), "bottom": np.array([0, 1])}
    with pytest.raises(ValueError, match="Inconsistent"):
        collect_dirichlet_conditions(
            coordinates, boundaries, {"left": 0.0, "bottom": 1.0}
        )


def test_missing_boundary_name_is_rejected():
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0]])
    boundaries = {"left": np.array([0])}
    with pytest.raises(KeyError):
        collect_dirichlet_conditions(
            coordinates, boundaries, {"top": 1.0}
        )
