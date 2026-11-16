import numpy as np

from assembly_helper import build_element_systems
from boundary_conditions import collect_boundary_values

from backends.dense_reference import (
    allocate_global_system,
    assemble_global_system,
    impose_dirichlet_conditions,
    solve_linear_system,
)


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


def test_allocate_global_system_returns_zero_dense_arrays():
    global_lhs, global_rhs = allocate_global_system(4)

    assert global_lhs.shape == (4, 4)
    assert global_rhs.shape == (4,)
    np.testing.assert_allclose(global_lhs, 0.0)
    np.testing.assert_allclose(global_rhs, 0.0)


def test_backend_assembles_local_arrays_into_global_system():
    coordinates, elements = square_triangular_mesh()
    element_lhs, element_rhs = build_element_systems(coordinates, elements)

    global_lhs, global_rhs = allocate_global_system(coordinates.shape[0])
    global_lhs, global_rhs = assemble_global_system(
        global_lhs, global_rhs, element_lhs, element_rhs, elements
    )

    assert global_lhs.shape == (4, 4)
    assert global_rhs.shape == (4,)
    np.testing.assert_allclose(global_lhs, global_lhs.T, atol=1e-12)
    np.testing.assert_allclose(global_lhs @ np.ones(4), 0.0, atol=1e-12)
    np.testing.assert_allclose(global_rhs, 0.0)


def test_backend_imposes_dirichlet_conditions_without_mutating_input():
    lhs = np.array([[2.0, -1.0], [-1.0, 2.0]])
    rhs = np.zeros(2)

    lhs_bc, rhs_bc = impose_dirichlet_conditions(lhs, rhs, [0], [3.0])

    np.testing.assert_allclose(lhs_bc, [[1.0, 0.0], [0.0, 2.0]])
    np.testing.assert_allclose(rhs_bc, [3.0, 3.0])
    np.testing.assert_allclose(lhs, [[2.0, -1.0], [-1.0, 2.0]])
    np.testing.assert_allclose(rhs, 0.0)


def test_backend_solves_constrained_dense_system():
    lhs = np.array([[2.0, -1.0], [-1.0, 2.0]])
    rhs = np.zeros(2)
    lhs_bc, rhs_bc = impose_dirichlet_conditions(lhs, rhs, [0], [3.0])

    solution = solve_linear_system(lhs_bc, rhs_bc)
    np.testing.assert_allclose(solution, [3.0, 1.5])
