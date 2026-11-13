"""Private dense reference backend for the ``K u = F`` workflow.

This module is a complete, deliberately simple dense implementation. It is
the baseline that student backends are expected to replace or optimize.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

import visualization


def allocate_global_system(number_of_nodes: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Allocate a dense global stiffness matrix and forcing vector.

    Parameters
    ----------
    number_of_nodes:
        Number of global mesh nodes.

    Returns
    -------
    global_lhs:
        Dense array of shape (number_of_nodes, number_of_nodes), initialized to zero.
    global_rhs:
        Dense array of shape (number_of_nodes,), initialized to zero.
    """
    if number_of_nodes < 1:
        raise ValueError("number_of_nodes must be positive.")

    global_lhs = np.zeros((number_of_nodes, number_of_nodes), dtype=float)
    global_rhs = np.zeros(number_of_nodes, dtype=float)
    return global_lhs, global_rhs


def assemble_global_system(
    global_lhs: ArrayLike,
    global_rhs: ArrayLike,
    element_lhs: ArrayLike,
    element_rhs: ArrayLike,
    elements: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Scatter local element contributions into dense global arrays.

    Parameters
    ----------
    global_lhs:
        Dense global stiffness matrix, shape (n_nodes, n_nodes).
    global_rhs:
        Dense global forcing vector, shape (n_nodes,).
    element_lhs:
        Local element stiffness arrays, shape (n_elements, n_local, n_local).
    element_rhs:
        Local element forcing arrays, shape (n_elements, n_local).
    elements:
        Zero-based element connectivity, shape (n_elements, n_local).

    Returns
    -------
    global_lhs:
        Updated dense global stiffness matrix.
    global_rhs:
        Updated dense global forcing vector.
    """
    global_lhs = np.asarray(global_lhs, dtype=float)
    global_rhs = np.asarray(global_rhs, dtype=float)
    element_lhs = np.asarray(element_lhs, dtype=float)
    element_rhs = np.asarray(element_rhs, dtype=float)
    elements = np.asarray(elements, dtype=np.int64)

    if global_lhs.ndim != 2 or global_lhs.shape[0] != global_lhs.shape[1]:
        raise ValueError("global_lhs must be a square dense matrix.")
    if global_rhs.shape != (global_lhs.shape[0],):
        raise ValueError("global_rhs must contain one value per global row.")
    if elements.ndim != 2 or elements.shape[0] == 0:
        raise ValueError("elements must be a non-empty 2D array.")
    if elements.shape[1] not in (3, 4):
        raise ValueError("elements must contain 3-node P1 or 4-node Q1 elements.")
    if np.any(elements < 0) or np.any(elements >= global_lhs.shape[0]):
        raise ValueError("elements contains invalid global node indices.")

    number_of_elements, nodes_per_element = elements.shape
    expected_lhs_shape = (
        number_of_elements,
        nodes_per_element,
        nodes_per_element,
    )
    expected_rhs_shape = (number_of_elements, nodes_per_element)

    if element_lhs.shape != expected_lhs_shape:
        raise ValueError(
            "element_lhs must have shape (n_elements, n_local, n_local)."
        )
    if element_rhs.shape != expected_rhs_shape:
        raise ValueError("element_rhs must have shape (n_elements, n_local).")
    if not np.all(np.isfinite(element_lhs)) or not np.all(np.isfinite(element_rhs)):
        raise ValueError("Element contributions must be finite.")

    for element_index, connectivity in enumerate(elements):
        for local_i in range(nodes_per_element):
            global_i = int(connectivity[local_i])
            global_rhs[global_i] += element_rhs[element_index, local_i]

            for local_j in range(nodes_per_element):
                global_j = int(connectivity[local_j])
                global_lhs[global_i, global_j] += element_lhs[
                    element_index, local_i, local_j
                ]

    return global_lhs, global_rhs


def impose_dirichlet_conditions(
    global_lhs: ArrayLike,
    global_rhs: ArrayLike,
    constrained_dofs: ArrayLike,
    prescribed_values: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Impose prescribed Dirichlet values on a dense system.

    The constrained rows and columns are zeroed, their diagonal entries are
    set to one, and contributions from constrained columns are moved to the
    RHS of the free equations.

    Returns a new dense system and leaves the input arrays unchanged.
    """
    matrix = np.asarray(global_lhs, dtype=float).copy()
    vector = np.asarray(global_rhs, dtype=float).copy()
    constrained = np.asarray(constrained_dofs, dtype=np.int64)
    prescribed = np.asarray(prescribed_values, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("global_lhs must be a square matrix.")
    if vector.shape != (matrix.shape[0],):
        raise ValueError("global_rhs must contain one value per matrix row.")
    if constrained.shape != prescribed.shape:
        raise ValueError(
            "constrained_dofs and prescribed_values must have the same length."
        )

    number_of_nodes = matrix.shape[0]
    if np.any(constrained < 0) or np.any(constrained >= number_of_nodes):
        raise ValueError("constrained_dofs contains invalid node indices.")
    if not np.all(np.isfinite(prescribed)):
        raise ValueError("prescribed_values must contain only finite values.")

    free_mask = np.ones(number_of_nodes, dtype=bool)
    free_mask[constrained] = False
    free_rows = np.nonzero(free_mask)[0]

    # Move known boundary values from constrained columns into the free RHS.
    for row in free_rows:
        for index, dof in enumerate(constrained):
            vector[row] -= matrix[row, dof] * prescribed[index]

    # Apply the constrained equations.
    for index, dof in enumerate(constrained):
        matrix[dof, :] = 0.0
        matrix[:, dof] = 0.0
        matrix[dof, dof] = 1.0
        vector[dof] = prescribed[index]

    return matrix, vector


def solve_linear_system(
    global_lhs: ArrayLike,
    global_rhs: ArrayLike,
) -> NDArray[np.float64]:
    """Solve a dense linear system using ``numpy.linalg.solve``."""

    if global_lhs.ndim != 2 or global_lhs.shape[0] != global_lhs.shape[1]:
        raise ValueError("lhs must be a square matrix.")
    if global_rhs.shape != (global_lhs.shape[0],):
        raise ValueError("rhs must contain one value per matrix row.")
    if not np.all(np.isfinite(global_lhs)) or not np.all(np.isfinite(global_rhs)):
        raise ValueError("The linear system must contain only finite values.")

    try:
        solution = np.linalg.solve(global_lhs, global_rhs)
    except np.linalg.LinAlgError as error:
        raise RuntimeError(
            "The dense linear system could not be solved. "
            "Check that sufficient boundary conditions have been imposed."
        ) from error

    if not np.all(np.isfinite(solution)):
        raise RuntimeError("The solver returned non-finite values.")

    return solution

def visualize(
    coordinates: ArrayLike,
    elements: ArrayLike,
    solution: ArrayLike,
    scalar_name: str = "u",
    show_edges: bool = True,
    screenshot: str | None = None,
):
    """Visualize the nodal solution with the project visualization utility."""
    return visualization.plot_results(
        coordinates,
        elements,
        solution,
        scalar_name=scalar_name,
        show_edges=show_edges,
        screenshot=screenshot,
    )
