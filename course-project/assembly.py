"""Deliberately simple, dense global assembly for the CODE course project.

This is a reference implementation, not an efficient one. The global matrix
is dense, and element contributions are accumulated with explicit Python
loops. Optimizing this module is an intended part of the course project.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from laplacian_element import calculate_element_lhs, calculate_element_rhs


def assemble_laplacian_system(
    coordinates: ArrayLike,
    elements: ArrayLike,
    solution: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Assemble a dense Jacobian and negative residual.

    Parameters
    ----------
    coordinates:
        Nodal coordinates, shape (n_nodes, 2).
    elements:
        Zero-based connectivity for P1 triangles (3 nodes) or Q1
        quadrilaterals (4 nodes).
    solution:
        Current nodal solution, shape (n_nodes,).

    Returns
    -------
    lhs:
        Dense global Jacobian, shape (n_nodes, n_nodes).
    rhs:
        Dense global negative residual, i.e. ``-R(solution)``.

    Notes
    -----
    This implementation intentionally favors transparency over performance:
    it stores the global matrix densely and uses nested Python loops even
    though each element only contributes to a small subset of entries.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    elements = np.asarray(elements, dtype=np.int64)
    solution = np.asarray(solution, dtype=float)

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (n_nodes, 2).")
    if coordinates.shape[0] == 0 or not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates must be non-empty and finite.")
    if elements.ndim != 2 or elements.shape[0] == 0:
        raise ValueError("elements must be a non-empty 2D array.")
    if elements.shape[1] not in (3, 4):
        raise ValueError("elements must contain 3-node P1 or 4-node Q1 elements.")
    if np.any(elements < 0) or np.any(elements >= coordinates.shape[0]):
        raise ValueError("elements contains invalid node indices.")
    if solution.shape != (coordinates.shape[0],):
        raise ValueError("solution must contain one value per mesh node.")
    if not np.all(np.isfinite(solution)):
        raise ValueError("solution must contain only finite values.")

    number_of_nodes = coordinates.shape[0]

    # Intentionally dense: most entries are zero for a finite element matrix.
    lhs = np.zeros((number_of_nodes, number_of_nodes), dtype=float)
    rhs = np.zeros(number_of_nodes, dtype=float)

    for connectivity in elements:
        element_coordinates = coordinates[connectivity]
        element_solution = solution[connectivity]

        element_lhs = calculate_element_lhs(element_coordinates)
        element_rhs = calculate_element_rhs(
            element_coordinates, element_solution
        )

        # Intentionally use explicit loops instead of vectorized/scatter
        # operations. Adjacent elements contribute to shared global entries.
        for local_i in range(len(connectivity)):
            global_i = connectivity[local_i]
            rhs[global_i] += element_rhs[local_i]

            for local_j in range(len(connectivity)):
                global_j = connectivity[local_j]
                lhs[global_i, global_j] += element_lhs[local_i, local_j]

    return lhs, rhs
