"""Element-level LHS and RHS contributions for the Laplace equation.

The element LHS is the Jacobian of the Laplace residual. The element RHS is
the negative residual, so the Newton/correction equation has the form

    LHS_e @ delta_u_e = RHS_e.

For the homogeneous Laplace equation, R_e = K_e @ u_e, hence
LHS_e = K_e and RHS_e = -K_e @ u_e.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def calculate_element_lhs(element_coordinates: ArrayLike) -> NDArray[np.float64]:
    """Calculate the element Laplacian matrix for one P1 triangle or Q1 quad.

    Parameters
    ----------
    element_coordinates:
        Coordinates of this element's nodes, shape (3, 2) for P1 triangles
        or (4, 2) for Q1 quadrilaterals. For Q1, nodes must be ordered around
        the element (the usual Gmsh quadrilateral ordering).

    Returns
    -------
    ndarray
        Element Laplacian matrix, shape (n_nodes, n_nodes).
    """
    coordinates = np.asarray(element_coordinates, dtype=float)

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("element_coordinates must have shape (3, 2) or (4, 2).")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("element_coordinates must contain only finite values.")

    if coordinates.shape[0] == 3:
        return _calculate_p1_lhs(coordinates)
    if coordinates.shape[0] == 4:
        return _calculate_q1_lhs(coordinates)

    raise ValueError("Only P1 triangles (3 nodes) and Q1 quadrilaterals (4 nodes) are supported.")


def calculate_element_rhs(
    element_coordinates: ArrayLike,
    element_solution: ArrayLike,
) -> NDArray[np.float64]:
    """Calculate the local RHS, i.e. the negative element residual.

    Parameters
    ----------
    element_coordinates:
        Coordinates of this element's nodes, shape (3, 2) for P1 or (4, 2)
        for Q1.
    element_solution:
        Current solution values at this element's nodes, shape (3,) for P1
        or (4,) for Q1.

    Returns
    -------
    ndarray
        The local right-hand-side vector ``-K_e @ u_e`` for the homogeneous
        Laplace equation. Source terms and boundary conditions are not
        included here; they will be handled by the global workflow.
    """
    coordinates = np.asarray(element_coordinates, dtype=float)
    solution = np.asarray(element_solution, dtype=float)

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("element_coordinates must have shape (3, 2) or (4, 2).")
    if solution.shape != (coordinates.shape[0],):
        raise ValueError("element_solution must contain one value per element node.")
    if not np.all(np.isfinite(solution)):
        raise ValueError("element_solution must contain only finite values.")

    lhs = calculate_element_lhs(coordinates)
    return -(lhs @ solution)


def _calculate_p1_lhs(coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
    """Integrate constant P1 shape-function gradients over a triangle."""
    # Reference-triangle derivatives for nodes (0,0), (1,0), (0,1).
    gradients_reference = np.array(
        [
            [-1.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    # J maps reference coordinates to physical coordinates.
    jacobian = coordinates.T @ gradients_reference
    determinant = float(np.linalg.det(jacobian))
    _check_jacobian(determinant, jacobian)

    gradients_physical = gradients_reference @ np.linalg.inv(jacobian)
    area = 0.5 * abs(determinant)
    return area * (gradients_physical @ gradients_physical.T)


def _calculate_q1_lhs(coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
    """Integrate Q1 shape-function gradients using 2x2 Gauss quadrature."""
    gauss_point = 1.0 / np.sqrt(3.0)
    points = (-gauss_point, gauss_point)
    lhs = np.zeros((4, 4), dtype=float)

    for xi in points:
        for eta in points:
            # Derivatives of the four bilinear shape functions with respect
            # to the reference coordinates (xi, eta).
            dN_dxi = 0.25 * np.array(
                [-(1.0 - eta), 1.0 - eta, 1.0 + eta, -(1.0 + eta)]
            )
            dN_deta = 0.25 * np.array(
                [-(1.0 - xi), -(1.0 + xi), 1.0 + xi, 1.0 - xi]
            )
            gradients_reference = np.column_stack((dN_dxi, dN_deta))

            jacobian = coordinates.T @ gradients_reference
            determinant = float(np.linalg.det(jacobian))
            _check_jacobian(determinant, jacobian)

            gradients_physical = gradients_reference @ np.linalg.inv(jacobian)
            lhs += (
                gradients_physical
                @ gradients_physical.T
                * abs(determinant)
            )  # Both Gauss weights are 1.

    return lhs


def _check_jacobian(determinant: float, jacobian: NDArray[np.float64]) -> None:
    """Raise a clear error for a degenerate element mapping."""
    scale = float(np.linalg.norm(jacobian, ord=2))
    tolerance = np.finfo(float).eps * scale * scale
    if abs(determinant) <= tolerance:
        raise ValueError("Degenerate element: its Jacobian determinant is zero.")
