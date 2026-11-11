"""Element-level stiffness and forcing contributions for the Poisson equation.

For

    div(grad(u)) = f(x, y)

the element system is

    K_e u_e = F_e,

where K_e is the element stiffness matrix and F_e is the mass-weighted
forcing source
    F_e = M_e f_e.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_element_coordinates(
    element_coordinates: ArrayLike,
) -> NDArray[np.float64]:
    coordinates = np.asarray(element_coordinates, dtype=float)

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("element_coordinates must have shape (3, 2) or (4, 2).")
    if coordinates.shape[0] not in (3, 4):
        raise ValueError(
            "Only P1 triangles (3 nodes) and Q1 quadrilaterals (4 nodes) are supported."
        )
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("element_coordinates must contain only finite values.")

    return coordinates


def calculate_element_lhs(element_coordinates: ArrayLike) -> NDArray[np.float64]:
    """Calculate the element stiffness matrix for one P1 triangle or Q1 quad.

    Parameters
    ----------
    element_coordinates:
        Coordinates of this element's nodes, shape (3, 2) for P1 triangles
        or (4, 2) for Q1 quadrilaterals. For Q1, nodes must be ordered around
        the element, using the usual Gmsh quadrilateral ordering.

    Returns
    -------
    ndarray
        Element stiffness matrix, shape (n_nodes, n_nodes).
    """
    coordinates = _validate_element_coordinates(element_coordinates)

    if coordinates.shape[0] == 3:
        return _calculate_p1_lhs(coordinates)
    return _calculate_q1_lhs(coordinates)


def calculate_element_rhs(
    element_coordinates: ArrayLike,
    forcing_values: ArrayLike = 0.0,
) -> NDArray[np.float64]:
    """Calculate the local RHS source vector for one element.

    Parameters
    ----------
    element_coordinates:
        Coordinates of this element's nodes, shape (3, 2) for P1 or (4, 2)
        for Q1.
    forcing_values:
        Scalar forcing value or one value per element node. A scalar is
        applied to every node.

    Returns
    -------
    ndarray
        Element RHS ``F_e = M_e f_e``, where ``M_e`` is the element mass
        matrix. With zero forcing this is the zero vector.
    """
    coordinates = _validate_element_coordinates(element_coordinates)

    values = np.asarray(forcing_values, dtype=float)
    if values.ndim == 0:
        values = np.full(coordinates.shape[0], float(values))

    if values.shape != (coordinates.shape[0],):
        raise ValueError("forcing_values must contain one value per element node.")
    if not np.all(np.isfinite(values)):
        raise ValueError("forcing_values must contain only finite values.")

    return _calculate_element_mass(coordinates) @ values


def _calculate_element_mass(
    coordinates: NDArray[np.float64],
) -> NDArray[np.float64]:
    if coordinates.shape[0] == 3:
        return _calculate_p1_mass(coordinates)
    return _calculate_q1_mass(coordinates)


def _calculate_p1_lhs(coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
    """Integrate constant P1 shape-function gradients over a triangle."""
    gradients_reference = np.array(
        [
            [-1.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    jacobian = coordinates.T @ gradients_reference
    determinant = float(np.linalg.det(jacobian))
    _check_jacobian(determinant, jacobian)

    gradients_physical = gradients_reference @ np.linalg.inv(jacobian)
    area = 0.5 * abs(determinant)
    return area * (gradients_physical @ gradients_physical.T)


def _calculate_p1_mass(coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
    """Integrate P1 shape functions over a triangle."""
    gradients_reference = np.array(
        [
            [-1.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    jacobian = coordinates.T @ gradients_reference
    determinant = float(np.linalg.det(jacobian))
    _check_jacobian(determinant, jacobian)

    area = 0.5 * abs(determinant)
    return (area / 12.0) * (np.eye(3) + np.ones((3, 3)))


def _calculate_q1_mass(coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
    """Integrate Q1 shape functions using 2x2 Gauss quadrature."""
    gauss_point = 1.0 / np.sqrt(3.0)
    points = (-gauss_point, gauss_point)
    mass = np.zeros((4, 4), dtype=float)

    for xi in points:
        for eta in points:
            shape_functions = 0.25 * np.array(
                [
                    (1.0 - xi) * (1.0 - eta),
                    (1.0 + xi) * (1.0 - eta),
                    (1.0 + xi) * (1.0 + eta),
                    (1.0 - xi) * (1.0 + eta),
                ]
            )

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

            mass += np.outer(shape_functions, shape_functions) * abs(determinant)

    return mass


def _calculate_q1_lhs(coordinates: NDArray[np.float64]) -> NDArray[np.float64]:
    """Integrate Q1 shape-function gradients using 2x2 Gauss quadrature."""
    gauss_point = 1.0 / np.sqrt(3.0)
    points = (-gauss_point, gauss_point)
    lhs = np.zeros((4, 4), dtype=float)

    for xi in points:
        for eta in points:
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
            )

    return lhs


def _check_jacobian(determinant: float, jacobian: NDArray[np.float64]) -> None:
    """Raise a clear error for a degenerate element mapping."""
    scale = float(np.linalg.norm(jacobian, ord=2))
    tolerance = np.finfo(float).eps * scale * scale
    if abs(determinant) <= tolerance:
        raise ValueError("Degenerate element: its Jacobian determinant is zero.")
