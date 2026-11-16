"""Build local element systems for the Poisson equation.

This module only constructs the local ``K_e`` and ``F_e`` arrays. Global
allocation, assembly, boundary-condition imposition, solving, and
visualization belong to the selected backend module.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from laplacian_element import calculate_element_lhs, calculate_element_rhs


def _evaluate_forcing_values(
    coordinates: ArrayLike,
    forcing: float | Callable[[float, float], float] | ArrayLike = 0.0,
) -> NDArray[np.float64]:
    """Evaluate a forcing field at the mesh nodes.

    ``forcing`` may be a constant, a function ``f(x, y)``, or an array with
    one value per node.
    """
    coordinates = np.asarray(coordinates, dtype=float)

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (n_nodes, 2).")
    if coordinates.shape[0] == 0 or not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates must be non-empty and finite.")

    if callable(forcing):
        values = np.array(
            [float(forcing(float(x), float(y))) for x, y in coordinates]
        )
    else:
        forcing_array = np.asarray(forcing, dtype=float)
        if forcing_array.ndim == 0:
            values = np.full(coordinates.shape[0], float(forcing_array))
        elif forcing_array.shape == (coordinates.shape[0],):
            values = forcing_array
        else:
            raise ValueError(
                "forcing must be a scalar, a function f(x, y), "
                "or one value per node."
            )

    if not np.all(np.isfinite(values)):
        raise ValueError("forcing values must be finite.")

    return np.asarray(values, dtype=float)


def build_element_systems(
    coordinates: ArrayLike,
    elements: ArrayLike,
    forcing: float | Callable[[float, float], float] | ArrayLike = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build local element systems ``K_e u_e = F_e``.

    Returns
    -------
    element_lhs:
        shape (n_elements, n_local, n_local).
    element_rhs:
        shape (n_elements, n_local).
    """
    coordinates = np.asarray(coordinates, dtype=float)
    elements = np.asarray(elements, dtype=np.int64)

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (n_nodes, 2).")
    if coordinates.shape[0] == 0 or not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates must be non-empty and finite.")
    if elements.ndim != 2 or elements.shape[0] == 0:
        raise ValueError("elements must be a non-empty 2D array.")
    if elements.shape[1] not in (3, 4):
        raise ValueError(
            "elements must contain 3-node P1 or 4-node Q1 elements."
        )
    if np.any(elements < 0) or np.any(elements >= coordinates.shape[0]):
        raise ValueError("elements contains invalid node indices.")

    forcing_values = _evaluate_forcing_values(coordinates, forcing)
    number_of_elements, nodes_per_element = elements.shape

    local_lhs = np.empty(
        (number_of_elements, nodes_per_element, nodes_per_element),
        dtype=float,
    )
    local_rhs = np.empty(
        (number_of_elements, nodes_per_element),
        dtype=float,
    )

    for element_index, connectivity in enumerate(elements):
        element_coordinates = coordinates[connectivity]
        local_lhs[element_index] = calculate_element_lhs(element_coordinates)
        local_rhs[element_index] = calculate_element_rhs(
            element_coordinates, forcing_values[connectivity]
        )

    return local_lhs, local_rhs
