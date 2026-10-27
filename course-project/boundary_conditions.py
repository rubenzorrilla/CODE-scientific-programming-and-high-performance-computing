"""Collect and impose Dirichlet boundary conditions on a dense system."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

BoundaryValue: TypeAlias = float | int | Callable[[float, float], float]


def collect_dirichlet_conditions(
    coordinates: ArrayLike,
    boundaries: Mapping[str, ArrayLike],
    boundary_values: Mapping[str, BoundaryValue],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Collect prescribed solution values from named boundary-node groups.

    ``boundary_values`` maps a boundary name to a constant or a function
    ``value(x, y)``. Values assigned more than once at a shared corner node
    must agree.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (n_nodes, 2).")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates must contain only finite values.")

    values_by_dof: dict[int, float] = {}
    for boundary_name, boundary_value in boundary_values.items():
        if boundary_name not in boundaries:
            raise KeyError(f"Boundary {boundary_name!r} is not present in the mesh.")

        node_indices = np.asarray(boundaries[boundary_name], dtype=np.int64).ravel()
        if np.any(node_indices < 0) or np.any(node_indices >= len(coordinates)):
            raise ValueError(
                f"Boundary {boundary_name!r} contains invalid node indices."
            )

        for dof in node_indices:
            x, y = coordinates[dof]
            value = (
                float(boundary_value(float(x), float(y)))
                if callable(boundary_value)
                else float(boundary_value)
            )
            if not np.isfinite(value):
                raise ValueError("Dirichlet boundary values must be finite.")

            dof = int(dof)
            if dof in values_by_dof and not np.isclose(
                values_by_dof[dof], value, rtol=1e-10, atol=1e-12
            ):
                raise ValueError(
                    f"Inconsistent Dirichlet values at node {dof}: "
                    f"{values_by_dof[dof]} and {value}."
                )
            values_by_dof[dof] = value

    dofs = np.asarray(sorted(values_by_dof), dtype=np.int64)
    values = np.asarray([values_by_dof[int(dof)] for dof in dofs], dtype=float)
    return dofs, values


def apply_dirichlet_conditions(
    lhs: ArrayLike,
    rhs: ArrayLike,
    constrained_dofs: ArrayLike,
    prescribed_values: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply Dirichlet constraints to the dense system ``lhs @ x = rhs``.

    Known values are moved to the RHS, then constrained rows and columns are
    zeroed and their diagonal entries set to one.

    In the residual-based workflow, ``x`` is the correction ``delta_u``.
    Thus ``prescribed_values`` are boundary corrections, i.e. prescribed
    solution values minus the current solution values.
    """
    matrix = np.asarray(lhs, dtype=float).copy()
    vector = np.asarray(rhs, dtype=float).copy()
    dofs = np.asarray(constrained_dofs, dtype=np.int64).ravel()
    values = np.asarray(prescribed_values, dtype=float).ravel()

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("lhs must be a square matrix.")
    if vector.shape != (matrix.shape[0],):
        raise ValueError("rhs must contain one value per matrix row.")
    if values.shape != dofs.shape:
        raise ValueError("prescribed_values must have one value per constrained DOF.")
    if not np.all(np.isfinite(vector)) or not np.all(np.isfinite(values)):
        raise ValueError("rhs and prescribed values must be finite.")
    if np.any(dofs < 0) or np.any(dofs >= matrix.shape[0]):
        raise ValueError("constrained_dofs contains invalid indices.")
    if np.unique(dofs).size != dofs.size:
        raise ValueError("constrained_dofs must not contain duplicates.")

    if dofs.size == 0:
        return matrix, vector

    # Lift known values into the RHS of all equations before modifying columns.
    vector -= matrix[:, dofs] @ values

    # Replace constrained rows/columns with identity rows.
    matrix[dofs, :] = 0.0
    matrix[:, dofs] = 0.0
    matrix[dofs, dofs] = 1.0
    vector[dofs] = values

    return matrix, vector
