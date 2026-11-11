"""Collect prescribed Dirichlet boundary values from named boundary groups."""

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
