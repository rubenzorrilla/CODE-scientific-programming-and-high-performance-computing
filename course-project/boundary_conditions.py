"""Collect prescribed Dirichlet boundary values from named boundary groups."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

BoundaryValue: TypeAlias = (
    float | int | Callable[[float, float], float] | Literal["free"]
)


def collect_dirichlet_conditions(
    coordinates: ArrayLike,
    boundaries: Mapping[str, ArrayLike],
    boundary_values: Mapping[str, BoundaryValue],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Collect prescribed solution values from named boundary-node groups.

    ``boundary_values`` maps a boundary name to:
        - A constant value.
        - A function ``value(x, y)``.
        - The string ``"free"`` to leave the boundary unconstrained.

    If different values are prescribed at a shared corner node, the value
    from the first boundary in the input dictionary is retained.
    """
    coordinates = np.asarray(coordinates, dtype=float)

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (n_nodes, 2).")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates must contain only finite values.")

    values_by_dof: dict[int, float] = {}
    source_by_dof: dict[int, str] = {}

    for boundary_name, boundary_value in boundary_values.items():

        if boundary_name not in boundaries:
            raise KeyError(
                f"Boundary {boundary_name!r} is not present in the mesh."
            )

        node_indices = np.asarray(
            boundaries[boundary_name], dtype=np.int64
        ).ravel()

        if np.any(node_indices < 0) or np.any(node_indices >= len(coordinates)):
            raise ValueError(
                f"Boundary {boundary_name!r} contains invalid node indices."
            )

        # Skip boundaries without prescribed Dirichlet conditions.
        if boundary_value == "free":
            continue

        if isinstance(boundary_value, str):
            raise ValueError(
                f"Invalid value for boundary {boundary_name!r}. "
                "Use a constant, a callable, or 'free'."
            )

        for dof in node_indices:
            x, y = coordinates[dof]

            value = (
                float(boundary_value(float(x), float(y)))
                if callable(boundary_value)
                else float(boundary_value)
            )

            if not np.isfinite(value):
                raise ValueError(
                    "Dirichlet boundary values must be finite."
                )

            dof = int(dof)

            if dof in values_by_dof:
                if not np.isclose(
                    values_by_dof[dof], value, rtol=1e-10, atol=1e-12
                ):
                    warnings.warn(
                        f"Conflicting Dirichlet values at node {dof}, "
                        f"shared by boundaries {source_by_dof[dof]!r} "
                        f"and {boundary_name!r}. "
                        f"Keeping the value from {source_by_dof[dof]!r}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                continue

            values_by_dof[dof] = value
            source_by_dof[dof] = boundary_name

    dofs = np.asarray(sorted(values_by_dof), dtype=np.int64)
    values = np.asarray(
        [values_by_dof[int(dof)] for dof in dofs],
        dtype=float,
    )

    return dofs, values