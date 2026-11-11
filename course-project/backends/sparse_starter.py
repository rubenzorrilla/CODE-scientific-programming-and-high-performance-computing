"""Sparse backend starter: student implementation of the global sparse workflow.

The functions below are intentionally unimplemented. Fill them in while keeping
the same interface as ``backends.dense_reference``.
"""

from __future__ import annotations

from numpy.typing import ArrayLike, NDArray


def allocate_global_system(number_of_nodes: int):
    """Allocate a sparse global LHS and a dense RHS."""
    raise NotImplementedError(
        "Sparse starter: allocate a sparse matrix for the LHS and a dense "
        "vector for the RHS."
    )


def assemble_global_system(
    global_lhs: ArrayLike,
    global_rhs: ArrayLike,
    element_lhs: ArrayLike,
    element_rhs: ArrayLike,
    elements: ArrayLike,
):
    """Scatter local element arrays into the sparse global system."""
    raise NotImplementedError(
        "Sparse starter: loop over elements and local nodes, accumulating "
        "values into the sparse matrix and dense RHS."
    )


def impose_dirichlet_conditions(
    global_lhs: ArrayLike,
    global_rhs: ArrayLike,
    constrained_dofs: ArrayLike,
    prescribed_values: ArrayLike,
):
    """Impose Dirichlet values on a sparse system without mutating it."""
    raise NotImplementedError(
        "Sparse starter: move constrained-column contributions into the RHS, "
        "then replace constrained rows/columns with identity rows/columns."
    )


def solve_linear_system(global_lhs: ArrayLike, global_rhs: ArrayLike):
    """Solve the sparse linear system."""
    raise NotImplementedError(
        "Sparse starter: solve the sparse system with a sparse linear solver."
    )


def visualize(
    coordinates: ArrayLike,
    elements: ArrayLike,
    solution: ArrayLike,
    scalar_name: str = "u",
    show_edges: bool = True,
    screenshot: str | None = None,
):
    """Visualize the nodal solution."""
    raise NotImplementedError(
        "Sparse starter: create a mesh visualization and display or save it."
    )
