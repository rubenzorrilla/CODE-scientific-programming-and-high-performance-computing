"""Sparse GPU backend starter: student implementation of a sparse GPU workflow.

The functions below are intentionally unimplemented. Fill them in while keeping
the same interface as ``backends.dense_reference``.
"""

from __future__ import annotations

from numpy.typing import ArrayLike, NDArray


def allocate_global_system(number_of_nodes: int):
    """Allocate a sparse global LHS and dense RHS on a GPU-capable backend."""
    raise NotImplementedError(
        "Sparse GPU starter: allocate a sparse matrix and a dense RHS on your "
        "GPU library."
    )


def assemble_global_system(
    global_lhs: ArrayLike,
    global_rhs: ArrayLike,
    element_lhs: ArrayLike,
    element_rhs: ArrayLike,
    elements: ArrayLike,
):
    """Scatter local element arrays into the sparse GPU global system."""
    raise NotImplementedError(
        "Sparse GPU starter: loop over elements and local nodes, accumulating "
        "values into the sparse matrix and dense RHS on the GPU."
    )


def impose_dirichlet_conditions(
    global_lhs: ArrayLike,
    global_rhs: ArrayLike,
    constrained_dofs: ArrayLike,
    prescribed_values: ArrayLike,
):
    """Impose Dirichlet values on the sparse GPU system without mutating it."""
    raise NotImplementedError(
        "Sparse GPU starter: move constrained-column contributions into the RHS, "
        "then replace constrained rows/columns with identity rows/columns."
    )


def solve_linear_system(global_lhs: ArrayLike, global_rhs: ArrayLike):
    """Solve the sparse GPU linear system."""
    raise NotImplementedError(
        "Sparse GPU starter: solve the sparse system with your GPU sparse solver."
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
        "Sparse GPU starter: create a mesh visualization and display or save it."
    )
