"""Dense GPU backend starter: student implementation of a dense GPU workflow.

The functions below are intentionally unimplemented. Fill them in while keeping
the same interface as ``backends.dense_reference``.
"""

from __future__ import annotations

from numpy.typing import ArrayLike, NDArray


def allocate_global_system(number_of_nodes: int):
    """Allocate a dense global LHS and RHS on a GPU-capable backend."""
    raise NotImplementedError(
        "Dense GPU starter: allocate dense arrays using your GPU array library."
    )


def assemble_global_system(
    global_lhs: ArrayLike,
    global_rhs: ArrayLike,
    element_lhs: ArrayLike,
    element_rhs: ArrayLike,
    elements: ArrayLike,
):
    """Scatter local element arrays into the dense GPU global system."""
    raise NotImplementedError(
        "Dense GPU starter: scatter local element contributions into the dense "
        "GPU arrays."
    )


def impose_dirichlet_conditions(
    global_lhs: ArrayLike,
    global_rhs: ArrayLike,
    constrained_dofs: ArrayLike,
    prescribed_values: ArrayLike,
):
    """Impose Dirichlet values on the dense GPU system without mutating it."""
    raise NotImplementedError(
        "Dense GPU starter: move constrained-column contributions into the RHS, "
        "then zero constrained rows/columns and set unit diagonal entries."
    )


def solve_linear_system(global_lhs: ArrayLike, global_rhs: ArrayLike):
    """Solve the dense GPU linear system."""
    raise NotImplementedError(
        "Dense GPU starter: solve the dense system with your GPU solver."
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
        "Dense GPU starter: create a mesh visualization and display or save it."
    )
