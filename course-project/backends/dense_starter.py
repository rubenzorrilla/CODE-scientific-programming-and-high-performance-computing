"""Dense backend starter: student implementation of the global dense workflow.

The functions below are intentionally unimplemented. Fill them in while keeping
the same interface as ``backends.dense_reference``.
"""

from __future__ import annotations

from numpy.typing import ArrayLike, NDArray


def allocate_global_system(number_of_nodes: int):
    """Allocate a dense global LHS and RHS."""
    raise NotImplementedError(
        "Dense starter: allocate a dense (n_nodes, n_nodes) LHS and a dense "
        "RHS of length n_nodes."
    )


def assemble_global_system(
    global_lhs: ArrayLike,
    global_rhs: ArrayLike,
    element_lhs: ArrayLike,
    element_rhs: ArrayLike,
    elements: ArrayLike,
):
    """Scatter local element arrays into the dense global system."""
    raise NotImplementedError(
        "Dense starter: loop over elements and local nodes, accumulate each "
        "element LHS/RHS into the global arrays."
    )


def impose_dirichlet_conditions(
    global_lhs: ArrayLike,
    global_rhs: ArrayLike,
    constrained_dofs: ArrayLike,
    prescribed_values: ArrayLike,
):
    """Impose Dirichlet values without mutating the input system."""
    raise NotImplementedError(
        "Dense starter: move constrained-column contributions into the RHS, "
        "then zero constrained rows and columns and set unit diagonal entries."
    )


def solve_linear_system(global_lhs: ArrayLike, global_rhs: ArrayLike):
    """Solve the dense linear system."""
    raise NotImplementedError(
        "Dense starter: solve the dense system with a direct dense solver."
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
        "Dense starter: create a mesh visualization and display or save it."
    )
