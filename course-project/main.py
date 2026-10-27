"""Main workflow: mesh, assembly, Dirichlet BCs, solve, and visualization."""

import argparse
from pathlib import Path

import numpy as np

from assembly import assemble_laplacian_system
from boundary_conditions import (
    apply_dirichlet_conditions,
    collect_dirichlet_conditions,
)
from mesh import generate_mesh
from solver import solve_linear_system
from visualization import plot_results


def exact_solution(x, y):
    """Analytical harmonic field used to define boundary data and check error."""
    return np.sin(np.pi * x) * np.sinh(np.pi * y) / np.sinh(np.pi)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve the Laplace equation on a structured square mesh."
    )
    parser.add_argument(
        "--element-type",
        choices=("P1", "Q1"),
        default="P1",
        help="Finite element type: P1 triangles or Q1 quadrilaterals.",
    )
    parser.add_argument("--nx", type=int, default=20, help="Elements along x.")
    parser.add_argument("--ny", type=int, default=20, help="Elements along y.")
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="Optional image path. If supplied, render off-screen.",
    )
    args = parser.parse_args()

    coordinates, elements, boundaries = generate_mesh(
        length_x=1.0,
        length_y=1.0,
        num_elements_x=args.nx,
        num_elements_y=args.ny,
        element_type=args.element_type,
    )

    n_nodes = coordinates.shape[0]
    solution = np.zeros(n_nodes, dtype=float)

    # Dirichlet conditions for u(x,y) = sin(pi*x)*sinh(pi*y)/sinh(pi):
    # u = 0 on left, right and bottom; u = sin(pi*x) on the top edge.
    boundary_values = {
        "left": 0.0,
        "right": 0.0,
        "bottom": 0.0,
        "top": lambda x, y: np.sin(np.pi * x),
    }
    constrained_dofs, prescribed_solution_values = collect_dirichlet_conditions(
        coordinates, boundaries, boundary_values
    )

    # Assemble the Jacobian and negative residual at the current iterate.
    # The chosen initial iterate is zero, so one linear correction solves the
    # linear Laplace problem exactly up to the linear solver tolerance.
    lhs, rhs = assemble_laplacian_system(coordinates, elements, solution)

    prescribed_corrections = prescribed_solution_values - solution[constrained_dofs]
    lhs_bc, rhs_bc = apply_dirichlet_conditions(
        lhs, rhs, constrained_dofs, prescribed_corrections
    )

    correction = solve_linear_system(lhs_bc, rhs_bc)
    solution += correction

    # The assembled RHS is -R(u); evaluate it again to measure the residual on
    # unconstrained nodes after the solve.
    _, final_negative_residual = assemble_laplacian_system(
        coordinates, elements, solution
    )
    free_mask = np.ones(n_nodes, dtype=bool)
    free_mask[constrained_dofs] = False
    free_residual_norm = np.linalg.norm(final_negative_residual[free_mask])

    exact_values = exact_solution(coordinates[:, 0], coordinates[:, 1])
    max_nodal_error = np.max(np.abs(solution - exact_values))

    print(f"Element type: {args.element_type}")
    print(f"Number of nodes: {n_nodes}")
    print(f"Number of elements: {elements.shape[0]}")
    print(f"Number of constrained DOFs: {constrained_dofs.size}")
    print(f"Free-DOF residual norm: {free_residual_norm:.6e}")
    print(f"Maximum nodal error against analytical solution: {max_nodal_error:.6e}")

    plot_results(
        coordinates,
        elements,
        solution,
        scalar_name="u",
        show_edges=True,
        screenshot=args.screenshot,
    )


if __name__ == "__main__":
    main()
