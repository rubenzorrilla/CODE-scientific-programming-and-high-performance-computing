"""Main workflow: mesh, local element systems, backend global operations,
Dirichlet BCs, solve, and visualization."""

import argparse
import importlib
from pathlib import Path

import numpy as np

from assembly import assemble_element_systems
from boundary_conditions import collect_dirichlet_conditions
from mesh import generate_mesh


def exact_solution(x, y):
    """Analytical harmonic field used to define boundary data and check error."""
    return np.sin(np.pi * x) * np.sinh(np.pi * y) / np.sinh(np.pi)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve the Poisson equation on a structured square mesh."
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
        "--backend",
        default="backends.dense_reference",
        help=(
            "Backend module implementing global allocation, assembly, "
            "Dirichlet imposition, solving, and visualization."
        ),
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="Optional image path. If supplied, render off-screen.",
    )
    args = parser.parse_args()

    backend = importlib.import_module(args.backend)

    coordinates, elements, boundaries = generate_mesh(
        length_x=1.0,
        length_y=1.0,
        num_elements_x=args.nx,
        num_elements_y=args.ny,
        element_type=args.element_type,
    )

    # Dirichlet conditions for u(x,y) = sin(pi*x)*sinh(pi*y)/sinh(pi):
    # u = 0 on left, right and bottom; u = sin(pi*x) on the top edge.
    boundary_values = {
        "left": 0.0,
        "right": 0.0,
        "bottom": 0.0,
        "top": lambda x, y: np.sin(np.pi * x),
    }
    constrained_dofs, prescribed_values = collect_dirichlet_conditions(
        coordinates, boundaries, boundary_values
    )

    # The default example is homogeneous: forcing = 0.
    # To use a forcing field, replace this with a scalar, array, or f(x, y).
    forcing = 0.0

    # Local element systems: K_e u_e = F_e.
    element_lhs, element_rhs = assemble_element_systems(
        coordinates, elements, forcing=forcing
    )

    # The selected backend performs every global operation.
    global_lhs, global_rhs = backend.allocate_global_system(coordinates.shape[0])
    global_lhs, global_rhs = backend.assemble_global_system(
        global_lhs, global_rhs, element_lhs, element_rhs, elements
    )

    constrained_lhs, constrained_rhs = backend.impose_dirichlet_conditions(
        global_lhs, global_rhs, constrained_dofs, prescribed_values
    )

    solution = backend.solve_linear_system(constrained_lhs, constrained_rhs)

    # Residual check on the original PDE, excluding constrained nodes.
    residual = global_lhs @ solution - global_rhs
    free_mask = np.ones(coordinates.shape[0], dtype=bool)
    free_mask[constrained_dofs] = False
    free_residual_norm = np.linalg.norm(residual[free_mask])

    exact_values = exact_solution(coordinates[:, 0], coordinates[:, 1])
    max_nodal_error = np.max(np.abs(solution - exact_values))

    print(f"Backend: {args.backend}")
    print(f"Element type: {args.element_type}")
    print(f"Number of nodes: {coordinates.shape[0]}")
    print(f"Number of elements: {elements.shape[0]}")
    print(f"Number of constrained DOFs: {constrained_dofs.size}")
    print(f"Free-DOF residual norm: {free_residual_norm:.6e}")
    print(f"Maximum nodal error against analytical solution: {max_nodal_error:.6e}")

    backend.visualize(
        coordinates,
        elements,
        solution,
        scalar_name="u",
        show_edges=True,
        screenshot=args.screenshot,
    )


if __name__ == "__main__":
    main()
