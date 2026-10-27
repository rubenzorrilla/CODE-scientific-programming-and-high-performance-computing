"""Visualize nodal scalar results on a P1/Q1 mesh using PyVista."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike


def create_visualization_grid(
    coordinates: ArrayLike,
    elements: ArrayLike,
    values: ArrayLike,
    scalar_name: str = "u",
):
    """Create a PyVista UnstructuredGrid with a nodal scalar field.

    Parameters
    ----------
    coordinates:
        Nodal coordinates with shape (n_nodes, 2).
    elements:
        Zero-based connectivity, shape (n_elements, 3) for P1 triangles
        or (n_elements, 4) for Q1 quadrilaterals.
    values:
        One scalar value per node.
    scalar_name:
        Name to assign to the scalar data.

    Returns
    -------
    pyvista.UnstructuredGrid
        Mesh with the scalar field attached to its points.
    """
    import pyvista as pv

    coordinates = np.asarray(coordinates, dtype=float)
    elements = np.asarray(elements, dtype=np.int64)
    values = np.asarray(values, dtype=float)

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (n_nodes, 2).")
    if coordinates.shape[0] == 0 or not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates must be non-empty and finite.")
    if elements.ndim != 2 or elements.shape[0] == 0:
        raise ValueError("elements must be a non-empty 2D connectivity array.")
    if elements.shape[1] not in (3, 4):
        raise ValueError("elements must describe P1 triangles or Q1 quadrilaterals.")
    if np.any(elements < 0) or np.any(elements >= coordinates.shape[0]):
        raise ValueError("elements contains node indices outside the coordinates array.")
    if values.shape != (coordinates.shape[0],):
        raise ValueError("values must contain exactly one scalar per mesh node.")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must contain only finite numbers.")
    if not scalar_name:
        raise ValueError("scalar_name must not be empty.")

    nodes_per_element = elements.shape[1]
    cell_type = pv.CellType.TRIANGLE if nodes_per_element == 3 else pv.CellType.QUAD

    # VTK cell arrays use [number_of_nodes, node_0, node_1, ...] for each cell.
    cells = np.column_stack(
        (np.full(elements.shape[0], nodes_per_element, dtype=np.int64), elements)
    ).ravel()
    cell_types = np.full(elements.shape[0], cell_type, dtype=np.uint8)

    # The finite element mesh is 2D; VTK points have three coordinates.
    points = np.column_stack((coordinates, np.zeros(coordinates.shape[0])))

    grid = pv.UnstructuredGrid(cells, cell_types, points)
    grid.point_data[scalar_name] = values
    return grid


def plot_results(
    coordinates: ArrayLike,
    elements: ArrayLike,
    values: ArrayLike,
    scalar_name: str = "u",
    show_edges: bool = True,
    screenshot: str | Path | None = None,
    window_size: tuple[int, int] = (1100, 800),
):
    """Plot a nodal scalar field over a P1/Q1 mesh.

    With no ``screenshot`` path, an interactive window is opened. If a path
    is provided, the plot is rendered off-screen and saved there.

    Returns the PyVista grid for optional further inspection or export.
    """
    import pyvista as pv

    grid = create_visualization_grid(coordinates, elements, values, scalar_name)

    screenshot_path = None
    if screenshot is not None:
        screenshot_path = Path(screenshot)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(
        window_size=window_size,
        off_screen=screenshot_path is not None,
    )
    plotter.add_mesh(
        grid,
        scalars=scalar_name,
        cmap="viridis",
        show_edges=show_edges,
        edge_color="black",
        scalar_bar_args={"title": scalar_name, "vertical": True},
    )
    plotter.add_axes()
    plotter.view_xy()
    plotter.reset_camera()

    if screenshot_path is None:
        plotter.show()
    else:
        plotter.show(screenshot=str(screenshot_path), auto_close=True)

    return grid
