"""Structured mesh generation with the Gmsh Python API."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import gmsh
import numpy as np
from numpy.typing import NDArray

ElementType = Literal["P1", "Q1"]


def generate_mesh(
    length_x: float = 1.0,
    length_y: float = 1.0,
    num_elements_x: int = 10,
    num_elements_y: int = 10,
    element_type: ElementType = "P1",
    output_file: str | Path | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64], dict[str, NDArray[np.int64]]]:
    """Generate a structured rectangular mesh using Gmsh.

    Parameters
    ----------
    length_x, length_y:
        Side lengths of the rectangular domain. Both must be positive.
    num_elements_x, num_elements_y:
        Number of elements along the horizontal and vertical directions.
    element_type:
        ``"P1"`` for linear triangles or ``"Q1"`` for bilinear quadrilaterals.
    output_file:
        Optional path at which to save the generated mesh in Gmsh's native format.

    Returns
    -------
    coordinates:
        Array of nodal coordinates with shape ``(n_nodes, 2)``.
    elements:
        Element connectivity with shape ``(n_elements, 3)`` for P1 or
        ``(n_elements, 4)`` for Q1. Indices are zero-based.
    boundaries:
        Dictionary mapping ``"left"``, ``"right"``, ``"bottom"``, and ``"top"``
        to arrays of zero-based node indices.

    Notes
    -----
    Gmsh node tags are mapped explicitly to contiguous zero-based indices.
    The mesh is structured using transfinite meshing constraints.
    """
    if length_x <= 0.0 or length_y <= 0.0:
        raise ValueError("length_x and length_y must be positive.")
    if num_elements_x < 1 or num_elements_y < 1:
        raise ValueError("num_elements_x and num_elements_y must be at least 1.")
    if element_type not in ("P1", "Q1"):
        raise ValueError("element_type must be either 'P1' or 'Q1'.")

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("structured_rectangle")

        # Counter-clockwise corner points.
        p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
        p2 = gmsh.model.geo.addPoint(length_x, 0.0, 0.0)
        p3 = gmsh.model.geo.addPoint(length_x, length_y, 0.0)
        p4 = gmsh.model.geo.addPoint(0.0, length_y, 0.0)

        bottom = gmsh.model.geo.addLine(p1, p2)
        right = gmsh.model.geo.addLine(p2, p3)
        top = gmsh.model.geo.addLine(p3, p4)
        left = gmsh.model.geo.addLine(p4, p1)

        loop = gmsh.model.geo.addCurveLoop([bottom, right, top, left])
        surface = gmsh.model.geo.addPlaneSurface([loop])

        gmsh.model.geo.synchronize()

        # The argument is the number of nodes, hence elements + 1.
        gmsh.model.mesh.setTransfiniteCurve(bottom, num_elements_x + 1)
        gmsh.model.mesh.setTransfiniteCurve(top, num_elements_x + 1)
        gmsh.model.mesh.setTransfiniteCurve(left, num_elements_y + 1)
        gmsh.model.mesh.setTransfiniteCurve(right, num_elements_y + 1)
        gmsh.model.mesh.setTransfiniteSurface(surface)

        if element_type == "Q1":
            gmsh.model.mesh.setRecombine(2, surface)

        # Physical groups give the boundaries stable, human-readable names.
        boundary_curves = {
            "bottom": bottom,
            "right": right,
            "top": top,
            "left": left,
        }
        for name, curve in boundary_curves.items():
            physical_tag = gmsh.model.addPhysicalGroup(1, [curve])
            gmsh.model.setPhysicalName(1, physical_tag, name)

        domain_tag = gmsh.model.addPhysicalGroup(2, [surface])
        gmsh.model.setPhysicalName(2, domain_tag, "domain")

        gmsh.model.mesh.generate(2)

        if output_file is not None:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(str(output_path))

        node_tags, node_coordinates, _ = gmsh.model.mesh.getNodes()
        node_coordinates = np.asarray(node_coordinates, dtype=float).reshape(-1, 3)
        coordinates = node_coordinates[:, :2]

        # Gmsh tags are identifiers, not array indices.
        tag_to_index = {int(tag): i for i, tag in enumerate(node_tags)}

        if element_type == "P1":
            gmsh_element_type = 2  # 3-node first-order triangle
            nodes_per_element = 3
        else:
            gmsh_element_type = 3  # 4-node first-order quadrilateral
            nodes_per_element = 4

        element_tags, element_node_tags = gmsh.model.mesh.getElementsByType(
            gmsh_element_type
        )
        if len(element_tags) == 0:
            raise RuntimeError(
                f"Gmsh generated no elements of the requested type {element_type}."
            )

        element_node_tags = np.asarray(element_node_tags, dtype=np.int64).reshape(
            -1, nodes_per_element
        )
        elements = np.empty_like(element_node_tags)
        for i, tag in enumerate(element_node_tags.flat):
            elements.flat[i] = tag_to_index[int(tag)]

        boundaries: dict[str, NDArray[np.int64]] = {}
        for name, curve in boundary_curves.items():
            boundary_node_tags, _, _ = gmsh.model.mesh.getNodes(1, curve, includeBoundary=True)
            boundary_indices = np.array(
                sorted({tag_to_index[int(tag)] for tag in boundary_node_tags}),
                dtype=np.int64,
            )
            boundaries[name] = boundary_indices

        return (
            np.asarray(coordinates, dtype=np.float64),
            np.asarray(elements, dtype=np.int64),
            boundaries,
        )
    finally:
        gmsh.finalize()
