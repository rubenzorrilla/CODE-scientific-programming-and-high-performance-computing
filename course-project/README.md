# Course Project — Scientific Computing and HPC

This project is part of the [Master's Degree in Computational and Data-Assisted Engineering (CODE)](https://camins.upc.edu/en/Studies/master/code) at the Universitat Politècnica de Catalunya (UPC).

## Overview

The project solves the Laplace equation on a structured square mesh using the finite element method (FEM). It is designed to teach scientific Python, global assembly, boundary-condition imposition, linear solution strategies, visualization, and performance analysis. The FEM derivation is treated as a black box.

## Current workflow

1. **Mesh generation:** Create a structured mesh with [Gmsh](https://gmsh.info/), using P1 triangles or Q1 quadrilaterals.
2. **Element calculations:** Compute each element's Laplacian matrix and negative residual.
3. **Global assembly:** Accumulate element contributions into a global dense Jacobian and residual vector. This implementation is deliberately naive: it uses dense storage and explicit Python loops. It is the baseline students will profile and optimize.
4. **Boundary conditions:** Collect prescribed values from named boundaries and impose Dirichlet conditions on the dense system.
5. **Linear solve:** Solve the correction system with NumPy's dense direct solver.
6. **Visualization:** Display the computed nodal solution with [PyVista](https://pyvista.org/).

## Project structure

```text
course-project/
├── README.md
├── requirements.txt
├── main.py
├── mesh.py
├── laplacian_element.py
├── assembly.py
├── boundary_conditions.py
├── solver.py
├── visualization.py
└── tests/
    ├── test_mesh.py
    ├── test_laplacian_element.py
    ├── test_assembly.py
    ├── test_boundary_conditions.py
    ├── test_solver.py
    └── test_visualization.py
```

## Setup

Activate your course virtual environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the solver

Run commands from the `course-project/` directory.

```bash
python main.py
```

The default configuration uses P1 triangular elements. Use Q1 quadrilaterals instead:

```bash
python main.py --element-type Q1
```

Change the mesh resolution:

```bash
python main.py --element-type P1 --nx 40 --ny 30
```

Save an image instead of opening an interactive window:

```bash
python main.py --element-type Q1 --nx 20 --ny 20 --screenshot results/solution.png
```

The example solves the homogeneous Laplace equation on the unit square. Dirichlet data are zero on the left, right, and bottom boundaries, and

\[
u(x,1)=\sin(\pi x)
\]

on the top boundary. The corresponding analytical solution is

\[
u(x,y)=rac{\sin(\pi x)\sinh(\pi y)}{\sinh(\pi)}.
\]

The main script reports the free-DOF residual norm and the maximum nodal error relative to this analytical solution.

## Residual convention

For the homogeneous Laplace equation, each element contributes

\[
\mathbf{R}_e(\mathbf{u}_e)=\mathbf{K}_e\mathbf{u}_e,
\qquad
\mathbf{J}_e=rac{\partial\mathbf{R}_e}{\partial\mathbf{u}_e}=\mathbf{K}_e.
\]

`assembly.py` assembles the global Jacobian and the negative residual, \(-\mathbf{R}(\mathbf{u})\). The linear system therefore solves for a correction:

\[
\mathbf{J}\,\Delta\mathbf{u}=-\mathbf{R}(\mathbf{u}).
\]

The initial assembly deliberately uses a dense NumPy matrix and nested Python loops. Since a finite element matrix is sparse, most stored entries are zero. Students are expected to profile this baseline and improve its memory use and execution time. The dense solver also has cubic computational cost in the number of unknowns.

The prescribed values imposed on the correction are

\[
\Delta u_D = u_D^{\mathrm{prescribed}} - u_D^{\mathrm{current}}.
\]

Starting from a zero field, one linear correction gives the solution of this linear problem.

## Running the tests

Tests are located in the `tests/` directory. From the `course-project/` directory, with the course virtual environment activated, run:

```bash
python -m pytest -v
```

`python -m pytest` discovers the tests under `tests/` and runs them. The module form also ensures that the current project directory is on Python's module search path, so no manual `PYTHONPATH` setting or packaging configuration is required.
