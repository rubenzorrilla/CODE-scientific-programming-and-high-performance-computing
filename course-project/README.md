# Course Project — Scientific Computing and HPC

This project is part of the [Master's Degree in Computational and Data-Assisted Engineering (CODE)](https://camins.upc.edu/en/Studies/master/code) at the Universitat Politècnica de Catalunya (UPC).

## Overview

The project solves the Poisson equation

\[
\nabla^2 u = f(x,y)
\]

on a structured square mesh using the finite element method (FEM). When \(f=0\), this is the homogeneous Laplace equation. The project is designed to teach scientific Python, global assembly, boundary-condition imposition, linear solution strategies, visualization, and performance analysis. Mesh generation and local element calculations are provided as a black box; students implement the backend global operations. The FEM derivation is treated as a black box.

## Current workflow

1. **Mesh generation:** Create a structured mesh with [Gmsh](https://gmsh.info/), using P1 triangles or Q1 quadrilaterals.
2. **Local element calculations:** Compute each element's stiffness matrix \(K_e\) and mass-weighted forcing vector \(F_e=M_e f_e\). This produces the local element systems \(K_e u_e = F_e\) as multidimensional arrays.
3. **Global allocation:** A selected backend allocates the global matrix and RHS.
4. **Global assembly:** The same backend scatters the local element contributions into the global system.
5. **Boundary conditions:** Prescribed Dirichlet values are collected from named boundaries. A boundary may also be marked `"free"` to leave it unconstrained.
6. **BC imposition:** The backend imposes those Dirichlet values on the global system.
7. **Linear solve:** The backend solves the constrained system \(K u = F\).
8. **Visualization:** The backend displays the computed nodal solution with [PyVista](https://pyvista.org/).

## Project structure

```text
course-project/
├── README.md
├── requirements.txt
├── main.py
├── mesh.py
├── laplacian_element.py
├── assembly_helper.py
├── boundary_conditions.py
├── visualization.py
├── backends/
│   ├── __init__.py
│   ├── dense_reference.py
│   ├── dense_starter.py
│   ├── sparse_starter.py
│   ├── dense_gpu_starter.py
│   └── sparse_gpu_starter.py
└── tests/
    ├── test_mesh.py
    ├── test_laplacian_element.py
    ├── test_assembly_helper.py
    ├── test_boundary_conditions.py
    ├── test_backends.py
    ├── test_main.py
    └── test_visualization.py
```

## Backend interface

Each backend module is called from `main.py` and should expose:

```python
allocate_global_system(number_of_nodes)
assemble_global_system(global_lhs, global_rhs, element_lhs, element_rhs, elements)
impose_dirichlet_conditions(global_lhs, global_rhs, constrained_dofs, prescribed_values)
solve_linear_system(global_lhs, global_rhs)
visualize(coordinates, elements, solution, scalar_name="u", show_edges=True, screenshot=None)
```

The local arrays produced by `assembly_helper.build_element_systems` have shapes:

- `element_lhs`: `(n_elements, n_local, n_local)`
- `element_rhs`: `(n_elements, n_local)`

`backends/dense_reference.py` is a private dense reference implementation. The starter modules are boiler plates:

- `backends/dense_starter.py`
- `backends/sparse_starter.py`
- `backends/dense_gpu_starter.py`
- `backends/sparse_gpu_starter.py`

These starter modules intentionally raise `NotImplementedError` until students implement the global operations.

`impose_dirichlet_conditions` should return a new system and leave the original input system unchanged, because `main.py` uses the original system for a PDE residual check.

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

The default backend is the dense reference implementation:

```bash
python main.py --backend backends.dense_reference
```

Select a student starter backend:

```bash
python main.py --backend backends.dense_starter
```

Use Q1 quadrilaterals:

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

The default example solves the homogeneous Laplace equation, \(f=0\), on the unit square. Dirichlet data are zero on the left, right, and bottom boundaries, and

\[
u(x,1)=\sin(\pi x)
\]

on the top boundary. The corresponding analytical solution is

\[
u(x,y)=\frac{\sin(\pi x)\sinh(\pi y)}{\sinh(\pi)}.
\]

The main script always reports the free-DOF residual norm. It reports the maximum nodal error relative to a reference solution only when `reference_solution` is supplied and agrees with the prescribed Dirichlet values.

## Forcing field

`assembly_helper.build_element_systems` accepts a forcing field as a constant, a nodal array, or a function \(f(x,y)\):

```python
forcing = lambda x, y: x + 2.0 * y
element_lhs, element_rhs = build_element_systems(
    coordinates, elements, forcing=forcing
)
```

The forcing field is evaluated at mesh nodes and mass-weighted in each element:

\[
F_e = M_e f_e.
\]

When `forcing = 0`, the system reduces to the homogeneous Laplace case. If forcing is nonzero, the default analytical solution in `main.py` no longer applies, so error should be checked against a corresponding known solution or through the PDE residual.

## Changing the example

The default problem in `main.py` is defined by three values:

```python
boundary_values = {
    "left": 0.0,
    "right": 0.0,
    "bottom": 0.0,
    "top": lambda x, y: np.sin(np.pi * x),
}
forcing = 0.0
reference_solution = exact_solution
```

`run_workflow` accepts these as arguments:

```python
run_workflow(
    backend,
    element_type,
    nx,
    ny,
    screenshot,
    boundary_values,
    forcing=0.0,
    reference_solution=None,
)
```

To test another case:

- Change `boundary_values` to change the Dirichlet data.
- Set a boundary value to `"free"` to leave that boundary unconstrained.
- Change `forcing` to change the source term. `build_element_systems` accepts a constant, a nodal array, or a function `f(x, y)`.
- Change `reference_solution` to use a different analytical field.
- Set `reference_solution = None` when the reference field does not match the PDE or boundary data.

If `reference_solution` is supplied but does not match the prescribed boundary values, the workflow raises a `ValueError`. If the forcing term is nonzero, the default `exact_solution` is not valid unless it is changed accordingly, and the workflow prints a warning about the reference-error comparison.

The PDE residual check remains valid for any correctly configured problem. The reference-error value is only meaningful when the reference solution actually corresponds to the chosen PDE and boundary data.

## Linear system

The global system is

\[
K u = F.
\]

After imposing Dirichlet conditions, the constrained system is

\[
K_{\mathrm{bc}} u = F_{\mathrm{bc}}.
\]

The values passed to the backend's `impose_dirichlet_conditions` are prescribed solution values, not corrections.

## Running the tests

Tests are located in the `tests/` directory. From the `course-project/` directory, with the course virtual environment activated, run:

```bash
python -m pytest -v
```

`python -m pytest` discovers the tests under `tests/` and runs them. The module form also ensures that the current project directory is on Python's module search path, so no manual `PYTHONPATH` setting or packaging configuration is required.
