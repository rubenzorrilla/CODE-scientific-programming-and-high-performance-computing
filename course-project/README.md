# Course Project — Scientific Computing and HPC

This project is part of the [Master's Degree in Computational and Data-Assisted Engineering (CODE)](https://camins.upc.edu/en/Studies/master/code) at the Universitat Politècnica de Catalunya (UPC).

## Overview

The project explores the computational solution of the Laplace equation using the finite element method (FEM). The emphasis is on scientific programming, sparse linear algebra, performance optimization, and parallel computing rather than on deriving the FEM formulation.

## Project components

- **Mesh generation:** Generate structured rectangular meshes with [Gmsh](https://gmsh.info/).
- **Finite element assembly:** Assemble the residual and its Jacobian for linear triangular (P1) and bilinear quadrilateral (Q1) elements.
- **Sparse linear algebra:** Represent and solve the resulting sparse systems.
- **Performance analysis:** Measure computational cost, memory use, speedup, and parallel efficiency.

## Repository layout

```text
course-project/
├── README.md
├── requirements.txt
├── src/
│   └── laplace_fem/
│       ├── __init__.py
│       └── mesh.py
└── tests/
    └── test_mesh.py
```

## Setup

Python 3.10 or newer is recommended.

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate       # Windows

python -m pip install -r requirements.txt
```

## Generate a mesh

The mesh module generates a structured rectangular mesh and returns node coordinates, element connectivity, and boundary-node indices.

```python
from laplace_fem import generate_mesh

coordinates, elements, boundaries = generate_mesh(
    length_x=1.0,
    length_y=1.0,
    num_elements_x=10,
    num_elements_y=10,
    element_type="P1",
)
```

For quadrilateral elements, use `element_type="Q1"`.

The function uses zero-based indices in its returned connectivity and boundary arrays, regardless of the node tags used internally by Gmsh.

To run the example from the repository root without installing the package, set the source directory on `PYTHONPATH`:

```bash
PYTHONPATH=src python -c "from laplace_fem import generate_mesh; print(generate_mesh(1, 1, 4, 4)[1].shape)"
```

## Residual-based formulation

The finite element assembly module will use the residual convention

\[
\mathbf{R}(\mathbf{u}) = \mathbf{K}\mathbf{u} - \mathbf{f},
\qquad
\mathbf{J}(\mathbf{u}) = \frac{\partial \mathbf{R}}{\partial \mathbf{u}}.
\]

For the linear Laplace problem, the Jacobian is the stiffness matrix and does not depend on the current solution. This interface is intended to remain useful when the project is extended to nonlinear residuals.

## Development status

The initial implementation provides structured mesh generation for P1 triangles and Q1 quadrilaterals. Finite element assembly, boundary-condition treatment, and solver components will be introduced in subsequent steps.
