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
The initial implementation provides structured mesh generation for P1 triangles and Q1 quadrilaterals. Finite element assembly, boundary-condition treatment, and solver components will be introduced in subsequent steps.
