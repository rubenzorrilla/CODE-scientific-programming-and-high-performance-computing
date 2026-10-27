"""Dense linear solution strategy for the initial course-project baseline."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def solve_linear_system(lhs: ArrayLike, rhs: ArrayLike) -> NDArray[np.float64]:
    """Solve a dense linear system using ``numpy.linalg.solve``.

    This intentionally simple direct solver is suitable for small baseline
    cases. The dense matrix and cubic solution cost provide clear targets for
    later performance and sparse-linear-algebra work.
    """
    matrix = np.asarray(lhs, dtype=float)
    vector = np.asarray(rhs, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("lhs must be a square matrix.")
    if vector.shape != (matrix.shape[0],):
        raise ValueError("rhs must contain one value per matrix row.")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(vector)):
        raise ValueError("The linear system must contain only finite values.")

    try:
        solution = np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError as error:
        raise RuntimeError(
            "The dense linear system could not be solved. "
            "Check that sufficient boundary conditions have been imposed."
        ) from error

    if not np.all(np.isfinite(solution)):
        raise RuntimeError("The solver returned non-finite values.")
    return np.asarray(solution, dtype=float)
