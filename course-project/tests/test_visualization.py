import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from visualization import create_visualization_grid


def test_create_triangle_grid_with_nodal_values():
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    elements = np.array([[0, 1, 2]])
    values = np.array([0.0, 1.0, 2.0])

    grid = create_visualization_grid(coordinates, elements, values, "temperature")

    assert grid.n_points == 3
    assert grid.n_cells == 1
    np.testing.assert_allclose(grid.point_data["temperature"], values)
    np.testing.assert_allclose(grid.points[:, 2], 0.0)
    assert grid.celltypes[0] == pv.CellType.TRIANGLE


def test_create_quad_grid():
    coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )
    elements = np.array([[0, 1, 2, 3]])
    values = np.array([0.0, 1.0, 2.0, 1.0])

    grid = create_visualization_grid(coordinates, elements, values)

    assert grid.n_points == 4
    assert grid.n_cells == 1
    assert grid.celltypes[0] == pv.CellType.QUAD
    np.testing.assert_allclose(grid.point_data["u"], values)


@pytest.mark.parametrize(
    "coordinates,elements,values",
    [
        (np.zeros((3, 3)), np.array([[0, 1, 2]]), np.zeros(3)),
        (np.zeros((3, 2)), np.array([[0, 1, 3]]), np.zeros(3)),
        (np.zeros((3, 2)), np.array([[0, 1, 2]]), np.zeros(2)),
    ],
)
def test_invalid_visualization_data(coordinates, elements, values):
    with pytest.raises(ValueError):
        create_visualization_grid(coordinates, elements, values)
