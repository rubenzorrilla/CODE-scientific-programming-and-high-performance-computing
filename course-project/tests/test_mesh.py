import numpy as np
import pytest

from mesh import generate_mesh


@pytest.mark.parametrize(
    "element_type,nodes_per_element,elements_per_cell",
    [
        ("P1", 3, 2),
        ("Q1", 4, 1),
    ],
)
def test_structured_rectangle_mesh(
    element_type, nodes_per_element, elements_per_cell
):
    nx, ny = 4, 3
    coordinates, elements, boundaries = generate_mesh(
        length_x=2.0,
        length_y=1.0,
        num_elements_x=nx,
        num_elements_y=ny,
        element_type=element_type,
    )

    assert coordinates.shape == ((nx + 1) * (ny + 1), 2)
    assert elements.shape == (nx * ny * elements_per_cell, nodes_per_element)
    assert np.all(coordinates[:, 0] >= -1e-12)
    assert np.all(coordinates[:, 0] <= 2.0 + 1e-12)
    assert np.all(coordinates[:, 1] >= -1e-12)
    assert np.all(coordinates[:, 1] <= 1.0 + 1e-12)

    assert set(boundaries) == {"left", "right", "bottom", "top"}
    assert len(boundaries["left"]) == ny + 1
    assert len(boundaries["right"]) == ny + 1
    assert len(boundaries["bottom"]) == nx + 1
    assert len(boundaries["top"]) == nx + 1


def test_invalid_element_type():
    with pytest.raises(ValueError):
        generate_mesh(element_type="P2")


def test_invalid_dimensions():
    with pytest.raises(ValueError):
        generate_mesh(length_x=0.0)
