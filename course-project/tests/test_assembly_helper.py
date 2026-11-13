import numpy as np

from assembly_helper import collect_element_systems, _evaluate_forcing_values


def square_triangular_mesh():
    coordinates = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    return coordinates, elements


def test_local_element_systems_have_expected_shapes():
    coordinates, elements = square_triangular_mesh()
    local_lhs, local_rhs = collect_element_systems(coordinates, elements)

    assert local_lhs.shape == (2, 3, 3)
    assert local_rhs.shape == (2, 3)
    np.testing.assert_allclose(local_rhs, 0.0, atol=1e-12)
    for element_lhs in local_lhs:
        np.testing.assert_allclose(element_lhs, element_lhs.T, atol=1e-12)
        np.testing.assert_allclose(element_lhs @ np.ones(3), 0.0, atol=1e-12)


def test_scalar_forcing_is_evaluated_at_every_node():
    coordinates, elements = square_triangular_mesh()
    local_lhs, local_rhs = collect_element_systems(
        coordinates, elements, forcing=1.0
    )

    assert local_lhs.shape == (2, 3, 3)
    assert local_rhs.shape == (2, 3)
    assert np.all(local_rhs > 0.0)


def test_forcing_function_is_evaluated_at_mesh_nodes():
    coordinates, elements = square_triangular_mesh()
    forcing = lambda x, y: x + y
    values = _evaluate_forcing_values(coordinates, forcing)

    assert values.shape == (coordinates.shape[0],)
    np.testing.assert_allclose(values, [0.0, 1.0, 1.0, 2.0])

    local_lhs, local_rhs = collect_element_systems(
        coordinates, elements, forcing=forcing
    )
    assert local_lhs.shape == (2, 3, 3)
    assert local_rhs.shape == (2, 3)
