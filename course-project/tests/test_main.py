import numpy as np
import pytest

from main import exact_solution, is_zero_forcing, compute_reference_error


def test_reference_error_is_zero_when_reference_matches_solution():
    coordinates = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    reference_values = exact_solution(coordinates[:, 0], coordinates[:, 1])

    error = compute_reference_error(
        reference_values,
        coordinates,
        np.array([0, 2]),
        reference_values[[0, 2]],
        exact_solution,
    )

    assert error == 0.0


def test_reference_error_rejects_mismatched_boundary_values():
    coordinates = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    reference_values = exact_solution(coordinates[:, 0], coordinates[:, 1])

    with pytest.raises(
        ValueError,
        match="reference_solution does not match the prescribed boundary values"
    ):
        compute_reference_error(
            reference_values,
            coordinates,
            np.array([0, 1]),
            np.array([0.0, 0.5]),
            exact_solution,
        )


def test_zero_forcing_detection():
    assert is_zero_forcing(0.0)
    assert is_zero_forcing(np.zeros(3))
    assert not is_zero_forcing(1.0)
    assert not is_zero_forcing(lambda x, y: 1.0)
