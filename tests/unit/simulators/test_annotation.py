import numpy as np
import pytest

from glide.simulators import simulate_annotation


@pytest.fixture
def y_true_oracle():
    return np.array([0.0, 1.0])


def test_simulate_annotation_annotated_positions_preserved(y_true_oracle):
    xi = np.array([1, 1])
    result = simulate_annotation(y_true_oracle, xi)
    np.testing.assert_array_equal(result, y_true_oracle)


def test_simulate_annotation_unannotated_positions_are_nan(y_true_oracle):
    xi = np.array([0, 0])
    result = simulate_annotation(y_true_oracle, xi)
    assert np.all(np.isnan(result))


def test_simulate_annotation_inputs_not_mutated(y_true_oracle):
    xi = np.array([0, 1])
    original = y_true_oracle.copy()
    simulate_annotation(y_true_oracle, xi)
    np.testing.assert_array_equal(y_true_oracle, original)


def test_simulate_annotation_empty_arrays():
    result = simulate_annotation(np.array([]), np.array([]))
    assert len(result) == 0
