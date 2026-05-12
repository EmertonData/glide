import numpy as np
import pytest

from glide.simulators import simulate_annotation


def test_simulate_annotation_annotated_positions_preserved():
    y_true_oracle = np.array([0.0, 1.0])
    xi = np.array([1, 1])
    result = simulate_annotation(y_true_oracle, xi)
    np.testing.assert_array_equal(result, y_true_oracle)


def test_simulate_annotation_unannotated_positions_are_nan():
    y_true_oracle = np.array([0.0, 1.0])
    xi = np.array([0, 0])
    result = simulate_annotation(y_true_oracle, xi)
    assert np.all(np.isnan(result))


def test_simulate_annotation_mixed():
    y_true_oracle = np.array([0, 1, 1, 0])
    xi = np.array([1, 0, 1, 0])
    result = simulate_annotation(y_true_oracle, xi)
    assert result[0] == pytest.approx(0.0, abs=0.01)
    assert np.isnan(result[1])
    assert result[2] == pytest.approx(1.0, abs=0.01)
    assert np.isnan(result[3])


def test_simulate_annotation_inputs_not_mutated():
    y_true_oracle = np.array([1.0, 0.0])
    xi = np.array([0, 1])
    original = y_true_oracle.copy()
    simulate_annotation(y_true_oracle, xi)
    np.testing.assert_array_equal(y_true_oracle, original)


def test_simulate_annotation_empty_arrays():
    result = simulate_annotation(np.array([]), np.array([]))
    assert len(result) == 0
