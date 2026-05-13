import numpy as np
import pytest

from glide.simulators import simulate_annotation


@pytest.fixture
def y_true_oracle():
    return np.array([0.0, 1.0, 1.0, 0.0])


@pytest.fixture
def xi():
    return np.array([1, 0, 1, 0])


def test_simulate_annotation_length_mismatch(y_true_oracle):
    xi = np.array([1, 0])
    with pytest.raises(ValueError, match="must have the same length"):
        simulate_annotation(y_true_oracle, xi)


def test_simulate_annotation_y_true_oracle_contains_nan():
    y_true_oracle = np.array([0.0, np.nan])
    xi = np.array([1, 0])
    with pytest.raises(ValueError, match="y_true_oracle contains NaN"):
        simulate_annotation(y_true_oracle, xi)


@pytest.mark.parametrize("bad_xi", [np.nan, 2])
def test_simulate_annotation_xi_non_binary(bad_xi):
    y_true_oracle = np.array([0.0, 1.0])
    xi = np.array([1.0, bad_xi])
    with pytest.raises(ValueError, match="xi must only contain 0 and 1 values"):
        simulate_annotation(y_true_oracle, xi)


def test_simulate_annotation_mixed_annotated_unannotated(y_true_oracle, xi):
    result = simulate_annotation(y_true_oracle, xi)
    assert result.dtype == float
    expected = np.array([0.0, np.nan, 1.0, np.nan])
    np.testing.assert_array_equal(result, expected)


def test_simulate_annotation_empty_arrays():
    result = simulate_annotation(np.array([]), np.array([]))
    assert len(result) == 0
