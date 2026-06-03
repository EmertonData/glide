import numpy as np
import pytest

from glide.samplers.core import _build_output, _compute_cutoff_indices, _shuffle


@pytest.fixture
def pi() -> np.ndarray:
    return np.array([0.3, 0.6, 0.9])


@pytest.fixture
def xi() -> np.ndarray:
    return np.array([1.0, 0.0, 1.0])


@pytest.fixture
def order() -> np.ndarray:
    return np.array([2, 0, 1])


# --- _shuffle ---


def test_shuffle_applies_order(pi, xi):
    rng = np.random.default_rng(42)
    (pi_shuffled, xi_shuffled), order = _shuffle((pi, xi), rng)

    expected_order = np.array([2, 1, 0])
    np.testing.assert_array_equal(order, expected_order)
    np.testing.assert_allclose(pi_shuffled, pi[expected_order])
    np.testing.assert_allclose(xi_shuffled, xi[expected_order])


# --- _compute_cutoff_indices ---


def test_compute_cutoff_indices(order):
    cumulative_costs = np.array([1.0, 1.0, 2.0])
    kept = _compute_cutoff_indices(cumulative_costs, order, budget=1.0)
    np.testing.assert_array_equal(kept, order[:2])


# --- _build_output ---


def test_build_output(pi, xi, order):
    kept_indices = order[:2]
    pi_out, xi_out = _build_output(kept_indices, pi, xi)

    expected_pi = np.array([0.6, 0.0, 0.3])
    expected_xi = np.array([0.0, np.nan, 1.0])
    np.testing.assert_allclose(pi_out, expected_pi)
    np.testing.assert_array_equal(xi_out, expected_xi)
