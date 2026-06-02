import numpy as np
import pytest

from glide.samplers.core import _apply_budget_cutoff, _draw_shuffled_bernoulli


@pytest.fixture
def pi() -> np.ndarray:
    return np.array([0.3, 0.6, 0.9])


@pytest.fixture
def order() -> np.ndarray:
    return np.array([2, 0, 1])


@pytest.fixture
def xi_shuffled() -> np.ndarray:
    return np.array([1.0, 0.0, 1.0])


@pytest.fixture
def cumulative_costs() -> np.ndarray:
    return np.array([3.0, 5.0, 8.0])


@pytest.fixture
def budget() -> float:
    return 6.0


def test_draw_shuffled_bernoulli(pi):
    order, xi_shuffled = _draw_shuffled_bernoulli(pi, random_seed=42)

    expected_order = [2, 1, 0]
    expected_xi_shuffled = [1.0, 0.0, 0.0]

    np.testing.assert_array_equal(order, expected_order)
    np.testing.assert_array_equal(xi_shuffled, expected_xi_shuffled)


def test_apply_budget_cutoff(pi, order, xi_shuffled, cumulative_costs, budget):
    pi_out, xi_out = _apply_budget_cutoff(xi_shuffled, pi, cumulative_costs, order, budget)

    expected_pi = [0.3, 0.0, 0.9]
    expected_xi = [0.0, np.nan, 1.0]

    np.testing.assert_allclose(pi_out, expected_pi)
    np.testing.assert_array_equal(xi_out, expected_xi)
