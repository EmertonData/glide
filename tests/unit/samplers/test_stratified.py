from unittest.mock import patch

import numpy as np
import pytest

import glide.samplers.stratified as stratified_module
from glide.samplers import StratifiedSampler


@pytest.fixture
def sampler() -> StratifiedSampler:
    return StratifiedSampler()


@pytest.fixture
def y_proxy() -> np.ndarray:
    return np.array([0.60, 0.1, 0.50, 0.55, 0.67, 0.33, 0.0, 0.7])


@pytest.fixture
def groups() -> np.ndarray:
    return np.array(["A", "A", "A", "A", "B", "B", "B", "B"], dtype=object)


# --- _proportional_allocation ---


def test_proportional_allocation_proportional_to_N_h(sampler):
    groups = np.array(["A", "A", "B", "B"], dtype=object)
    budget = 4
    total = len(groups)

    allocation = sampler._proportional_allocation(groups, budget)

    expected_ratio = budget / total
    for stratum_id, n_h in allocation.items():
        N_h = (groups == stratum_id).sum()
        actual_ratio = n_h / N_h
        assert actual_ratio == pytest.approx(expected_ratio, abs=0.01)


# --- sample ---


def test_sample_returns_valid_array(sampler, y_proxy, groups):
    xi = sampler.sample(y_proxy, groups, 4, random_seed=0)
    assert isinstance(xi, np.ndarray)
    assert len(xi) == len(y_proxy)
    assert np.isin(xi, [0, 1]).all()


def test_sample_invalid_budget_zero(sampler, y_proxy, groups):
    with pytest.raises(ValueError):
        sampler.sample(y_proxy, groups, 0)


def test_sample_invalid_budget_negative(sampler, y_proxy, groups):
    with pytest.raises(ValueError):
        sampler.sample(y_proxy, groups, -1)


def test_sample_invalid_budget_float(sampler, y_proxy, groups):
    with pytest.raises((ValueError, TypeError)):
        sampler.sample(y_proxy, groups, 1.5)


def test_sample_invalid_budget_boolean(sampler, y_proxy, groups):
    with pytest.raises((ValueError, TypeError)):
        sampler.sample(y_proxy, groups, True)


def test_sample_budget_exceeds_dataset_length(sampler, y_proxy, groups):
    with pytest.raises(ValueError):
        sampler.sample(y_proxy, groups, len(y_proxy) + 1)


def test_sample_raises_on_zero_allocation(sampler, y_proxy, groups):
    with pytest.raises(ValueError, match="fewer than two allocations"):
        sampler.sample(y_proxy, groups, 2)


def test_sample_raises_on_over_allocation(sampler, y_proxy, groups):
    with pytest.raises(ValueError, match="has been over-allocated"):
        sampler.sample(y_proxy, groups, 7)


def test_sample_default_strategy_is_neyman(sampler, y_proxy, groups):
    default_xi = sampler.sample(y_proxy, groups, 4, random_seed=0)
    neyman_xi = sampler.sample(y_proxy, groups, 4, strategy="neyman", random_seed=0)

    np.testing.assert_array_equal(default_xi, neyman_xi)


def test_sample_proportional_strategy(sampler, y_proxy, groups):
    xi = sampler.sample(y_proxy, groups, 4, strategy="proportional", random_seed=0)

    expected_xi = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(xi, expected_xi)


def test_sample_neyman_strategy(sampler, y_proxy, groups):
    xi = sampler.sample(y_proxy, groups, 4, strategy="neyman", random_seed=0)

    expected_xi = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(xi, expected_xi)


def test_sample_invalid_strategy_raises(sampler, y_proxy, groups):
    with pytest.raises(ValueError, match="'strategy' must be"):
        sampler.sample(y_proxy, groups, 4, strategy="unknown")


def test_sample_is_reproducible(sampler, y_proxy, groups):
    xi1 = sampler.sample(y_proxy, groups, 4, random_seed=42)
    xi2 = sampler.sample(y_proxy, groups, 4, random_seed=42)

    np.testing.assert_array_equal(xi1, xi2)


def test_sample_delegates_to_validation(sampler, y_proxy, groups):
    with (
        patch.object(stratified_module, "_validate_is_integer") as mock_validate_is_integer,
        patch.object(stratified_module, "_validate_strictly_positive") as mock_validate_strictly_positive,
        patch.object(stratified_module, "_validate_budget_bound") as mock_validate_budget_bound,
        patch.object(stratified_module, "_validate_y_proxy") as mock_validate_y_proxy,
    ):
        sampler.sample(y_proxy, groups, 4, random_seed=0)

        mock_validate_is_integer.assert_called_once_with(4, "budget")
        mock_validate_strictly_positive.assert_called_once_with(4, "budget")
        mock_validate_budget_bound.assert_called_once_with(4, len(y_proxy))

        assert mock_validate_y_proxy.call_count == 3
        np.testing.assert_array_equal(mock_validate_y_proxy.call_args_list[0][0][0], y_proxy)
        np.testing.assert_array_equal(mock_validate_y_proxy.call_args_list[1][0][0], y_proxy[groups == "A"])
        assert mock_validate_y_proxy.call_args_list[1][0][1] == "A"
        np.testing.assert_array_equal(mock_validate_y_proxy.call_args_list[2][0][0], y_proxy[groups == "B"])
        assert mock_validate_y_proxy.call_args_list[2][0][1] == "B"
