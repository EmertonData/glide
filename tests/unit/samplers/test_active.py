from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

import glide.samplers.active as active_module
from glide.samplers import ActiveSampler


@pytest.fixture
def uncertainties() -> NDArray:
    return np.arange(1, 10) * 0.1


@pytest.fixture
def sampler() -> ActiveSampler:
    return ActiveSampler()


# --- sample ---


@pytest.mark.parametrize("bad_budget", [0.0, True, -1, 1.5])
def test_sample_invalid_budget(sampler, uncertainties, bad_budget):
    with pytest.raises(ValueError, match="budget"):
        sampler.sample(uncertainties, budget=bad_budget, random_seed=0)


def test_sample_budget_exceeds_length(sampler, uncertainties):
    with pytest.raises(ValueError, match="budget"):
        sampler.sample(uncertainties, budget=len(uncertainties) + 1, random_seed=0)


def test_sample_valid_output(sampler, uncertainties):
    pi, xi = sampler.sample(uncertainties, budget=5, random_seed=42)
    expected_pi = np.array(
        [0.11111111, 0.22222222, 0.33333333, 0.44444444, 0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.0]
    )
    np.testing.assert_allclose(pi, expected_pi, atol=1e-10)
    expected_xi = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(xi, expected_xi)


def test_sample_pi_clipped_and_higher_uncertainty_gets_higher_pi(sampler):
    pi, _ = sampler.sample(np.array([0.001, 10.0]), budget=2, random_seed=0)
    np.testing.assert_allclose(pi, np.array([0.0, 1.0]), atol=0.001)


def test_sample_is_reproducible(sampler, uncertainties):
    pi1, xi1 = sampler.sample(uncertainties, budget=5, random_seed=42)
    pi2, xi2 = sampler.sample(uncertainties, budget=5, random_seed=42)
    np.testing.assert_array_equal(pi1, pi2)
    np.testing.assert_array_equal(xi1, xi2)


def test_sample_delegates_to_validation(sampler, uncertainties):
    with (
        patch.object(active_module, "_validate_is_integer") as mock_validate_is_integer,
        patch.object(active_module, "_validate_strictly_positive") as mock_validate_strictly_positive,
        patch.object(active_module, "_validate_budget_bound") as mock_validate_budget_bound,
        patch.object(active_module, "_validate_uncertainties") as mock_validate_uncertainties,
    ):
        sampler.sample(uncertainties, budget=5, random_seed=0)

        mock_validate_is_integer.assert_called_once_with(5, "budget")
        mock_validate_strictly_positive.assert_called_once_with(5, "budget")
        mock_validate_budget_bound.assert_called_once_with(5, len(uncertainties))
        mock_validate_uncertainties.assert_called_once()
        np.testing.assert_array_equal(mock_validate_uncertainties.call_args[0][0], uncertainties)
