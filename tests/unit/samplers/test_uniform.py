from unittest.mock import patch

import numpy as np
import pytest

from glide.samplers import UniformSampler


@pytest.fixture
def y_proxy() -> np.ndarray:
    return np.array([0.0, 1.0, 1.0, 0.0, 1.0])


@pytest.fixture
def sampler() -> UniformSampler:
    return UniformSampler()


# --- sample ---


def test_sample_delegates_validation(sampler, y_proxy):
    with (
        patch("glide.samplers.uniform._validate_is_integer") as mock_validate_is_integer,
        patch("glide.samplers.uniform._validate_strictly_positive") as mock_validate_strictly_positive,
        patch("glide.samplers.uniform._validate_n_samples_bound") as mock_validate_n_samples_bound,
    ):
        sampler.sample(y_proxy, n_samples=1, random_seed=0)
    mock_validate_is_integer.assert_called_once_with(1, "n_samples")
    mock_validate_strictly_positive.assert_called_once_with(1, "n_samples")
    mock_validate_n_samples_bound.assert_called_once_with(1, 5)


def test_sample_valid_output(sampler, y_proxy):
    xi = sampler.sample(y_proxy, n_samples=1, random_seed=0)
    expected_xi = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(xi, expected_xi)


def test_sample_2d_y_proxy(sampler):
    y_proxies = np.array([[0.0, 1.0], [1.0, 1.0]])
    xi = sampler.sample(y_proxies, n_samples=1, random_seed=0)
    expected_xi = np.array([0.0, 1.0])
    np.testing.assert_array_equal(xi, expected_xi)


def test_sample_is_reproducible(sampler, y_proxy):
    xi1 = sampler.sample(y_proxy, n_samples=2, random_seed=42)
    xi2 = sampler.sample(y_proxy, n_samples=2, random_seed=42)
    np.testing.assert_array_equal(xi1, xi2)


def test_sample_different_seeds_results_differ(sampler, y_proxy):
    xi1 = sampler.sample(y_proxy, n_samples=2, random_seed=0)
    xi2 = sampler.sample(y_proxy, n_samples=2, random_seed=1)
    assert not np.array_equal(xi1, xi2)
