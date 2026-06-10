from unittest.mock import call, patch

import numpy as np
import pytest

from glide.samplers import UniformSampler


@pytest.fixture
def n_total() -> int:
    return 5


@pytest.fixture
def sampler() -> UniformSampler:
    return UniformSampler()


# --- sample ---


def test_sample_delegates_validation(sampler):
    with (
        patch("glide.samplers.uniform._validate_is_integer") as mock_validate_is_integer,
        patch("glide.samplers.uniform._validate_strictly_positive") as mock_validate_strictly_positive,
    ):
        sampler.sample(n_total=2, n_samples=1, random_seed=0)
    mock_validate_is_integer.assert_has_calls([call(2, "n_total"), call(1, "n_samples")])
    mock_validate_strictly_positive.assert_has_calls([call(2, "n_total"), call(1, "n_samples")])


def test_sample_n_samples_exceeds_n_total(sampler, n_total):
    with pytest.raises(ValueError, match="n_samples"):
        sampler.sample(n_total=n_total, n_samples=n_total + 1, random_seed=0)


def test_sample_valid_output(sampler):
    xi = sampler.sample(n_total=2, n_samples=1, random_seed=0)
    expected_xi = np.array([0.0, 1.0])
    np.testing.assert_array_equal(xi, expected_xi)


def test_sample_is_reproducible(sampler, n_total):
    xi1 = sampler.sample(n_total=n_total, n_samples=2, random_seed=42)
    xi2 = sampler.sample(n_total=n_total, n_samples=2, random_seed=42)
    np.testing.assert_array_equal(xi1, xi2)


def test_sample_different_seeds_results_differ(sampler, n_total):
    xi1 = sampler.sample(n_total=n_total, n_samples=2, random_seed=0)
    xi2 = sampler.sample(n_total=n_total, n_samples=2, random_seed=1)
    assert not np.array_equal(xi1, xi2)
