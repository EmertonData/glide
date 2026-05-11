import numpy as np
import pytest

from glide.samplers.uniform import UniformSampler


@pytest.fixture
def n_samples() -> int:
    return 5


@pytest.fixture
def sampler() -> UniformSampler:
    return UniformSampler()


# --- sample ---


@pytest.mark.parametrize("bad_n_samples", [0.0, True, -1, 1.5])
def test_sample_invalid_n_samples(sampler, bad_n_samples):
    with pytest.raises(ValueError, match="n_samples"):
        sampler.sample(n_samples=bad_n_samples, budget=1, random_seed=0)


@pytest.mark.parametrize("bad_budget", [0.0, True, -1, 1.5])
def test_sample_invalid_budget(sampler, n_samples, bad_budget):
    with pytest.raises(ValueError, match="budget"):
        sampler.sample(n_samples=n_samples, budget=bad_budget, random_seed=0)


def test_sample_budget_exceeds_n_samples(sampler, n_samples):
    with pytest.raises(ValueError, match="budget"):
        sampler.sample(n_samples=n_samples, budget=n_samples + 1, random_seed=0)


def test_sample_valid_output(sampler):
    xi = sampler.sample(n_samples=2, budget=1, random_seed=0)
    expected_xi = np.array([0.0, 1.0])
    np.testing.assert_array_equal(xi, expected_xi)


def test_sample_is_reproducible(sampler, n_samples):
    xi1 = sampler.sample(n_samples=n_samples, budget=2, random_seed=42)
    xi2 = sampler.sample(n_samples=n_samples, budget=2, random_seed=42)
    np.testing.assert_array_equal(xi1, xi2)
