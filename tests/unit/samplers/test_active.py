import numpy as np
import pytest
from numpy.typing import NDArray

from glide.samplers.active import ActiveSampler


@pytest.fixture
def uncertainties() -> NDArray:
    return np.array([i * 0.1 for i in range(1, 10)])


@pytest.fixture
def sampler() -> ActiveSampler:
    return ActiveSampler()


# --- _preprocess ---


def test_preprocess_extracts_uncertainties(sampler, uncertainties):
    result = sampler._preprocess(uncertainties)
    expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    np.testing.assert_array_almost_equal(result, expected)


def test_preprocess_raises_on_zero_uncertainty(sampler):
    with pytest.raises(ValueError, match="non-positive"):
        sampler._preprocess(np.array([0.0, 0.5]))


def test_preprocess_raises_on_negative_uncertainty(sampler):
    with pytest.raises(ValueError, match="non-positive"):
        sampler._preprocess(np.array([-0.1, 0.5]))


def test_preprocess_raises_on_nan_uncertainty(sampler):
    with pytest.raises(ValueError, match="NaN"):
        sampler._preprocess(np.array([float("nan"), 0.5]))


# --- sample ---


def test_sample_returns_valid_arrays(sampler, uncertainties):
    pi, xi = sampler.sample(uncertainties, budget=5, random_seed=0)
    assert len(pi) == len(uncertainties)
    assert len(xi) == len(uncertainties)
    assert np.all(pi > 0)
    assert np.all(pi <= 1)
    assert set(xi.tolist()).issubset({0.0, 1.0})


def test_sample_pi_clipped_and_higher_uncertainty_gets_higher_pi(sampler):
    pi, _ = sampler.sample(np.array([0.001, 10.0]), budget=2, random_seed=0)
    assert pi[0] == pytest.approx(0.0, abs=0.001)
    assert pi[1] == pytest.approx(1.0, abs=0.001)


def test_sample_invalid_budget_zero(sampler, uncertainties):
    with pytest.raises(ValueError, match="budget"):
        sampler.sample(uncertainties, budget=0, random_seed=0)


def test_sample_invalid_boolean_budget(sampler, uncertainties):
    with pytest.raises(ValueError, match="budget"):
        sampler.sample(uncertainties, budget=True, random_seed=0)


def test_sample_invalid_budget_negative(sampler, uncertainties):
    with pytest.raises(ValueError, match="budget"):
        sampler.sample(uncertainties, budget=-1, random_seed=0)


def test_sample_invalid_budget_float(sampler, uncertainties):
    with pytest.raises(ValueError, match="budget"):
        sampler.sample(uncertainties, budget=1.5, random_seed=0)


def test_sample_is_reproducible(sampler, uncertainties):
    pi1, xi1 = sampler.sample(uncertainties, budget=5, random_seed=42)
    pi2, xi2 = sampler.sample(uncertainties, budget=5, random_seed=42)
    np.testing.assert_array_equal(xi1, xi2)


def test_sample_seed_defaults_to_none_without_exception(sampler, uncertainties):
    pi, xi = sampler.sample(uncertainties, budget=5)
    assert len(pi) == len(uncertainties)


def test_sample_budget_exceeds_length(sampler, uncertainties):
    with pytest.raises(ValueError, match="budget"):
        sampler.sample(uncertainties, budget=len(uncertainties) + 1, random_seed=0)
