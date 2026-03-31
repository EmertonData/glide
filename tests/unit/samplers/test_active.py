import numpy as np
import pytest

from glide.core.dataset import Dataset
from glide.samplers.active import ActiveSampler


@pytest.fixture
def dataset() -> Dataset:
    return Dataset([{"uncertainty": round(i * 0.1, 1)} for i in range(1, 10)])


@pytest.fixture
def sampler() -> ActiveSampler:
    return ActiveSampler()


# --- _preprocess ---


def test_preprocess_extracts_uncertainties(sampler, dataset):
    uncertainties = sampler._preprocess(dataset, "uncertainty")
    expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    np.testing.assert_array_almost_equal(uncertainties, expected)


def test_preprocess_raises_on_zero_uncertainty(sampler):
    zero_dataset = Dataset([{"uncertainty": 0.0}, {"uncertainty": 0.5}])
    with pytest.raises(ValueError, match="zero"):
        sampler._preprocess(zero_dataset, "uncertainty")


def test_preprocess_raises_on_unknown_field(sampler, dataset):
    with pytest.raises(ValueError, match="Unknown fields"):
        sampler._preprocess(dataset, "nonexistent")


# --- sample ---


def test_sample_returns_valid_result(sampler, dataset):
    result = sampler.sample(dataset, uncertainty_field="uncertainty", budget=5, seed=0)
    # returns a dataset
    assert isinstance(result, Dataset)
    # same length
    assert len(result) == len(dataset)
    # all records have "pi" and "xi" fields
    assert all("pi" in record and "xi" in record for record in result)
    # valid probabilities
    pi_values = result["pi"]
    assert np.all(pi_values > 0)
    assert np.all(pi_values <= 1)
    # xi is binary
    xi_values = result["xi"]
    assert set(xi_values).issubset({0.0, 1.0})


def test_sample_pi_equal_uncertainty_gives_equal_probabilities(sampler):
    # uncertainties=[1.0, 1.0], weights=[1, 1], budget=1 → pi=[0.5, 0.5]
    equal_dataset = Dataset([{"uncertainty": 1.0}, {"uncertainty": 1.0}])
    result = sampler.sample(equal_dataset, uncertainty_field="uncertainty", budget=1, seed=0)
    pi_values = result["pi"]
    np.testing.assert_array_almost_equal(pi_values, [0.5, 0.5])


def test_sample_higher_uncertainty_gets_higher_pi(sampler):
    # uncertainties=[0.25, 1.0], weights=[0.25, 1.0], sum=1.25, budget=1 → pi=[0.2, 0.8]
    skewed_dataset = Dataset([{"uncertainty": 0.25}, {"uncertainty": 1.0}])
    result = sampler.sample(skewed_dataset, uncertainty_field="uncertainty", budget=1, seed=0)
    pi_values = result["pi"]
    assert pi_values[1] > pi_values[0]
    np.testing.assert_array_almost_equal(pi_values, [0.2, 0.8])


def test_sample_pi_clipped_at_one_when_budget_is_large(sampler):
    # uncertainties=[0.1, 10.0], weights=[0.1, 10.0], sum=10.1, budget=10
    # raw pi=[0.1/10.1*10, 10/10.1*10] ≈ [0.099, 9.9] → clipped to [0.099, 1.0]
    skewed_dataset = Dataset([{"uncertainty": 0.1}, {"uncertainty": 10.0}])
    result = sampler.sample(skewed_dataset, uncertainty_field="uncertainty", budget=10, seed=0)
    pi_values = result["pi"]
    assert pi_values[1] == pytest.approx(1.0)
    assert pi_values[0] < 1.0


def test_sample_invalid_budget_zero(sampler, dataset):
    with pytest.raises(ValueError, match="budget"):
        sampler.sample(dataset, uncertainty_field="uncertainty", budget=0, seed=0)


def test_sample_invalid_budget_negative(sampler, dataset):
    with pytest.raises(ValueError, match="budget"):
        sampler.sample(dataset, uncertainty_field="uncertainty", budget=-1, seed=0)


def test_sample_invalid_budget_float(sampler, dataset):
    with pytest.raises(ValueError, match="budget"):
        sampler.sample(dataset, uncertainty_field="uncertainty", budget=1.5, seed=0)  # type: ignore[arg-type]


def test_sample_custom_field_names(sampler, dataset):
    result = sampler.sample(
        dataset, uncertainty_field="uncertainty", budget=1, seed=0, pi_field="my_pi", xi_field="my_xi"
    )
    assert all("my_pi" in record and "my_xi" in record for record in result)


def test_sample_is_reproducible(sampler, dataset):
    result1 = sampler.sample(dataset, uncertainty_field="uncertainty", budget=5, seed=42)
    result2 = sampler.sample(dataset, uncertainty_field="uncertainty", budget=5, seed=42)
    np.testing.assert_array_equal(result1["xi"], result2["xi"])
