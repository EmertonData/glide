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
    with pytest.raises(ValueError, match="non-positive"):
        sampler._preprocess(zero_dataset, "uncertainty")


def test_preprocess_raises_on_negative_uncertainty(sampler):
    negative_dataset = Dataset([{"uncertainty": -0.1}, {"uncertainty": 0.5}])
    with pytest.raises(ValueError, match="non-positive"):
        sampler._preprocess(negative_dataset, "uncertainty")


def test_preprocess_raises_on_nan_uncertainty(sampler):
    nan_dataset = Dataset([{"uncertainty": float("nan")}, {"uncertainty": 0.5}])
    with pytest.raises(ValueError, match="NaN or absent"):
        sampler._preprocess(nan_dataset, "uncertainty")


def test_preprocess_raises_on_absent_uncertainty_value(sampler):
    absent_dataset = Dataset([{"uncertainty": 0.5}, {"score": 0.3}])
    with pytest.raises(ValueError, match="NaN or absent"):
        sampler._preprocess(absent_dataset, "uncertainty")


def test_preprocess_raises_on_unknown_field(sampler, dataset):
    with pytest.raises(ValueError, match="Unknown fields"):
        sampler._preprocess(dataset, "nonexistent")


# --- sample ---


def test_sample_returns_dataset_with_valid_properties(sampler, dataset):
    result = sampler.sample(dataset, uncertainty_field="uncertainty", budget=5, random_seed=0)
    assert isinstance(result, Dataset)
    assert len(result) == len(dataset)
    assert all("pi" in record and "xi" in record for record in result)
    pi_values = result["pi"]
    assert np.all(pi_values > 0)
    assert np.all(pi_values <= 1)
    xi_values = result["xi"]
    assert set(xi_values).issubset({0.0, 1.0})


def test_sample_pi_clipped_and_higher_uncertainty_gets_higher_pi(sampler):
    skewed_dataset = Dataset([{"uncertainty": 0.001}, {"uncertainty": 10.0}])
    result = sampler.sample(skewed_dataset, uncertainty_field="uncertainty", budget=2, random_seed=0)
    pi_values = result["pi"]
    assert pi_values[1] == pytest.approx(1.0, abs=0.001)
    assert pi_values[0] == pytest.approx(0.0, abs=0.001)


def test_sample_invalid_budget_zero(sampler, dataset):
    with pytest.raises(ValueError, match="budget"):
        sampler.sample(dataset, uncertainty_field="uncertainty", budget=0, random_seed=0)


def test_sample_invalid_boolean_budget(sampler, dataset):
    with pytest.raises(ValueError, match="budget"):
        sampler.sample(dataset, uncertainty_field="uncertainty", budget=True, random_seed=0)


def test_sample_invalid_budget_negative(sampler, dataset):
    with pytest.raises(ValueError, match="budget"):
        sampler.sample(dataset, uncertainty_field="uncertainty", budget=-1, random_seed=0)


def test_sample_invalid_budget_float(sampler, dataset):
    with pytest.raises(ValueError, match="budget"):
        sampler.sample(dataset, uncertainty_field="uncertainty", budget=1.5, random_seed=0)  # type: ignore[arg-type]


def test_sample_custom_field_names(sampler, dataset):
    result = sampler.sample(
        dataset, uncertainty_field="uncertainty", budget=1, random_seed=0, pi_field="my_pi", xi_field="my_xi"
    )
    assert all("my_pi" in record and "my_xi" in record for record in result)


def test_sample_is_reproducible(sampler, dataset):
    result1 = sampler.sample(dataset, uncertainty_field="uncertainty", budget=5, random_seed=42)
    result2 = sampler.sample(dataset, uncertainty_field="uncertainty", budget=5, random_seed=42)
    np.testing.assert_array_equal(result1["xi"], result2["xi"])


def test_sample_seed_defaults_to_none(sampler, dataset):
    # seed is optional — calling without it must not raise
    result = sampler.sample(dataset, uncertainty_field="uncertainty", budget=5)
    assert isinstance(result, Dataset)


def test_sample_budget_exceeds_dataset_length(sampler, dataset):
    with pytest.raises(ValueError, match="budget"):
        sampler.sample(dataset, uncertainty_field="uncertainty", budget=len(dataset) + 1, random_seed=0)
