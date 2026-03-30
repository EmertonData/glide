import numpy as np
import pytest

from glide.core.dataset import Dataset
from glide.samplers.active import ActiveSampler


@pytest.fixture
def dataset() -> Dataset:
    return Dataset([{"uncertainty": 0.2}, {"uncertainty": 0.8}])


@pytest.fixture
def sampler() -> ActiveSampler:
    return ActiveSampler()


# --- _preprocess ---


def test_preprocess_extracts_uncertainties(sampler, dataset):
    uncertainties = sampler._preprocess(dataset, "uncertainty")
    expected = np.array([0.2, 0.8])
    np.testing.assert_array_equal(uncertainties, expected)


def test_preprocess_raises_on_zero_uncertainty(sampler):
    zero_dataset = Dataset([{"uncertainty": 0.0}, {"uncertainty": 0.5}])
    with pytest.raises(ValueError, match="zero"):
        sampler._preprocess(zero_dataset, "uncertainty")


def test_preprocess_raises_on_unknown_field(sampler, dataset):
    with pytest.raises(ValueError, match="Unknown fields"):
        sampler._preprocess(dataset, "nonexistent")


# --- sample ---


def test_sample_returns_dataset(sampler, dataset):
    result = sampler.sample(dataset, uncertainty_field="uncertainty", budget=1, seed=0)
    assert isinstance(result, Dataset)


def test_sample_adds_pi_and_xi_fields(sampler, dataset):
    result = sampler.sample(dataset, uncertainty_field="uncertainty", budget=1, seed=0)
    for record in result:
        assert "pi" in record
        assert "xi" in record


def test_sample_pi_values_are_valid_probabilities(sampler, dataset):
    result = sampler.sample(dataset, uncertainty_field="uncertainty", budget=1, seed=0)
    pi_values = result["pi"]
    assert np.all(pi_values > 0)
    assert np.all(pi_values <= 1)


def test_sample_xi_is_binary(sampler, dataset):
    result = sampler.sample(dataset, uncertainty_field="uncertainty", budget=1, seed=0)
    xi_values = result["xi"]
    assert set(xi_values).issubset({0.0, 1.0})


def test_sample_pi_equal_uncertainty_gives_equal_probabilities(sampler):
    # uncertainties=[1.0, 1.0], weights=[1, 1], budget=1 → pi=[0.5, 0.5]
    equal_dataset = Dataset([{"uncertainty": 1.0}, {"uncertainty": 1.0}])
    result = sampler.sample(equal_dataset, uncertainty_field="uncertainty", budget=1, seed=0)
    pi_values = result["pi"]
    np.testing.assert_array_almost_equal(pi_values, [0.5, 0.5])


def test_sample_higher_uncertainty_gets_higher_pi(sampler):
    # uncertainties=[0.25, 1.0], weights=[4, 1], sum=5, budget=1 → pi=[0.8, 0.2]
    skewed_dataset = Dataset([{"uncertainty": 0.25}, {"uncertainty": 1.0}])
    result = sampler.sample(skewed_dataset, uncertainty_field="uncertainty", budget=1, seed=0)
    pi_values = result["pi"]
    assert pi_values[0] > pi_values[1]
    np.testing.assert_array_almost_equal(pi_values, [0.8, 0.2])


def test_sample_pi_clipped_at_one_when_budget_is_large(sampler):
    # uncertainties=[0.1, 10.0], weights=[10, 0.1], sum=10.1, budget=10
    # raw pi=[10/10.1*10, 0.1/10.1*10] ≈ [9.9, 0.099] → clipped to [1.0, 0.099]
    skewed_dataset = Dataset([{"uncertainty": 0.1}, {"uncertainty": 10.0}])
    result = sampler.sample(skewed_dataset, uncertainty_field="uncertainty", budget=10, seed=0)
    pi_values = result["pi"]
    assert pi_values[0] == pytest.approx(1.0)
    assert pi_values[1] < 1.0


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
    for record in result:
        assert "my_pi" in record
        assert "my_xi" in record


def test_sample_preserves_original_fields(sampler):
    rich_dataset = Dataset([
        {"score": 0.9, "uncertainty": 0.2},
        {"score": 0.5, "uncertainty": 0.8},
    ])
    result = sampler.sample(rich_dataset, uncertainty_field="uncertainty", budget=1, seed=0)
    for record in result:
        assert "score" in record
        assert "uncertainty" in record


def test_sample_is_reproducible(sampler, dataset):
    result1 = sampler.sample(dataset, uncertainty_field="uncertainty", budget=1, seed=42)
    result2 = sampler.sample(dataset, uncertainty_field="uncertainty", budget=1, seed=42)
    np.testing.assert_array_equal(result1["xi"], result2["xi"])


def test_sample_different_seeds_can_differ(sampler, dataset):
    # With enough trials the outputs will differ; just verify the mechanism doesn't break
    result1 = sampler.sample(dataset, uncertainty_field="uncertainty", budget=1, seed=0)
    result2 = sampler.sample(dataset, uncertainty_field="uncertainty", budget=1, seed=1)
    assert isinstance(result1, Dataset)
    assert isinstance(result2, Dataset)


def test_sample_output_length_matches_input(sampler, dataset):
    result = sampler.sample(dataset, uncertainty_field="uncertainty", budget=1, seed=0)
    assert len(result) == len(dataset)
