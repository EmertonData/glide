import numpy as np
import pytest

from glide.core.dataset import Dataset
from glide.samplers.stratified import StratifiedSampler


@pytest.fixture
def sampler() -> StratifiedSampler:
    return StratifiedSampler()


@pytest.fixture
def dataset() -> Dataset:
    return Dataset(
        [
            {"group": "A", "y_proxy": 0.60},
            {"group": "A", "y_proxy": 0.45},
            {"group": "B", "y_proxy": 0.67},
            {"group": "B", "y_proxy": 0.33},
        ]
    )


@pytest.fixture
def dataset_with_varied(dataset) -> Dataset:
    extended_records = list(dataset)
    extended_records.extend(
        [
            {"group": "A", "y_proxy": 0.50},
            {"group": "A", "y_proxy": 0.55},
            {"group": "B", "y_proxy": 0.0},
            {"group": "B", "y_proxy": 1.0},
        ]
    )
    return Dataset(extended_records)


# --- _preprocess ---


def test_preprocess_groups_preserved(sampler, dataset):
    _, groups = sampler._preprocess(dataset, "y_proxy", "group")

    assert np.array_equal(groups, np.array(["A", "A", "B", "B"], dtype=object))


def test_preprocess_raises_on_empty_dataset(sampler):
    dataset = Dataset([])

    with pytest.raises(ValueError, match="empty"):
        sampler._preprocess(dataset, "y_proxy", "group")


def test_preprocess_raises_on_unknown_y_proxy_field(sampler):
    dataset = Dataset([{"group": "A", "y_proxy": 0.5}])

    with pytest.raises(ValueError):
        sampler._preprocess(dataset, "nonexistent_field", "group")


def test_preprocess_raises_on_unknown_groups_field(sampler):
    dataset = Dataset([{"group": "A", "y_proxy": 0.5}])

    with pytest.raises((ValueError, KeyError)):
        sampler._preprocess(dataset, "y_proxy", "nonexistent_group_field")


def test_preprocess_raises_on_stratum_size_less_than_two(sampler):
    dataset = Dataset([{"group": "A", "y_proxy": 0.5}, {"group": "A", "y_proxy": 0.6}, {"group": "B", "y_proxy": 0.7}])

    with pytest.raises(ValueError, match="fewer than 2"):
        sampler._preprocess(dataset, "y_proxy", "group")


def test_preprocess_raises_on_nan_proxy(sampler):
    dataset = Dataset([{"group": "A", "y_proxy": 0.5}, {"group": "A", "y_proxy": np.nan}])

    with pytest.raises(ValueError, match="NaN"):
        sampler._preprocess(dataset, "y_proxy", "group")


def test_preprocess_raises_on_zero_variance(sampler):
    dataset = Dataset([{"group": "A", "y_proxy": 5.0}, {"group": "B", "y_proxy": 5.0}])

    with pytest.raises(ValueError, match="Input proxy values have zero variance"):
        sampler._preprocess(dataset, "y_proxy", "group")


def test_preprocess_raises_on_all_strata_zero_variance(sampler):
    dataset = Dataset(
        [
            {"group": "A", "y_proxy": 0},
            {"group": "A", "y_proxy": 0},
            {"group": "B", "y_proxy": 1},
            {"group": "B", "y_proxy": 1},
        ]
    )

    with pytest.raises(ValueError, match="has zero variance in proxy"):
        sampler._preprocess(dataset, "y_proxy", "group")


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


def test_sample_returns_dataset_with_valid_properties(sampler, dataset):
    result = sampler.sample(dataset, "y_proxy", "group", 2, random_seed=0)

    assert isinstance(result, Dataset)
    assert len(result) == len(dataset)
    assert all("pi" in record and "xi" in record for record in result)
    pi_values = result["pi"]
    assert np.all(pi_values > 0)
    assert np.all(pi_values <= 1)
    xi_values = result["xi"]
    assert set(xi_values).issubset({0.0, 1.0})


def test_sample_invalid_budget_zero(sampler, dataset):
    with pytest.raises(ValueError):
        sampler.sample(dataset, "y_proxy", "group", 0)


def test_sample_invalid_budget_negative(sampler, dataset):
    with pytest.raises(ValueError):
        sampler.sample(dataset, "y_proxy", "group", -1)


def test_sample_invalid_budget_float(sampler, dataset):
    with pytest.raises((ValueError, TypeError)):
        sampler.sample(dataset, "y_proxy", "group", 1.5)


def test_sample_invalid_budget_boolean(sampler, dataset):
    with pytest.raises((ValueError, TypeError)):
        sampler.sample(dataset, "y_proxy", "group", True)


def test_sample_budget_exceeds_dataset_length(sampler, dataset):
    with pytest.raises(ValueError):
        sampler.sample(dataset, "y_proxy", "group", len(dataset) + 1)


def test_sample_default_strategy_is_neyman(sampler, dataset_with_varied):
    budget = 8

    default_result = sampler.sample(dataset_with_varied, "y_proxy", "group", budget)
    neyman_result = sampler.sample(dataset_with_varied, "y_proxy", "group", budget, strategy="neyman")

    assert [r["pi"] for r in default_result] == [r["pi"] for r in neyman_result]


def test_sample_neyman_strategy(sampler, dataset_with_varied):
    result = sampler.sample(dataset_with_varied, "y_proxy", "group", 8, strategy="neyman")

    pi_a = result[0]["pi"]  # Group A
    pi_b = result[2]["pi"]  # Group B
    assert pi_b > pi_a


def test_sample_invalid_strategy_raises(sampler, dataset):
    with pytest.raises(ValueError, match="Unknown strategy"):
        sampler.sample(dataset, "y_proxy", "group", 2, strategy="unknown")


def test_sample_is_reproducible(sampler, dataset):
    result1 = sampler.sample(dataset, "y_proxy", "group", 2, random_seed=42)
    result2 = sampler.sample(dataset, "y_proxy", "group", 2, random_seed=42)

    np.testing.assert_array_equal(result1["xi"], result2["xi"])


def test_sample_seed_defaults_to_none_without_exception(sampler, dataset):
    result = sampler.sample(dataset, "y_proxy", "group", 2)

    assert isinstance(result, Dataset)
