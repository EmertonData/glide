import numpy as np
import pytest

from glide.core.simulated_datasets import (
    generate_binary_dataset,
    generate_binary_dataset_with_oracle_sampling,
    generate_gaussian_dataset,
    generate_stratified_binary_dataset,
)


def test_generate_binary_dataset_structure_and_counts():
    y_true, y_proxy = generate_binary_dataset(n=1, N=2, random_seed=0)
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_proxy, np.ndarray)
    assert len(y_true) == 3
    assert len(y_proxy) == 3
    assert np.sum(~np.isnan(y_true)) == 1
    assert np.sum(~np.isnan(y_proxy)) == 3


def test_generate_binary_dataset_invalid_true_mean_raises():
    with pytest.raises(ValueError, match=r"true_mean must be in \(0, 1\), got 1\.5"):
        generate_binary_dataset(n=1, N=1, true_mean=1.5)


def test_generate_binary_dataset_invalid_proxy_mean_raises():
    with pytest.raises(ValueError, match=r"proxy_mean must be in \(0, 1\), got 0"):
        generate_binary_dataset(n=1, N=1, proxy_mean=0.0)


def test_generate_binary_dataset_impossible_correlation_raises():
    with pytest.raises(
        ValueError,
        match=r"Impossible combination of true_mean=0\.7, proxy_mean=0\.6, and correlation=0\.95",
    ):
        generate_binary_dataset(n=1, N=9, true_mean=0.7, proxy_mean=0.6, correlation=0.95)


def test_generate_binary_dataset_reproducibility():
    y_true1, y_proxy1 = generate_binary_dataset(n=1, N=2, random_seed=7)
    y_true2, y_proxy2 = generate_binary_dataset(n=1, N=2, random_seed=7)
    assert np.array_equal(y_true1, y_true2, equal_nan=True)
    assert np.array_equal(y_proxy1, y_proxy2, equal_nan=True)


def test_generate_binary_dataset_with_oracle_sampling_structure_and_counts():
    from glide.core.dataset import Dataset

    dataset = generate_binary_dataset_with_oracle_sampling(N=10, random_seed=0)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 10
    for record in dataset:
        assert "y_true" in record
        assert "y_proxy" in record
        assert "uncertainty" in record
        assert record["y_true"] in (0, 1)
        assert record["y_proxy"] in (0, 1)
        assert record["uncertainty"] > 0


def test_generate_binary_dataset_with_oracle_sampling_invalid_true_mean_raises():
    with pytest.raises(ValueError, match=r"true_mean must be in \(0, 1\), got 1\.5"):
        generate_binary_dataset_with_oracle_sampling(N=10, true_mean=1.5)


def test_generate_binary_dataset_with_oracle_sampling_invalid_proxy_mean_raises():
    with pytest.raises(ValueError, match=r"proxy_mean must be in \(0, 1\), got 0"):
        generate_binary_dataset_with_oracle_sampling(N=10, proxy_mean=0.0)


def test_generate_binary_dataset_with_oracle_sampling_impossible_correlation_raises():
    with pytest.raises(
        ValueError,
        match=r"Impossible combination of true_mean=0\.7, proxy_mean=0\.6, and correlation=0\.95",
    ):
        generate_binary_dataset_with_oracle_sampling(N=10, true_mean=0.7, proxy_mean=0.6, correlation=0.95)


def test_generate_binary_dataset_with_oracle_sampling_reproducibility():
    data1 = generate_binary_dataset_with_oracle_sampling(N=10, random_seed=7)
    data2 = generate_binary_dataset_with_oracle_sampling(N=10, random_seed=7)
    assert np.array_equal(data1, data2)


def test_generate_stratified_binary_dataset_structure_and_counts():
    y_true, y_proxy = generate_stratified_binary_dataset(
        n=[1, 2],
        N=[2, 1],
        true_mean=[0.6, 0.8],
        proxy_mean=[0.5, 0.7],
        correlation=[0.75, 0.75],
        random_seed=None,
    )
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_proxy, np.ndarray)
    assert len(y_true) == 6
    assert len(y_proxy) == 6
    assert np.sum(~np.isnan(y_true)) == 3  # 1 + 2 labeled samples
    assert np.sum(~np.isnan(y_proxy)) == 6  # all proxy samples present


def test_generate_stratified_binary_dataset_empty_strata_raises():
    with pytest.raises(ValueError, match=r"Number of strata must be at least 1, got 0"):
        generate_stratified_binary_dataset(n=[], N=[], true_mean=[], proxy_mean=[], correlation=[])


def test_generate_stratified_binary_dataset_mismatched_lists_raises():
    with pytest.raises(ValueError, match=r"All input lists must have the same length"):
        generate_stratified_binary_dataset(
            n=[1, 2],
            N=[2, 1, 3],
            true_mean=[0.5, 0.6],
            proxy_mean=[0.5, 0.6],
            correlation=[0.8, 0.8],
        )


def test_generate_stratified_binary_dataset_reproducibility():
    y_true1, y_proxy1 = generate_stratified_binary_dataset(
        n=[1, 2],
        N=[2, 1],
        true_mean=[0.6, 0.8],
        proxy_mean=[0.5, 0.7],
        correlation=[0.75, 0.75],
        random_seed=42,
    )
    y_true2, y_proxy2 = generate_stratified_binary_dataset(
        n=[1, 2],
        N=[2, 1],
        true_mean=[0.6, 0.8],
        proxy_mean=[0.5, 0.7],
        correlation=[0.75, 0.75],
        random_seed=42,
    )
    assert np.array_equal(y_true1, y_true2, equal_nan=True)
    assert np.array_equal(y_proxy1, y_proxy2, equal_nan=True)


def test_generate_gaussian_dataset_structure_and_counts():
    y_true, y_proxy = generate_gaussian_dataset(n=1, N=2, random_seed=0)
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_proxy, np.ndarray)
    assert len(y_true) == 3
    assert len(y_proxy) == 3
    assert np.sum(~np.isnan(y_true)) == 1
    assert np.sum(~np.isnan(y_proxy)) == 3
    assert np.isnan(y_true[1]) and np.isnan(y_true[2])  # unlabeled rows have NaN


def test_generate_gaussian_dataset_invalid_positive_correlation_raises():
    with pytest.raises(ValueError, match="Correlation should be between -1 and 1"):
        generate_gaussian_dataset(n=1, N=1, correlation=1.5)


def test_generate_gaussian_dataset_invalid_negative_correlation_raises():
    with pytest.raises(ValueError, match="Correlation should be between -1 and 1"):
        generate_gaussian_dataset(n=1, N=1, correlation=-1.5)


def test_generate_gaussian_dataset_reproducibility():
    y_true1, y_proxy1 = generate_gaussian_dataset(n=1, N=2, random_seed=7)
    y_true2, y_proxy2 = generate_gaussian_dataset(n=1, N=2, random_seed=7)
    assert np.array_equal(y_true1, y_true2, equal_nan=True)
    assert np.array_equal(y_proxy1, y_proxy2, equal_nan=True)
