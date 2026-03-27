import pytest

from glide.core.dataset import Dataset
from glide.core.simulated_datasets import generate_binary_dataset, generate_binary_dataset_with_oracle_sampling


def test_generate_binary_dataset_structure_and_counts():
    labeled, unlabeled = generate_binary_dataset(n=1, N=2, random_seed=0)
    dataset = labeled + unlabeled
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 3
    assert len(labeled) == 1
    assert len(unlabeled) == 2
    for record in dataset:
        assert "y_proxy" in record
        assert record["y_proxy"] in (0, 1)
        assert record.get("y_true", 0) in (0, 1)


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
    dataset1 = generate_binary_dataset(n=1, N=2, random_seed=7)
    dataset2 = generate_binary_dataset(n=1, N=2, random_seed=7)
    assert dataset1 == dataset2


def test_generate_binary_dataset_with_oracle_sampling_structure_and_counts():
    dataset = generate_binary_dataset_with_oracle_sampling(N=10, random_seed=0)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 10
    for record in dataset:
        assert "y_true" in record
        assert "y_proxy" in record
        assert "pi" in record
        assert record["y_true"] in (0, 1)
        assert record["y_proxy"] in (0, 1)
        assert record["pi"] > 0


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
    dataset1 = generate_binary_dataset_with_oracle_sampling(N=10, random_seed=7)
    dataset2 = generate_binary_dataset_with_oracle_sampling(N=10, random_seed=7)
    assert dataset1 == dataset2
