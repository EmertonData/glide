import pytest

from glide.core.dataset import Dataset
from glide.core.simulated_datasets import generate_binary_dataset, generate_gaussian_dataset


def test_generate_binary_dataset_structure_and_counts():
    ds = generate_binary_dataset(n=1, N=2, random_seed=0)
    labeled = [r for r in ds if "y_true" in r]
    unlabeled = [r for r in ds if "y_true" not in r]
    assert isinstance(ds, Dataset)
    assert len(ds) == 3
    assert len(labeled) == 1
    assert len(unlabeled) == 2
    for record in ds:
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
    ds1 = generate_binary_dataset(n=1, N=2, random_seed=7)
    ds2 = generate_binary_dataset(n=1, N=2, random_seed=7)
    assert ds1 == ds2


def test_generate_gaussian_dataset_structure_and_counts():
    labeled, unlabeled = generate_gaussian_dataset(n=1, N=2, random_seed=0)
    assert isinstance(labeled, Dataset)
    assert isinstance(unlabeled, Dataset)
    assert len(labeled) == 1
    assert len(unlabeled) == 2
    for record in labeled:
        assert "y_proxy" in record
        assert "y_true" in record
        assert isinstance(record["y_proxy"], float)
        assert isinstance(record["y_true"], float)
    for record in unlabeled:
        assert "y_proxy" in record
        assert "y_true" not in record


def test_generate_gaussian_dataset_invalid_correlation_raises():
    with pytest.raises(ValueError, match="Correlation should be strictly between -1 and 1"):
        generate_gaussian_dataset(n=1, N=1, correlation=1.5)

    with pytest.raises(ValueError):
        generate_gaussian_dataset(n=1, N=1, correlation=-1.5)


def test_generate_gaussian_dataset_reproducibility():
    labeled1, unlabeled1 = generate_gaussian_dataset(n=1, N=2, random_seed=7)
    labeled2, unlabeled2 = generate_gaussian_dataset(n=1, N=2, random_seed=7)
    assert labeled1 == labeled2
    assert unlabeled1 == unlabeled2
