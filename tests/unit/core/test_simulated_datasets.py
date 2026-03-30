import pytest

from glide.core.dataset import Dataset
from glide.core.simulated_datasets import (
    generate_binary_dataset,
    generate_binary_dataset_with_oracle_sampling,
    generate_stratified_binary_dataset,
    generate_gaussian_dataset,
)


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
        assert "rms_error" in record
        assert record["y_true"] in (0, 1)
        assert record["y_proxy"] in (0, 1)
        assert record["rms_error"] > 0


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


def test_generate_stratified_binary_dataset_structure():
    labeled, unlabeled = generate_stratified_binary_dataset(n=[1, 2], N=[2, 1], random_seed=0)
    assert isinstance(labeled, Dataset)
    assert isinstance(unlabeled, Dataset)
    assert len(labeled) == 3
    assert len(unlabeled) == 3
    for record in labeled:
        assert "y_true" in record
        assert "y_proxy" in record
        assert "stratum_id" in record


def test_generate_stratified_binary_dataset_stratum_id():
    labeled, unlabeled = generate_stratified_binary_dataset(n=[1, 2], N=[2, 1], random_seed=0)
    stratum_ids_labeled = [record["stratum_id"] for record in labeled.records]
    stratum_ids_unlabeled = [record["stratum_id"] for record in unlabeled.records]
    assert stratum_ids_labeled == [0, 1, 1]
    assert stratum_ids_unlabeled == [0, 0, 1]


def test_generate_stratified_binary_dataset_mismatched_lists_raises():
    with pytest.raises(ValueError, match=r"All input lists must have the same length"):
        generate_stratified_binary_dataset(n=[1, 2], N=[2, 1, 3], true_mean=[0.5, 0.6])


def test_generate_stratified_binary_dataset_reproducibility():
    labeled1, unlabeled1 = generate_stratified_binary_dataset(n=[1, 2], N=[2, 1], random_seed=42)
    labeled2, unlabeled2 = generate_stratified_binary_dataset(n=[1, 2], N=[2, 1], random_seed=42)
    assert labeled1 == labeled2
    assert unlabeled1 == unlabeled2


def test_generate_stratified_binary_dataset_empty_strata_raises():
    with pytest.raises(ValueError, match=r"Number of strata must be at least 1, got 0"):
        generate_stratified_binary_dataset(n=[], N=[])


def test_generate_stratified_binary_dataset_no_random_seed():
    labeled, unlabeled = generate_stratified_binary_dataset(n=[1, 2], N=[2, 1], random_seed=None)
    assert isinstance(labeled, Dataset)
    assert isinstance(unlabeled, Dataset)
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


def test_generate_gaussian_dataset_invalid_positive_correlation_raises():
    with pytest.raises(ValueError, match="Correlation should be between -1 and 1"):
        generate_gaussian_dataset(n=1, N=1, correlation=1.5)


def test_generate_gaussian_dataset_invalid_negative_correlation_raises():
    with pytest.raises(ValueError, match="Correlation should be between -1 and 1"):
        generate_gaussian_dataset(n=1, N=1, correlation=-1.5)


def test_generate_gaussian_dataset_reproducibility():
    labeled1, unlabeled1 = generate_gaussian_dataset(n=1, N=2, random_seed=7)
    labeled2, unlabeled2 = generate_gaussian_dataset(n=1, N=2, random_seed=7)
    assert labeled1 == labeled2
    assert unlabeled1 == unlabeled2
