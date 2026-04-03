import numpy as np
import pytest

from glide.core.simulated_datasets import (
    generate_binary_dataset,
    generate_binary_dataset_with_oracle_sampling,
    generate_gaussian_dataset,
    generate_stratified_binary_dataset,
)


def test_generate_binary_dataset_empirical_means_and_correlation():
    labeled, unlabeled = generate_binary_dataset(
        n=500, N=4500, true_mean=0.7, proxy_mean=0.6, correlation=0.8, random_seed=42
    )
    dataset = labeled + unlabeled
    y_true = labeled.to_numpy(fields=["y_true"]).flatten()
    y_proxy_all = dataset.to_numpy(fields=["y_proxy"]).flatten()
    true_mean = np.mean(y_true)
    proxy_mean = np.mean(y_proxy_all)
    assert true_mean == pytest.approx(0.7, abs=0.03)
    assert proxy_mean == pytest.approx(0.6, abs=0.03)
    y_proxy_labeled = labeled.to_numpy(fields=["y_proxy"]).flatten()
    empirical_corr = np.corrcoef(y_true, y_proxy_labeled)[0, 1]
    assert empirical_corr == pytest.approx(0.8, abs=0.05)


def test_generate_binary_dataset_with_oracle_sampling_empirical_means_and_correlation():
    dataset = generate_binary_dataset_with_oracle_sampling(
        N=5000, true_mean=0.7, proxy_mean=0.6, correlation=0.5, random_seed=42
    )
    y_true = dataset.to_numpy(fields=["y_true"]).flatten()
    y_proxy = dataset.to_numpy(fields=["y_proxy"]).flatten()
    assert np.mean(y_true) == pytest.approx(0.7, abs=0.03)
    assert np.mean(y_proxy) == pytest.approx(0.6, abs=0.03)
    empirical_corr = np.corrcoef(y_true, y_proxy)[0, 1]
    assert empirical_corr == pytest.approx(0.5, abs=0.05)
    RMSE_array = dataset.to_numpy(fields=["RMSE"])[:, 0]
    assert np.std(RMSE_array) == pytest.approx(0.07, abs=0.01)


def test_generate_binary_dataset_with_oracle_rms_error_non_uniform():
    # With lower correlation, RMSE variation is more visible
    dataset = generate_binary_dataset_with_oracle_sampling(
        N=1000, true_mean=0.5, proxy_mean=0.5, correlation=0.3, random_seed=42
    )
    RMSE_values = np.array([record["RMSE"] for record in dataset])
    assert np.std(RMSE_values) > 1e-2


def test_generate_gaussian_dataset_empirical_means_and_correlation():
    labeled, unlabeled = generate_gaussian_dataset(
        n=500, N=500, true_mean=0.7, true_std=0.2, proxy_mean=0.6, proxy_std=0.3, correlation=0.8, random_seed=42
    )

    y_proxy_unlabeled = unlabeled.to_numpy(fields=["y_proxy"])[:, 0]

    labeled_data_array = labeled.to_numpy(fields=["y_true", "y_proxy"])
    y_true, y_proxy_labeled = labeled_data_array[:, 0], labeled_data_array[:, 1]

    y_proxy_all = np.hstack((y_proxy_labeled, y_proxy_unlabeled))

    eps = 0.03
    assert np.mean(y_true) == pytest.approx(0.7, abs=eps)
    assert np.mean(y_proxy_all) == pytest.approx(0.6, abs=eps)

    empirical_corr = np.corrcoef(y_true, y_proxy_labeled)[0, 1]
    assert empirical_corr == pytest.approx(0.8, abs=eps)

    assert np.std(y_true) == pytest.approx(0.2, abs=eps)
    assert np.std(y_proxy_labeled) == pytest.approx(0.3, abs=eps)


def test_generate_stratified_binary_dataset_empirical_means_and_correlation_per_stratum():
    true_mean = [0.7, 0.8]
    proxy_mean = [0.6, 0.7]
    correlation = [0.8, 0.75]

    labeled, unlabeled = generate_stratified_binary_dataset(
        n=[250, 250],
        N=[2250, 2250],
        true_mean=true_mean,
        proxy_mean=proxy_mean,
        correlation=correlation,
        random_seed=42,
    )

    # Test per-stratum means and correlations
    for stratum_id in [0, 1]:
        # Filter by stratum
        labeled_stratum = [r for r in labeled if r["stratum_id"] == stratum_id]
        unlabeled_stratum = [r for r in unlabeled if r["stratum_id"] == stratum_id]
        dataset_stratum = labeled_stratum + unlabeled_stratum

        # Extract values
        y_true_stratum = np.array([r["y_true"] for r in labeled_stratum])
        y_proxy_all_stratum = np.array([r["y_proxy"] for r in dataset_stratum])
        y_proxy_labeled_stratum = np.array([r["y_proxy"] for r in labeled_stratum])

        # Expected values per stratum
        expected_true_mean = true_mean[stratum_id]
        expected_proxy_mean = proxy_mean[stratum_id]
        expected_corr = correlation[stratum_id]

        # Check means (with tolerance for randomness)
        empirical_true_mean = np.mean(y_true_stratum)
        empirical_proxy_mean = np.mean(y_proxy_all_stratum)
        assert empirical_true_mean == pytest.approx(expected_true_mean, abs=0.05)
        assert empirical_proxy_mean == pytest.approx(expected_proxy_mean, abs=0.05)

        # Check correlation
        empirical_corr = np.corrcoef(y_true_stratum, y_proxy_labeled_stratum)[0, 1]
        assert empirical_corr == pytest.approx(expected_corr, abs=0.1)
