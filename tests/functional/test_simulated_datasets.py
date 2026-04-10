import numpy as np
import pytest

from glide.core.simulated_datasets import (
    generate_binary_dataset,
    generate_binary_dataset_with_oracle_sampling,
    generate_gaussian_dataset,
    generate_stratified_binary_dataset,
)


def test_generate_binary_dataset_empirical_means_and_correlation():
    y_true, y_proxy = generate_binary_dataset(
        n=500, N=4500, true_mean=0.7, proxy_mean=0.6, correlation=0.8, random_seed=42
    )
    # Extract labeled and unlabeled
    labeled_mask = ~np.isnan(y_true)
    y_true_labeled = y_true[labeled_mask]
    y_proxy_labeled = y_proxy[labeled_mask]

    true_mean = np.mean(y_true_labeled)
    proxy_mean = np.mean(y_proxy)
    assert true_mean == pytest.approx(0.7, abs=0.03)
    assert proxy_mean == pytest.approx(0.6, abs=0.03)

    empirical_corr = np.corrcoef(y_true_labeled, y_proxy_labeled)[0, 1]
    assert empirical_corr == pytest.approx(0.8, abs=0.05)


def test_generate_binary_dataset_with_oracle_sampling_empirical_means_and_correlation():
    data = generate_binary_dataset_with_oracle_sampling(
        N=5000, true_mean=0.7, proxy_mean=0.6, correlation=0.5, random_seed=42
    )
    y_true = data["y_true"]
    y_proxy = data["y_proxy"]
    assert np.mean(y_true) == pytest.approx(0.7, abs=0.03)
    assert np.mean(y_proxy) == pytest.approx(0.6, abs=0.03)
    empirical_corr = np.corrcoef(y_true, y_proxy)[0, 1]
    assert empirical_corr == pytest.approx(0.5, abs=0.05)
    uncertainty_array = data["uncertainty"]
    assert np.std(uncertainty_array) == pytest.approx(0.07, abs=0.01)


def test_generate_binary_dataset_with_oracle_rms_error_non_uniform():
    # With lower correlation, uncertainty variation is more visible
    data = generate_binary_dataset_with_oracle_sampling(
        N=1000, true_mean=0.5, proxy_mean=0.5, correlation=0.3, random_seed=42
    )
    uncertainty_values = data["uncertainty"]
    assert np.std(uncertainty_values) > 1e-2


def test_generate_gaussian_dataset_empirical_means_and_correlation():
    y_true, y_proxy = generate_gaussian_dataset(
        n=500, N=500, true_mean=0.7, true_std=0.2, proxy_mean=0.6, proxy_std=0.3, correlation=0.8, random_seed=42
    )

    # Extract labeled and unlabeled
    labeled_mask = ~np.isnan(y_true)
    y_true_labeled = y_true[labeled_mask]
    y_proxy_labeled = y_proxy[labeled_mask]

    eps = 0.03
    assert np.mean(y_true_labeled) == pytest.approx(0.7, abs=eps)
    assert np.mean(y_proxy) == pytest.approx(0.6, abs=eps)

    empirical_corr = np.corrcoef(y_true_labeled, y_proxy_labeled)[0, 1]
    assert empirical_corr == pytest.approx(0.8, abs=eps)

    assert np.std(y_true_labeled) == pytest.approx(0.2, abs=eps)
    assert np.std(y_proxy_labeled) == pytest.approx(0.3, abs=eps)


def test_generate_stratified_binary_dataset_empirical_means_and_correlation_per_stratum():
    n = [250, 250]
    N = [2250, 2250]
    true_mean = [0.9, 0.8]
    proxy_mean = [0.8, 0.7]
    correlation = [0.5, 0.75]

    y_true, y_proxy, groups = generate_stratified_binary_dataset(
        n=n,
        N=N,
        true_mean=true_mean,
        proxy_mean=proxy_mean,
        correlation=correlation,
        random_seed=0,
    )

    # Filter by stratum using the returned groups array
    for stratum_id in range(len(n)):
        mask = groups == stratum_id

        y_true_stratum = y_true[mask]
        y_proxy_stratum = y_proxy[mask]

        # Extract labeled subset
        labeled_mask = ~np.isnan(y_true_stratum)
        y_true_labeled = y_true_stratum[labeled_mask]
        y_proxy_labeled = y_proxy_stratum[labeled_mask]

        # Expected values per stratum
        expected_true_mean = true_mean[stratum_id]
        expected_proxy_mean = proxy_mean[stratum_id]
        expected_corr = correlation[stratum_id]

        # Check means (with tolerance for randomness)
        empirical_true_mean = np.mean(y_true_labeled)
        empirical_proxy_mean = np.mean(y_proxy_stratum)
        assert empirical_true_mean == pytest.approx(expected_true_mean, abs=0.03)
        assert empirical_proxy_mean == pytest.approx(expected_proxy_mean, abs=0.03)

        # Check correlation
        empirical_corr = np.corrcoef(y_true_labeled, y_proxy_labeled)[0, 1]
        assert empirical_corr == pytest.approx(expected_corr, abs=0.05)
