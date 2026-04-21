import numpy as np
import pytest

from glide.simulators import generate_stratified_binary_dataset


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
        stratum_mask = groups == stratum_id

        y_true_stratum = y_true[stratum_mask]
        y_proxy_stratum = y_proxy[stratum_mask]

        # Extract labeled subset
        labeled_mask = ~np.isnan(y_true_stratum)
        y_true_labeled = y_true_stratum[labeled_mask]
        y_proxy_labeled = y_proxy_stratum[labeled_mask]

        # Expected values per stratum
        expected_true_mean = true_mean[stratum_id]
        expected_proxy_mean = proxy_mean[stratum_id]
        expected_corr = correlation[stratum_id]

        # Check means (with tolerance for randomness)
        empirical_true_mean = np.nanmean(y_true_stratum)
        empirical_proxy_mean = np.mean(y_proxy_stratum)
        assert empirical_true_mean == pytest.approx(expected_true_mean, abs=0.03)
        assert empirical_proxy_mean == pytest.approx(expected_proxy_mean, abs=0.03)

        # Check correlation
        empirical_corr = np.corrcoef(y_true_labeled, y_proxy_labeled)[0, 1]
        assert empirical_corr == pytest.approx(expected_corr, abs=0.05)
