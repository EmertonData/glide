import numpy as np
import pytest

from glide.simulators import generate_stratified_multi_binary_dataset


def test_generate_stratified_multi_binary_dataset_empirical_means_and_correlation_per_stratum():
    n_samples = [250, 250]
    true_mean = [0.9, 0.8]
    proxy_means = [[0.8, 0.85], [0.7, 0.75]]
    correlations = [[0.5, 0.6], [0.75, 0.7]]

    y_true, y_proxies, groups = generate_stratified_multi_binary_dataset(
        n_samples=n_samples,
        true_mean=true_mean,
        proxy_means=proxy_means,
        correlations=correlations,
        random_seed=9,
    )

    for stratum_id in range(len(n_samples)):
        stratum_mask = groups == stratum_id
        y_true_stratum = y_true[stratum_mask]
        y_proxies_stratum = y_proxies[stratum_mask]

        assert np.mean(y_true_stratum) == pytest.approx(true_mean[stratum_id], abs=0.05)

        for proxy_id in range(len(proxy_means[stratum_id])):
            y_proxy_stratum = y_proxies_stratum[:, proxy_id]
            assert np.mean(y_proxy_stratum) == pytest.approx(proxy_means[stratum_id][proxy_id], abs=0.05)

            empirical_corr = np.corrcoef(y_true_stratum, y_proxy_stratum)[0, 1]
            assert empirical_corr == pytest.approx(correlations[stratum_id][proxy_id], abs=0.05)
