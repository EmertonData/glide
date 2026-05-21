import numpy as np
import pytest

from glide.simulators import generate_stratified_binary_dataset


def test_generate_stratified_binary_dataset_empirical_means_and_correlation_per_stratum():
    n_total = [250, 250]
    true_mean = [0.9, 0.8]
    proxy_mean = [0.8, 0.7]
    correlation = [0.5, 0.75]

    y_true, y_proxy, groups = generate_stratified_binary_dataset(
        n_total=n_total,
        true_mean=true_mean,
        proxy_mean=proxy_mean,
        correlation=correlation,
        random_seed=0,
    )

    for stratum_id in range(len(n_total)):
        stratum_mask = groups == stratum_id
        y_true_stratum = y_true[stratum_mask]
        y_proxy_stratum = y_proxy[stratum_mask]

        assert np.mean(y_true_stratum) == pytest.approx(true_mean[stratum_id], abs=1e-9)
        assert np.mean(y_proxy_stratum) == pytest.approx(proxy_mean[stratum_id], abs=1e-9)

        empirical_corr = np.corrcoef(y_true_stratum, y_proxy_stratum)[0, 1]
        assert empirical_corr == pytest.approx(correlation[stratum_id], abs=1e-9)
