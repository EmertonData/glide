import numpy as np
import pytest

from glide.simulators import generate_clustered_binary_dataset


def test_generate_clustered_binary_dataset_empirical_means_and_correlation():
    n_total = 5000
    n_clusters = 50
    true_mean = 0.7
    proxy_mean = 0.6
    correlation = 0.8

    y_true, y_proxy, clusters = generate_clustered_binary_dataset(
        n_total=n_total,
        n_clusters=n_clusters,
        true_mean=true_mean,
        proxy_mean=proxy_mean,
        correlation=correlation,
        random_seed=42,
    )
    assert np.mean(y_true) == pytest.approx(true_mean, abs=0.03)
    assert np.mean(y_proxy) == pytest.approx(proxy_mean, abs=0.03)
    assert np.corrcoef(y_true, y_proxy)[0, 1] == pytest.approx(correlation, abs=0.05)
    np.testing.assert_array_equal(np.unique(clusters), np.arange(n_clusters))
