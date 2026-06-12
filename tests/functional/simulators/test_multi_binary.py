import numpy as np
import pytest

from glide.simulators import generate_multi_binary_dataset


def test_generate_multi_binary_dataset_empirical_statistics():
    y_true_mean = 0.6
    y_proxy_means = [0.5, 0.7]
    correlations = [0.5, 0.6]
    y_true, y_proxies = generate_multi_binary_dataset(1000, y_true_mean, y_proxy_means, correlations, random_seed=0)

    assert np.mean(y_true) == pytest.approx(y_true_mean, abs=0.02)
    assert np.mean(y_proxies[:, 0]) == pytest.approx(y_proxy_means[0], abs=0.02)
    assert np.mean(y_proxies[:, 1]) == pytest.approx(y_proxy_means[1], abs=0.02)
    assert np.corrcoef(y_true, y_proxies[:, 0])[0, 1] == pytest.approx(correlations[0], abs=0.05)
    assert np.corrcoef(y_true, y_proxies[:, 1])[0, 1] == pytest.approx(correlations[1], abs=0.05)
