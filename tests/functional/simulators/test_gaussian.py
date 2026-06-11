import numpy as np
import pytest

from glide.simulators import generate_gaussian_dataset


def test_generate_gaussian_dataset_empirical_means_and_correlation():
    true_mean = 0.7
    true_std = 0.2
    proxy_mean = 0.6
    proxy_std = 0.3
    correlation = 0.8
    eps = 0.03

    y_true, y_proxy = generate_gaussian_dataset(
        n_total=1000,
        true_mean=true_mean,
        true_std=true_std,
        proxy_mean=proxy_mean,
        proxy_std=proxy_std,
        correlation=correlation,
        random_seed=42,
    )

    assert np.mean(y_true) == pytest.approx(true_mean, abs=eps)
    assert np.mean(y_proxy) == pytest.approx(proxy_mean, abs=eps)

    empirical_corr = np.corrcoef(y_true, y_proxy)[0, 1]
    assert empirical_corr == pytest.approx(correlation, abs=eps)

    assert np.std(y_true) == pytest.approx(true_std, abs=eps)
    assert np.std(y_proxy) == pytest.approx(proxy_std, abs=eps)
