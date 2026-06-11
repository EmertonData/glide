import numpy as np
import pytest

from glide.simulators import generate_gaussian_dataset


def test_generate_gaussian_dataset_empirical_means_and_correlation():
    y_true, y_proxy = generate_gaussian_dataset(
        n_total=1000,
        true_mean=0.7,
        true_std=0.2,
        proxy_mean=0.6,
        proxy_std=0.3,
        correlation=0.8,
        random_seed=42,
    )

    eps = 0.03
    assert np.mean(y_true) == pytest.approx(0.7, abs=eps)
    assert np.mean(y_proxy) == pytest.approx(0.6, abs=eps)

    empirical_corr = np.corrcoef(y_true, y_proxy)[0, 1]
    assert empirical_corr == pytest.approx(0.8, abs=eps)

    assert np.std(y_true) == pytest.approx(0.2, abs=eps)
    assert np.std(y_proxy) == pytest.approx(0.3, abs=eps)
