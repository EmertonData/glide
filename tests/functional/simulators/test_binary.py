import numpy as np
import pytest

from glide.simulators import generate_binary_dataset


def test_generate_binary_dataset_empirical_means_and_correlation():
    y_true, y_proxy = generate_binary_dataset(
        n_labeled=500, n_unlabeled=4500, true_mean=0.7, proxy_mean=0.6, correlation=0.8, random_seed=42
    )
    # Extract labeled and unlabeled
    labeled_mask = ~np.isnan(y_true)
    y_true_labeled = y_true[labeled_mask]
    y_proxy_labeled = y_proxy[labeled_mask]

    true_mean = np.nanmean(y_true)
    proxy_mean = np.mean(y_proxy)
    assert true_mean == pytest.approx(0.7, abs=0.03)
    assert proxy_mean == pytest.approx(0.6, abs=0.03)

    empirical_corr = np.corrcoef(y_true_labeled, y_proxy_labeled)[0, 1]
    assert empirical_corr == pytest.approx(0.8, abs=0.05)
