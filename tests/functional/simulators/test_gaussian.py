import numpy as np
import pytest

from glide.simulators import generate_gaussian_dataset


def test_generate_gaussian_dataset_empirical_means_and_correlation():
    y_true, y_proxy = generate_gaussian_dataset(
        n=500, N=500, true_mean=0.7, true_std=0.2, proxy_mean=0.6, proxy_std=0.3, correlation=0.8, random_seed=42
    )

    # Extract labeled and unlabeled
    labeled_mask = ~np.isnan(y_true)
    y_true_labeled = y_true[labeled_mask]
    y_proxy_labeled = y_proxy[labeled_mask]

    eps = 0.03
    assert np.nanmean(y_true) == pytest.approx(0.7, abs=eps)
    assert np.mean(y_proxy_labeled) == pytest.approx(0.6, abs=eps)

    empirical_corr = np.corrcoef(y_true_labeled, y_proxy_labeled)[0, 1]
    assert empirical_corr == pytest.approx(0.8, abs=eps)

    assert np.nanstd(y_true_labeled) == pytest.approx(0.2, abs=eps)
    assert np.std(y_proxy_labeled) == pytest.approx(0.3, abs=eps)
