import numpy as np
import pytest

from glide.core.simulated_datasets import generate_binary_dataset, generate_gaussian_dataset


def test_generate_binary_dataset_empirical_means():
    ds = generate_binary_dataset(n=500, N=4500, true_mean=0.7, proxy_mean=0.6, random_seed=42)
    labeled = [r for r in ds if "y_true" in r]
    true_mean = np.mean([r["y_true"] for r in labeled])
    proxy_mean = np.mean([r["y_proxy"] for r in ds])
    assert true_mean == pytest.approx(0.7, abs=0.03)
    assert proxy_mean == pytest.approx(0.6, abs=0.03)


def test_generate_gaussian_dataset_empirical_means_and_correlation():
    labeled, unlabeled = generate_gaussian_dataset(
        n=500, N=500, true_mean=0.7, true_std=0.2, proxy_mean=0.6, proxy_std=0.3, correlation=0.8, random_seed=42
    )

    y_proxy_unlabeled = unlabeled.to_numpy(fields=["y_proxy"])[:, 0]

    labeled_data_array = labeled.to_numpy(fields=["y_true", "y_proxy"])
    y_true, y_proxy_labeled = labeled_data_array[:, 0], labeled_data_array[:, 1]

    y_proxy_all = np.concat((y_proxy_labeled, y_proxy_unlabeled))

    eps = 0.03
    assert np.mean(y_true) == pytest.approx(0.7, abs=eps)
    assert np.mean(y_proxy_all) == pytest.approx(0.6, abs=eps)

    empirical_corr = np.corrcoef(y_true, y_proxy_labeled)[0, 1]
    assert empirical_corr == pytest.approx(0.8, abs=eps)

    assert np.std(y_true) == pytest.approx(0.2, abs=eps)
    assert np.std(y_proxy_labeled) == pytest.approx(0.3, abs=eps)
