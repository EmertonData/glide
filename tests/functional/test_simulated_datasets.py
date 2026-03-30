import numpy as np
import pytest

from glide.core.simulated_datasets import (
    generate_binary_dataset,
    generate_binary_dataset_with_oracle_sampling,
    generate_gaussian_dataset,
)


def test_generate_binary_dataset_empirical_means_and_correlation():
    labeled, unlabeled = generate_binary_dataset(
        n=500, N=4500, true_mean=0.7, proxy_mean=0.6, correlation=0.8, random_seed=42
    )
    dataset = labeled + unlabeled
    y_true = labeled.to_numpy(fields=["y_true"]).flatten()
    y_proxy_all = dataset.to_numpy(fields=["y_proxy"]).flatten()
    true_mean = np.mean(y_true)
    proxy_mean = np.mean(y_proxy_all)
    assert abs(true_mean - 0.7) < 0.03
    assert abs(proxy_mean - 0.6) < 0.03
    y_proxy_labeled = labeled.to_numpy(fields=["y_proxy"]).flatten()
    empirical_corr = np.corrcoef(y_true, y_proxy_labeled)[0, 1]
    assert abs(empirical_corr - 0.8) < 0.05


def test_generate_binary_dataset_with_oracle_sampling_empirical_means_and_correlation():
    dataset = generate_binary_dataset_with_oracle_sampling(
        N=5000, true_mean=0.7, proxy_mean=0.6, correlation=0.8, random_seed=42
    )
    y_true = dataset.to_numpy(fields=["y_true"]).flatten()
    y_proxy = dataset.to_numpy(fields=["y_proxy"]).flatten()
    assert abs(np.mean(y_true) - 0.7) < 0.03
    assert abs(np.mean(y_proxy) - 0.6) < 0.03
    empirical_corr = np.corrcoef(y_true, y_proxy)[0, 1]
    assert abs(empirical_corr - 0.8) < 0.05


def test_generate_binary_dataset_with_oracle_rms_error_non_uniform():
    # With lower correlation, rms_error variation is more visible
    dataset = generate_binary_dataset_with_oracle_sampling(
        N=1000, true_mean=0.5, proxy_mean=0.5, correlation=0.3, random_seed=42
    )
    rms_error_values = np.array([record["rms_error"] for record in dataset])
    assert np.std(rms_error_values) > 1e-2


def test_generate_gaussian_dataset_empirical_means_and_correlation():
    labeled, unlabeled = generate_gaussian_dataset(
        n=500, N=500, true_mean=0.7, true_std=0.2, proxy_mean=0.6, proxy_std=0.3, correlation=0.8, random_seed=42
    )

    y_proxy_unlabeled = unlabeled.to_numpy(fields=["y_proxy"])[:, 0]

    labeled_data_array = labeled.to_numpy(fields=["y_true", "y_proxy"])
    y_true, y_proxy_labeled = labeled_data_array[:, 0], labeled_data_array[:, 1]

    y_proxy_all = np.hstack((y_proxy_labeled, y_proxy_unlabeled))

    eps = 0.03
    assert np.mean(y_true) == pytest.approx(0.7, abs=eps)
    assert np.mean(y_proxy_all) == pytest.approx(0.6, abs=eps)

    empirical_corr = np.corrcoef(y_true, y_proxy_labeled)[0, 1]
    assert empirical_corr == pytest.approx(0.8, abs=eps)

    assert np.std(y_true) == pytest.approx(0.2, abs=eps)
    assert np.std(y_proxy_labeled) == pytest.approx(0.3, abs=eps)
