import numpy as np
import pytest

from glide.core.simulated_datasets import generate_binary_dataset, generate_binary_dataset_with_oracle_sampling


def test_generate_binary_dataset_empirical_means_and_correlation():
    labeled, unlabeled = generate_binary_dataset(
        n=500, N=4500, true_mean=0.7, proxy_mean=0.6, correlation=0.8, random_seed=42
    )
    dataset = labeled + unlabeled
    y_true = labeled.to_numpy(fields=["y_true"]).flatten()
    y_proxy_all = dataset.to_numpy(fields=["y_proxy"]).flatten()
    true_mean = np.mean(y_true)
    proxy_mean = np.mean(y_proxy_all)
    assert true_mean == pytest.approx(0.7, abs=0.03)
    assert proxy_mean == pytest.approx(0.6, abs=0.03)
    y_proxy_labeled = labeled.to_numpy(fields=["y_proxy"]).flatten()
    empirical_corr = np.corrcoef(y_true, y_proxy_labeled)[0, 1]
    assert empirical_corr == pytest.approx(0.8, abs=0.05)


def test_generate_binary_dataset_with_oracle_sampling_empirical_means_and_correlation():
    dataset = generate_binary_dataset_with_oracle_sampling(
        N=5000, true_mean=0.7, proxy_mean=0.6, correlation=0.5, random_seed=42
    )
    y_true = dataset.to_numpy(fields=["y_true"]).flatten()
    y_proxy = dataset.to_numpy(fields=["y_proxy"]).flatten()
    assert np.mean(y_true) == pytest.approx(0.7, abs=0.03)
    assert np.mean(y_proxy) == pytest.approx(0.6, abs=0.03)
    empirical_corr = np.corrcoef(y_true, y_proxy)[0, 1]
    assert empirical_corr == pytest.approx(0.5, abs=0.05)
    RMSE_array = dataset.to_numpy(fields=["RMSE"])[:, 0]
    assert np.std(RMSE_array) == pytest.approx(0.07, abs=0.01)
