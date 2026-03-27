import numpy as np

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
