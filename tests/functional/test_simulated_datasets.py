import numpy as np

from glide.core.simulated_datasets import generate_binary_dataset


def test_generate_binary_dataset_empirical_means():
    labeled, unlabeled = generate_binary_dataset(n=500, N=4500, true_mean=0.7, proxy_mean=0.6, random_seed=42)
    dataset = labeled + unlabeled
    y_true = labeled.to_numpy(fields=["y_true"])
    y_proxy = dataset.to_numpy(fields=["y_proxy"])
    true_mean = np.mean(y_true)
    proxy_mean = np.mean(y_proxy)
    assert abs(true_mean - 0.7) < 0.03
    assert abs(proxy_mean - 0.6) < 0.03
