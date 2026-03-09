import numpy as np

from glide.core.simulated import generate_dataset_binary


def test_generate_dataset_binary_empirical_means():
    ds = generate_dataset_binary(n=500, N=4500, true_mean=0.7, proxy_mean=0.6, random_seed=42)
    labeled = [r for r in ds if "true" in r]
    true_mean = np.mean([r["true"] for r in labeled])
    proxy_mean = np.mean([r["proxy"] for r in ds])
    assert abs(true_mean - 0.7) < 0.03
    assert abs(proxy_mean - 0.6) < 0.03
