import numpy as np
import pytest

from glide.simulators import generate_binary_dataset


def test_generate_binary_dataset_empirical_means_and_correlation():
    y_true, y_proxy = generate_binary_dataset(
        n_total=500, true_mean=0.7, proxy_mean=0.6, correlation=0.8, random_seed=2
    )
    assert np.mean(y_true) == pytest.approx(0.7, abs=0.03)
    assert np.mean(y_proxy) == pytest.approx(0.6, abs=0.03)

    empirical_corr = np.corrcoef(y_true, y_proxy)[0, 1]
    assert empirical_corr == pytest.approx(0.8, abs=0.03)
