import numpy as np
import pytest

from glide.simulators import generate_binary_dataset_with_oracle_sampling


def test_generate_binary_dataset_with_oracle_sampling_empirical_means_and_correlation():
    y_true, y_proxy, uncertainty = generate_binary_dataset_with_oracle_sampling(
        n_total=5000, true_mean=0.7, proxy_mean=0.6, correlation=0.5, random_seed=42
    )
    assert np.mean(y_true) == pytest.approx(0.7, abs=0.03)
    assert np.mean(y_proxy) == pytest.approx(0.6, abs=0.03)
    empirical_corr = np.corrcoef(y_true, y_proxy)[0, 1]
    assert empirical_corr == pytest.approx(0.5, abs=0.05)
    assert np.std(uncertainty) == pytest.approx(0.07, abs=0.01)


def test_generate_binary_dataset_with_oracle_rms_error_non_uniform():
    # With lower correlation, uncertainty variation is more visible
    _, _, uncertainty = generate_binary_dataset_with_oracle_sampling(
        n_total=1000, true_mean=0.5, proxy_mean=0.5, correlation=0.3, random_seed=42
    )
    assert np.std(uncertainty) > 1e-2
