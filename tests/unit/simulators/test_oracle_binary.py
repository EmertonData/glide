import numpy as np
import pytest

from glide.simulators.oracle_binary import generate_binary_dataset_with_oracle_sampling


def test_generate_binary_dataset_with_oracle_sampling_structure_and_counts():
    y_true, y_proxy, uncertainty = generate_binary_dataset_with_oracle_sampling(n_total=10, random_seed=0)
    assert len(y_true) == 10
    assert len(y_proxy) == 10
    assert len(uncertainty) == 10
    assert np.isin(y_true, [0.0, 1.0]).all()
    assert np.isin(y_proxy, [0.0, 1.0]).all()
    assert np.all(uncertainty > 0)


def test_generate_binary_dataset_with_oracle_sampling_invalid_true_mean_raises():
    with pytest.raises(ValueError, match=r"true_mean must be in \(0, 1\), got 1\.5"):
        generate_binary_dataset_with_oracle_sampling(n_total=10, true_mean=1.5)


def test_generate_binary_dataset_with_oracle_sampling_invalid_proxy_mean_raises():
    with pytest.raises(ValueError, match=r"proxy_mean must be in \(0, 1\), got 0"):
        generate_binary_dataset_with_oracle_sampling(n_total=10, proxy_mean=0.0)


def test_generate_binary_dataset_with_oracle_sampling_impossible_correlation_raises():
    with pytest.raises(
        ValueError,
        match=r"Impossible combination of true_mean=0\.7, proxy_mean=0\.6, and correlation=0\.95",
    ):
        generate_binary_dataset_with_oracle_sampling(n_total=10, true_mean=0.7, proxy_mean=0.6, correlation=0.95)


def test_generate_binary_dataset_with_oracle_sampling_reproducibility():
    y_true1, y_proxy1, uncertainty1 = generate_binary_dataset_with_oracle_sampling(n_total=10, random_seed=7)
    y_true2, y_proxy2, uncertainty2 = generate_binary_dataset_with_oracle_sampling(n_total=10, random_seed=7)
    np.testing.assert_array_equal(y_true1, y_true2)
    np.testing.assert_array_equal(y_proxy1, y_proxy2)
    np.testing.assert_array_equal(uncertainty1, uncertainty2)
