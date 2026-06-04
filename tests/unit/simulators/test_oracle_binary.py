from unittest.mock import call, patch

import numpy as np
import pytest

from glide.simulators import generate_binary_dataset_with_oracle_sampling


def test_generate_binary_dataset_with_oracle_sampling_structure_and_counts():
    y_true_oracle, y_proxy, uncertainty = generate_binary_dataset_with_oracle_sampling(n_total=10, random_seed=0)
    assert len(y_true_oracle) == 10
    assert len(y_proxy) == 10
    assert len(uncertainty) == 10
    assert np.isin(y_true_oracle, [0.0, 1.0]).all()
    assert np.isin(y_proxy, [0.0, 1.0]).all()
    assert np.all(uncertainty > 0)


def test_generate_binary_dataset_with_oracle_sampling_delegates_validation():
    with patch("glide.simulators.oracle_binary._validate_bounds") as mock_validate_bounds:
        generate_binary_dataset_with_oracle_sampling(
            n_total=2, true_mean=0.7, proxy_mean=0.6, correlation=0.8, random_seed=0
        )
    mock_validate_bounds.assert_has_calls(
        [
            call(0.7, "true_mean", lower=0, upper=1, left_inclusive=False, right_inclusive=False),
            call(0.6, "proxy_mean", lower=0, upper=1, left_inclusive=False, right_inclusive=False),
        ]
    )


def test_generate_binary_dataset_with_oracle_sampling_impossible_correlation_raises():
    with pytest.raises(
        ValueError,
        match=r"Impossible combination of 'true_mean'=0\.7, 'proxy_mean'=0\.6, and 'correlation'=0\.95",
    ):
        generate_binary_dataset_with_oracle_sampling(n_total=10, true_mean=0.7, proxy_mean=0.6, correlation=0.95)


def test_generate_binary_dataset_with_oracle_sampling_reproducibility():
    y_true_oracle1, y_proxy1, uncertainty1 = generate_binary_dataset_with_oracle_sampling(n_total=10, random_seed=7)
    y_true_oracle2, y_proxy2, uncertainty2 = generate_binary_dataset_with_oracle_sampling(n_total=10, random_seed=7)
    np.testing.assert_array_equal(y_true_oracle1, y_true_oracle2)
    np.testing.assert_array_equal(y_proxy1, y_proxy2)
    np.testing.assert_array_equal(uncertainty1, uncertainty2)


def test_generate_binary_dataset_with_oracle_sampling_different_seed_results_differ():
    y_true1, y_proxy1, _ = generate_binary_dataset_with_oracle_sampling(n_total=10, random_seed=0)
    y_true2, y_proxy2, _ = generate_binary_dataset_with_oracle_sampling(n_total=10, random_seed=1)
    assert not np.array_equal(y_true1, y_true2) or not np.array_equal(y_proxy1, y_proxy2)
