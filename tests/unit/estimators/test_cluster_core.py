from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

import glide.estimators.cluster_core as cluster_core_module
from glide.estimators.cluster_core import (
    _compute_cluster_mean_estimate,
    _compute_cluster_std_estimate,
    _compute_cluster_tuning_parameter,
)


@pytest.fixture
def labeled_true_sums() -> NDArray:
    return np.array([2.0, 4.0])


@pytest.fixture
def labeled_proxy_sums() -> NDArray:
    return np.array([1.0, 3.0])


@pytest.fixture
def unlabeled_proxy_sums() -> NDArray:
    return np.array([3.0, 5.0])


# --- _compute_cluster_tuning_parameter ---


def test_compute_cluster_tuning_parameter_returns_one_when_power_tuning_false(
    labeled_true_sums, labeled_proxy_sums, unlabeled_proxy_sums
):
    result = _compute_cluster_tuning_parameter(
        labeled_true_sums, labeled_proxy_sums, unlabeled_proxy_sums, 2, 2, power_tuning=False
    )
    assert result == 1.0


def test_compute_cluster_tuning_parameter_delegates_to_validation(
    labeled_true_sums, labeled_proxy_sums, unlabeled_proxy_sums
):
    with patch.object(cluster_core_module, "_validate_non_constant") as mock_validate_non_constant:
        _compute_cluster_tuning_parameter(
            labeled_true_sums, labeled_proxy_sums, unlabeled_proxy_sums, 2, 2, power_tuning=True
        )

        mock_validate_non_constant.assert_called_once()
        expected_all_proxy_sums = np.hstack([labeled_proxy_sums, unlabeled_proxy_sums])
        np.testing.assert_array_equal(mock_validate_non_constant.call_args[0][0], expected_all_proxy_sums)
        assert mock_validate_non_constant.call_args[0][1] == (
            "Proxy cluster sums have zero variance across both labeled and unlabeled clusters; "
            "cannot estimate the tuning parameter."
        )


def test_compute_cluster_tuning_parameter_known_value(labeled_true_sums, labeled_proxy_sums, unlabeled_proxy_sums):
    expected = 27 / 52
    result = _compute_cluster_tuning_parameter(
        labeled_true_sums, labeled_proxy_sums, unlabeled_proxy_sums, 4, 6, power_tuning=True
    )
    assert result == pytest.approx(expected, abs=1e-10)


# --- _compute_cluster_mean_estimate ---


def test_compute_cluster_mean_estimate_known_value(labeled_true_sums, labeled_proxy_sums, unlabeled_proxy_sums):
    expected = 87 / 52
    result = _compute_cluster_mean_estimate(
        labeled_true_sums, labeled_proxy_sums, unlabeled_proxy_sums, 4, 6, lambda_=27 / 52
    )
    assert result == pytest.approx(expected, abs=1e-10)


# --- _compute_cluster_std_estimate ---


def test_compute_cluster_std_estimate_known_value(labeled_true_sums, labeled_proxy_sums, unlabeled_proxy_sums):
    expected = np.sqrt(73 / 832)
    result = _compute_cluster_std_estimate(
        labeled_true_sums, labeled_proxy_sums, unlabeled_proxy_sums, 4, 6, lambda_=27 / 52
    )
    assert result == pytest.approx(expected, abs=1e-10)
