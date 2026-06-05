from unittest.mock import call, patch

import numpy as np
import pytest
from numpy.typing import NDArray

import glide.estimators.cluster_core as cluster_core_module
from glide.estimators.cluster_core import (
    _compute_cluster_mean_estimate,
    _compute_cluster_std_estimate,
    _compute_cluster_tuning_parameter,
    _preprocess,
)


@pytest.fixture
def y_true() -> NDArray:
    return np.array([1.0, 1.0, 2.0, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])


@pytest.fixture
def y_proxy() -> NDArray:
    return np.array([0.5, 0.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0])


@pytest.fixture
def clusters() -> NDArray:
    return np.array(["A", "A", "B", "B", "C", "C", "C", "D", "D", "D"])


@pytest.fixture
def labeled_true_sums() -> NDArray:
    return np.array([2.0, 4.0])


@pytest.fixture
def labeled_proxy_sums() -> NDArray:
    return np.array([1.0, 3.0])


@pytest.fixture
def unlabeled_proxy_sums() -> NDArray:
    return np.array([3.0, 5.0])


# --- _preprocess ---


def test_preprocess_delegates_to_validation(y_true, y_proxy, clusters):
    with (
        patch.object(cluster_core_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(cluster_core_module, "_validate_has_no_nan") as mock_validate_has_no_nan,
        patch.object(cluster_core_module, "_validate_bounds") as mock_validate_bounds,
    ):
        _preprocess(y_true, y_proxy, clusters)

        mock_validate_equal_lengths.assert_called_once()
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], y_true)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], y_proxy)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][2], clusters)
        assert mock_validate_equal_lengths.call_args[1] == {"names": ["y_true", "y_proxy", "clusters"]}

        assert len(mock_validate_has_no_nan.call_args_list) == 2
        np.testing.assert_array_equal(mock_validate_has_no_nan.call_args_list[0][0][0], y_proxy)
        assert mock_validate_has_no_nan.call_args_list[0][0][1] == "y_proxy"
        np.testing.assert_array_equal(mock_validate_has_no_nan.call_args_list[1][0][0], clusters)
        assert mock_validate_has_no_nan.call_args_list[1][0][1] == "clusters"

        mock_validate_bounds.assert_has_calls(
            [
                call(
                    0,
                    "clusters_intersection",
                    upper=0,
                    error_message="Cluster 'None' contains both labeled and unlabeled observations.",
                ),
                call(2, "n_labeled_clusters", lower=2, error_message="Need at least 2 fully labeled clusters; got 2."),
                call(
                    2, "n_unlabeled_clusters", lower=2, error_message="Need at least 2 fully unlabeled clusters; got 2."
                ),
            ]
        )


def test_preprocess_valid_output(
    y_true, y_proxy, clusters, labeled_true_sums, labeled_proxy_sums, unlabeled_proxy_sums
):

    lt_sums, lp_sums, up_sums, l_sizes, u_sizes = _preprocess(y_true, y_proxy, clusters)

    np.testing.assert_array_equal(lt_sums, labeled_true_sums)
    np.testing.assert_array_equal(lp_sums, labeled_proxy_sums)
    np.testing.assert_array_equal(up_sums, unlabeled_proxy_sums)
    np.testing.assert_array_equal(l_sizes, np.array([2, 2]))
    np.testing.assert_array_equal(u_sizes, np.array([3, 3]))


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
