from unittest.mock import patch

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
    return np.array([4.0, np.nan, 6.0, np.nan])


@pytest.fixture
def y_proxy() -> NDArray:
    return np.array([2.0, 4.0, 6.0, 6.0])


@pytest.fixture
def clusters() -> NDArray:
    return np.array(["A", "C", "B", "D"])


@pytest.fixture
def labeled_true_sums() -> NDArray:
    return np.array([4.0, 6.0])


@pytest.fixture
def labeled_proxy_sums() -> NDArray:
    return np.array([2.0, 6.0])


@pytest.fixture
def unlabeled_proxy_sums() -> NDArray:
    return np.array([4.0, 6.0])


# --- _preprocess ---


def test_preprocess_delegates_to_validation(y_true, y_proxy, clusters):
    with (
        patch.object(cluster_core_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(cluster_core_module, "_validate_has_no_nan") as mock_validate_has_no_nan,
    ):
        _preprocess(y_true, y_proxy, clusters)

        mock_validate_equal_lengths.assert_called_once()
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], y_true)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], y_proxy)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][2], clusters)
        assert mock_validate_equal_lengths.call_args[1] == {"names": ["y_true", "y_proxy", "clusters"]}

        mock_validate_has_no_nan.assert_called_once_with(y_proxy, "y_proxy")


def test_preprocess_valid_output(y_true, y_proxy, clusters):
    lt_means, lp_means, up_means, l_sizes, u_sizes = _preprocess(y_true, y_proxy, clusters)

    np.testing.assert_array_equal(lt_means, np.array([4.0, 6.0]))
    np.testing.assert_array_equal(lp_means, np.array([2.0, 6.0]))
    np.testing.assert_array_equal(up_means, np.array([4.0, 6.0]))
    np.testing.assert_array_equal(l_sizes, np.array([1, 1]))
    np.testing.assert_array_equal(u_sizes, np.array([1, 1]))


def test_preprocess_valid_output_multi_observation_clusters():
    y_t = np.array([1.0, 2.0, 3.0, 4.0, np.nan, np.nan, np.nan, np.nan])
    y_p = np.array([1.1, 2.2, 3.1, 3.9, 3.5, 4.5, 5.0, 6.0])
    cls = np.array(["A", "A", "B", "B", "C", "C", "D", "D"])

    lt_means, lp_means, up_means, l_sizes, u_sizes = _preprocess(y_t, y_p, cls)

    np.testing.assert_array_equal(lt_means, np.array([1.5, 3.5]))
    np.testing.assert_allclose(lp_means, np.array([1.65, 3.5]))
    np.testing.assert_array_equal(up_means, np.array([4.0, 5.5]))
    np.testing.assert_array_equal(l_sizes, np.array([2, 2]))
    np.testing.assert_array_equal(u_sizes, np.array([2, 2]))


def test_preprocess_raises_on_partial_cluster():
    y_t = np.array([5.0, np.nan, 6.0, np.nan])
    y_p = np.array([4.9, 5.1, 6.0, 6.0])
    cls = np.array(["A", "A", "B", "C"])

    with pytest.raises(ValueError, match="Cluster 'A'"):
        _preprocess(y_t, y_p, cls)


def test_preprocess_raises_on_too_few_labeled_clusters():
    y_t = np.array([5.0, np.nan, np.nan])
    y_p = np.array([4.9, 5.1, 6.0])
    cls = np.array(["A", "B", "C"])

    with pytest.raises(ValueError, match="at least 2 fully labeled"):
        _preprocess(y_t, y_p, cls)


def test_preprocess_raises_on_too_few_unlabeled_clusters():
    y_t = np.array([5.0, 6.0, np.nan])
    y_p = np.array([4.9, 5.9, 5.0])
    cls = np.array(["A", "B", "C"])

    with pytest.raises(ValueError, match="at least 2 fully unlabeled"):
        _preprocess(y_t, y_p, cls)


def test_preprocess_raises_on_nan_in_proxy(y_true, clusters):
    y_p_with_nan = np.array([2.0, np.nan, 6.0, 6.0])
    with pytest.raises(ValueError, match="y_proxy"):
        _preprocess(y_true, y_p_with_nan, clusters)


def test_preprocess_raises_on_unequal_lengths(y_true, y_proxy):
    short_clusters = np.array(["A", "B", "C"])
    with pytest.raises(ValueError):
        _preprocess(y_true, y_proxy, short_clusters)


# --- _compute_cluster_tuning_parameter ---


def test_compute_cluster_tuning_parameter_returns_one_when_power_tuning_false(
    labeled_true_sums, labeled_proxy_sums, unlabeled_proxy_sums
):
    result = _compute_cluster_tuning_parameter(
        labeled_true_sums, labeled_proxy_sums, unlabeled_proxy_sums, 2, 2, power_tuning=False
    )
    assert result == 1.0


def test_compute_cluster_tuning_parameter_known_value(labeled_true_sums, labeled_proxy_sums, unlabeled_proxy_sums):
    expected = 6 / 11
    result = _compute_cluster_tuning_parameter(
        labeled_true_sums, labeled_proxy_sums, unlabeled_proxy_sums, 2, 2, power_tuning=True
    )
    assert result == pytest.approx(expected, abs=1e-10)


def test_compute_cluster_tuning_parameter_raises_on_zero_variance():
    constant_sums = np.array([3.0, 3.0])
    with pytest.raises(ValueError, match="zero variance"):
        _compute_cluster_tuning_parameter(np.array([4.0, 6.0]), constant_sums, constant_sums, 2, 2, power_tuning=True)


# --- _compute_cluster_mean_estimate ---


def test_compute_cluster_mean_estimate_lambda_one(labeled_true_sums, labeled_proxy_sums, unlabeled_proxy_sums):
    expected = 6.0
    result = _compute_cluster_mean_estimate(
        labeled_true_sums, labeled_proxy_sums, unlabeled_proxy_sums, 2, 2, lambda_=1.0
    )
    assert result == pytest.approx(expected, abs=1e-10)


def test_compute_cluster_mean_estimate_known_value(labeled_true_sums, labeled_proxy_sums, unlabeled_proxy_sums):
    expected = 61 / 11
    result = _compute_cluster_mean_estimate(
        labeled_true_sums, labeled_proxy_sums, unlabeled_proxy_sums, 2, 2, lambda_=6 / 11
    )
    assert result == pytest.approx(expected, abs=1e-10)


# --- _compute_cluster_std_estimate ---


def test_compute_cluster_std_estimate_lambda_one(labeled_true_sums, labeled_proxy_sums, unlabeled_proxy_sums):
    expected = np.sqrt(2)
    result = _compute_cluster_std_estimate(
        labeled_true_sums, labeled_proxy_sums, unlabeled_proxy_sums, 2, 2, lambda_=1.0
    )
    assert result == pytest.approx(expected, abs=1e-10)


def test_compute_cluster_std_estimate_known_value(labeled_true_sums, labeled_proxy_sums, unlabeled_proxy_sums):
    expected = np.sqrt(37) / 11
    result = _compute_cluster_std_estimate(
        labeled_true_sums, labeled_proxy_sums, unlabeled_proxy_sums, 2, 2, lambda_=6 / 11
    )
    assert result == pytest.approx(expected, abs=1e-10)
