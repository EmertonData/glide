from typing import Tuple
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

import glide.estimators.multi_ptd as multi_ptd_module
from glide.confidence_intervals import BootstrapConfidenceInterval
from glide.estimators import MultiPTDMeanEstimator
from glide.mean_inference_results import PredictionPoweredMeanInferenceResult

# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def estimator() -> MultiPTDMeanEstimator:
    return MultiPTDMeanEstimator()


@pytest.fixture
def y_arrays() -> Tuple[NDArray, NDArray]:
    y_true = np.array([5.0, 6.0, 7.0, np.nan, np.nan, np.nan])
    y_proxies = np.array([[4.5, 4.0], [5.5, 6.0], [6.5, 7.0], [6.0, 6.0], [7.0, 7.0], [8.0, 9.0]])
    return y_true, y_proxies


# ── _preprocess ───────────────────────────────────────────────────────────────


def test_preprocess_valid_output(estimator, y_arrays):
    y_true_all, y_proxies_all = y_arrays
    y_true, y_proxies_labeled, y_proxies_unlabeled = estimator._preprocess(y_true_all, y_proxies_all)
    np.testing.assert_array_equal(y_true, np.array([5.0, 6.0, 7.0]))
    np.testing.assert_array_equal(y_proxies_labeled, np.array([[4.5, 4.0], [5.5, 6.0], [6.5, 7.0]]))
    np.testing.assert_array_equal(y_proxies_unlabeled, np.array([[6.0, 6.0], [7.0, 7.0], [8.0, 9.0]]))


def test_preprocess_delegates(estimator, y_arrays):
    y_true, y_proxies = y_arrays
    labeled_mask = np.array([True, True, True, False, False, False])
    with (
        patch.object(multi_ptd_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(multi_ptd_module, "_validate_y_proxies") as mock_validate_y_proxies,
        patch.object(multi_ptd_module, "_validate_y_true") as mock_validate_y_true,
        patch.object(multi_ptd_module, "_split_labeled_unlabeled") as mock_split_labeled_unlabeled,
        patch.object(multi_ptd_module, "_validate_sample_sizes") as mock_validate_sample_sizes,
    ):
        mock_split_labeled_unlabeled.return_value = (
            np.array([5.0, 6.0, 7.0]),
            np.array([[4.5, 4.0], [5.5, 6.0], [6.5, 7.0]]),
            np.array([[6.0, 6.0], [7.0, 7.0], [8.0, 9.0]]),
            labeled_mask,
        )
        estimator._preprocess(y_true, y_proxies)

        mock_validate_equal_lengths.assert_called_once_with(y_true, y_proxies, names=["y_true", "y_proxies"])
        mock_validate_y_proxies.assert_called_once_with(y_proxies)
        mock_validate_y_true.assert_called_once_with(y_true)
        mock_split_labeled_unlabeled.assert_called_once()
        np.testing.assert_array_equal(mock_split_labeled_unlabeled.call_args[0][0], y_true)
        np.testing.assert_array_equal(mock_split_labeled_unlabeled.call_args[0][1], y_proxies)
        mock_validate_sample_sizes.assert_called_once()
        np.testing.assert_array_equal(mock_validate_sample_sizes.call_args[0][0], labeled_mask)


# ── estimate ──────────────────────────────────────────────────────────────────


def test_estimate_returns_valid_inference_result(estimator, y_arrays):
    y_true, y_proxies = y_arrays
    result = estimator.estimate(y_true, y_proxies, n_bootstrap=5, random_seed=0)
    assert isinstance(result, PredictionPoweredMeanInferenceResult)
    assert isinstance(result.confidence_interval, BootstrapConfidenceInterval)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "MultiPTDMeanEstimator"


def test_estimate_metadata(estimator, y_arrays):
    y_true, y_proxies = y_arrays
    result = estimator.estimate(y_true, y_proxies, n_bootstrap=5, metric_name="accuracy", random_seed=0)
    assert result.metric_name == "accuracy"
    assert result.estimator_name == "MultiPTDMeanEstimator"
    assert result.n_true == 3
    assert result.n_proxy == 6
    assert result.effective_sample_size == 4


def test_estimate_custom_confidence_level(estimator, y_arrays):
    y_true, y_proxies = y_arrays
    result = estimator.estimate(
        y_true, y_proxies, metric_name="perf", confidence_level=0.90, n_bootstrap=5, random_seed=0
    )

    expected_mean = 5.919
    expected_std = 0.457
    expected_lower = 5.515
    expected_upper = 6.521

    assert result.confidence_interval.confidence_level == 0.90
    assert result.confidence_interval.mean == pytest.approx(expected_mean, abs=0.001)
    assert result.std == pytest.approx(expected_std, abs=0.001)
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=0.001)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=0.001)


def test_estimate_reproducibility(estimator, y_arrays):
    y_true, y_proxies = y_arrays
    result_a = estimator.estimate(y_true, y_proxies, n_bootstrap=5, random_seed=7)
    result_b = estimator.estimate(y_true, y_proxies, n_bootstrap=5, random_seed=7)
    assert result_a.confidence_interval.lower_bound == result_b.confidence_interval.lower_bound
    assert result_a.confidence_interval.upper_bound == result_b.confidence_interval.upper_bound


def test_estimate_different_seeds_results_differ(estimator, y_arrays):
    y_true, y_proxies = y_arrays
    result_a = estimator.estimate(y_true, y_proxies, n_bootstrap=5, random_seed=0)
    result_b = estimator.estimate(y_true, y_proxies, n_bootstrap=5, random_seed=1)
    assert (
        result_a.confidence_interval.lower_bound != result_b.confidence_interval.lower_bound
        or result_a.confidence_interval.upper_bound != result_b.confidence_interval.upper_bound
    )


# --- __str__ / __repr__ ---


def test_str_format(estimator, y_arrays):
    y_true, y_proxies = y_arrays
    result = estimator.estimate(y_true, y_proxies, n_bootstrap=5, metric_name="performance", random_seed=0)
    output = str(result)
    expected = (
        "Metric: performance\n"
        "Point Estimate: 5.919\n"
        "Confidence Interval (95%): [5.507, 6.576]\n"
        "Estimator : MultiPTDMeanEstimator\n"
        "n_true: 3\n"
        "n_proxy: 6\n"
        "Effective Sample Size: 4"
    )
    assert output == expected


def test_repr_equals_str(estimator, y_arrays):
    y_true, y_proxies = y_arrays
    result = estimator.estimate(y_true, y_proxies, n_bootstrap=5, metric_name="performance", random_seed=0)
    assert repr(result) == str(result)
