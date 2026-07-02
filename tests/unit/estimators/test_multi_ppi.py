from typing import Tuple
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

import glide.estimators.multi_ppi as multi_ppi_module
from glide.confidence_intervals import CLTConfidenceInterval
from glide.estimators import MultiPPIMeanEstimator
from glide.mean_inference_results import PredictionPoweredMeanInferenceResult

# ── helpers ────────────────────────────────────────────────────────────────────


@pytest.fixture
def y_arrays() -> Tuple[NDArray, NDArray]:
    y_true = np.array([1.0, 2.0, np.nan, np.nan])
    y_proxies = np.array([[1.0, 0.0], [2.0, 2.0], [3.0, 1.0], [4.0, 3.0]])
    return y_true, y_proxies


@pytest.fixture
def estimator() -> MultiPPIMeanEstimator:
    return MultiPPIMeanEstimator()


# --- _preprocess ---


def test_preprocess_delegates(estimator, y_arrays):
    y_true, y_proxies = y_arrays
    labeled_mask = np.array([True, True, False, False])
    with (
        patch.object(multi_ppi_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(multi_ppi_module, "_validate_y_proxies") as mock_validate_y_proxies,
        patch.object(multi_ppi_module, "_validate_y_true") as mock_validate_y_true,
        patch.object(multi_ppi_module, "_split_labeled_unlabeled") as mock_split_labeled_unlabeled,
        patch.object(multi_ppi_module, "_validate_sample_sizes") as mock_validate_sample_sizes,
    ):
        mock_split_labeled_unlabeled.return_value = (
            np.array([1.0, 2.0]),
            np.array([[1.0, 0.0], [2.0, 2.0]]),
            np.array([[3.0, 1.0], [4.0, 3.0]]),
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


def test_preprocess_valid_output(estimator, y_arrays):
    y_true_all, y_proxies_all = y_arrays
    y_true, y_proxies_labeled, y_proxies_unlabeled = estimator._preprocess(y_true_all, y_proxies_all)
    np.testing.assert_array_equal(y_true, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(y_proxies_labeled, np.array([[1.0, 0.0], [2.0, 2.0]]))
    np.testing.assert_array_equal(y_proxies_unlabeled, np.array([[3.0, 1.0], [4.0, 3.0]]))


# --- estimate ---


def test_estimate_is_valid_inference_result(estimator, y_arrays):
    y_true, y_proxies = y_arrays
    result = estimator.estimate(y_true, y_proxies)
    assert isinstance(result, PredictionPoweredMeanInferenceResult)
    assert isinstance(result.confidence_interval, CLTConfidenceInterval)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "MultiPPIMeanEstimator"


def test_estimate_metadata(estimator, y_arrays):
    y_true, y_proxies = y_arrays
    result = estimator.estimate(y_true, y_proxies, metric_name="accuracy")
    assert result.metric_name == "accuracy"
    assert result.estimator_name == "MultiPPIMeanEstimator"
    assert result.n_true == 2
    assert result.n_proxy == 4
    assert result.effective_sample_size == 3


def test_estimate_custom_confidence_level(estimator, y_arrays):
    y_true, y_proxies = y_arrays
    result = estimator.estimate(y_true, y_proxies, confidence_level=0.85)

    expected_mean = 1.5
    expected_std = 0.395
    expected_lower = 0.931
    expected_upper = 2.069

    assert result.confidence_interval.confidence_level == 0.85
    assert result.confidence_interval.mean == pytest.approx(expected_mean, abs=0.01)
    assert result.std == pytest.approx(expected_std, abs=0.01)
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=0.01)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=0.01)


# --- __str__ / __repr__ ---


def test_str_format(estimator, y_arrays):
    y_true, y_proxies = y_arrays
    result = estimator.estimate(y_true, y_proxies, metric_name="performance")

    output = str(result)
    expected = (
        "Metric: performance\n"
        "Point Estimate: 1.500\n"
        "Confidence Interval (95%): [0.725, 2.275]\n"
        "Estimator : MultiPPIMeanEstimator\n"
        "n_true: 2\n"
        "n_proxy: 4\n"
        "Effective Sample Size: 3"
    )
    assert output == expected


def test_repr_equals_str(estimator, y_arrays):
    y_true, y_proxies = y_arrays
    result = estimator.estimate(y_true, y_proxies, metric_name="perf")
    assert repr(result) == str(result)
