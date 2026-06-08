from typing import Tuple
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

import glide.estimators.ppi as ppi_module
from glide.confidence_intervals import CLTConfidenceInterval
from glide.estimators import PPIMeanEstimator
from glide.mean_inference_results import PredictionPoweredMeanInferenceResult

# ── helpers ────────────────────────────────────────────────────────────────────


@pytest.fixture
def y_arrays() -> Tuple[NDArray, NDArray]:
    y_true = np.array([1.0, 2.0, np.nan, np.nan])
    y_proxy = np.array([1.0, 2.0, 3.0, 4.0])
    return y_true, y_proxy


@pytest.fixture
def estimator() -> PPIMeanEstimator:
    return PPIMeanEstimator()


# --- _preprocess ---


def test_preprocess_delegates_to_validation(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    with (
        patch.object(ppi_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(ppi_module, "_validate_y_proxy") as mock_validate_y_proxy,
        patch.object(ppi_module, "_validate_y_true") as mock_validate_y_true,
        patch.object(ppi_module, "_validate_sample_sizes") as mock_validate_sample_sizes,
    ):
        estimator._preprocess(y_true, y_proxy)

        mock_validate_equal_lengths.assert_called_once_with(y_true, y_proxy, names=["y_true", "y_proxy"])
        mock_validate_y_proxy.assert_called_once_with(y_proxy)
        mock_validate_y_true.assert_called_once_with(y_true)
        mock_validate_sample_sizes.assert_called_once()
        labeled_mask_arg = mock_validate_sample_sizes.call_args[0][0]
        np.testing.assert_array_equal(labeled_mask_arg, np.array([True, True, False, False]))


def test_preprocess_valid_output(estimator, y_arrays):
    y_true_all, y_proxy_all = y_arrays
    y_true, y_proxy_labeled, y_proxy_unlabeled = estimator._preprocess(y_true_all, y_proxy_all)
    np.testing.assert_array_equal(y_true, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(y_proxy_labeled, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(y_proxy_unlabeled, np.array([3.0, 4.0]))


# --- estimate ---


def test_estimate_is_valid_inference_result(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy)
    assert isinstance(result, PredictionPoweredMeanInferenceResult)
    assert isinstance(result.confidence_interval, CLTConfidenceInterval)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "PPIMeanEstimator"


def test_estimate_metadata(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy, metric_name="performance")
    assert result.metric_name == "performance"
    assert result.estimator_name == estimator.__class__.__name__
    assert result.n_true == 2
    assert result.n_proxy == 4
    assert result.effective_sample_size == 2


def test_estimate_custom_confidence_level(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy, metric_name="perf", confidence_level=0.90)

    expected_mean = 1.8
    expected_std = 0.431
    expected_lower = 1.09
    expected_upper = 2.51

    assert result.confidence_interval.confidence_level == 0.90
    assert result.confidence_interval.mean == pytest.approx(expected_mean, abs=0.01)
    assert result.std == pytest.approx(expected_std, abs=0.01)
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=0.01)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=0.01)


# --- __str__ / __repr__ ---


def test_str_format(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy, metric_name="performance")
    output = str(result)
    expected = (
        "Metric: performance\n"
        "Point Estimate: 1.800\n"
        "Confidence Interval (95%): [0.954, 2.646]\n"
        "Estimator : PPIMeanEstimator\n"
        "n_true: 2\n"
        "n_proxy: 4\n"
        "Effective Sample Size: 2"
    )
    assert output == expected


def test_repr_equals_str(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy, metric_name="perf")
    assert repr(result) == str(result)
