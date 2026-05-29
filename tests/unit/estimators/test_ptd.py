from typing import Tuple
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

import glide.estimators.ptd as ptd_module
from glide.confidence_intervals import BootstrapConfidenceInterval
from glide.estimators import PTDMeanEstimator
from glide.mean_inference_results import PredictionPoweredMeanInferenceResult

# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def estimator() -> PTDMeanEstimator:
    return PTDMeanEstimator()


@pytest.fixture
def y_arrays() -> Tuple[NDArray, NDArray]:
    y_true = np.array([5.0, 6.0, 7.0, np.nan, np.nan, np.nan])
    y_proxy = np.array([4.5, 5.5, 6.5, 6.0, 7.0, 8.0])
    return y_true, y_proxy


# ── _preprocess ───────────────────────────────────────────────────────────────


def test_preprocess_valid_output(estimator, y_arrays):
    y_true_all, y_proxy_all = y_arrays
    y_true, y_proxy_labeled, y_proxy_unlabeled = estimator._preprocess(y_true_all, y_proxy_all)
    assert len(y_true) == 3
    assert len(y_proxy_labeled) == 3
    assert len(y_proxy_unlabeled) == 3
    assert not np.any(np.isnan(y_true))


def test_preprocess_delegates_to_validation(estimator):
    y_true = np.array([1.0, 2.0, np.nan, np.nan])
    y_proxy = np.array([1.0, 2.0, 3.0, 4.0])

    with (
        patch.object(ptd_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(ptd_module, "_validate_y_proxy") as mock_validate_y_proxy,
        patch.object(ptd_module, "_validate_y_true") as mock_validate_y_true,
        patch.object(ptd_module, "_validate_sample_sizes") as mock_validate_sample_sizes,
    ):
        estimator._preprocess(y_true, y_proxy)

        mock_validate_equal_lengths.assert_called_once_with(y_true, y_proxy, names=["y_true", "y_proxy"])
        mock_validate_y_proxy.assert_called_once_with(y_proxy)
        mock_validate_y_true.assert_called_once_with(y_true)
        mock_validate_sample_sizes.assert_called_once()
        labeled_mask_arg = mock_validate_sample_sizes.call_args[0][0]
        np.testing.assert_array_equal(labeled_mask_arg, np.array([True, True, False, False]))


# ── estimate ──────────────────────────────────────────────────────────────────


def test_estimate_returns_valid_inference_result(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy, n_bootstrap=5, random_seed=0)
    assert isinstance(result, PredictionPoweredMeanInferenceResult)
    assert isinstance(result.confidence_interval, BootstrapConfidenceInterval)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "PTDMeanEstimator"


def test_estimate_metadata(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy, n_bootstrap=5, metric_name="accuracy", random_seed=0)
    assert result.metric_name == "accuracy"
    assert result.estimator_name == "PTDMeanEstimator"
    assert result.n_true == 3
    assert result.n_proxy == 6
    assert result.effective_sample_size == 4


def test_estimate_custom_confidence_level(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(
        y_true, y_proxy, metric_name="perf", confidence_level=0.90, n_bootstrap=5, random_seed=0
    )

    expected_mean = 6.572
    expected_std = 0.453
    expected_lower = 6.176
    expected_upper = 7.152

    assert result.confidence_interval.confidence_level == 0.90
    assert result.confidence_interval.mean == pytest.approx(expected_mean, abs=0.01)
    assert result.std == pytest.approx(expected_std, abs=0.01)
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=0.01)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=0.01)


def test_estimate_reproducibility(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result_a = estimator.estimate(y_true, y_proxy, n_bootstrap=5, random_seed=7)
    result_b = estimator.estimate(y_true, y_proxy, n_bootstrap=5, random_seed=7)
    assert result_a.confidence_interval.lower_bound == result_b.confidence_interval.lower_bound
    assert result_a.confidence_interval.upper_bound == result_b.confidence_interval.upper_bound


# --- __str__ / __repr__ ---


def test_str_format(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy, n_bootstrap=5, metric_name="performance", random_seed=0)
    output = str(result)
    expected = (
        "Metric: performance\n"
        "Point Estimate: 6.572\n"
        "Confidence Interval (95%): [6.171, 7.192]\n"
        "Estimator : PTDMeanEstimator\n"
        "n_true: 3\n"
        "n_proxy: 6\n"
        "Effective Sample Size: 4"
    )
    assert output == expected


def test_repr_equals_str(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy, n_bootstrap=5, metric_name="performance", random_seed=0)
    assert repr(result) == str(result)
