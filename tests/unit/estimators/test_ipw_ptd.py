from typing import Tuple
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

import glide.estimators.ipw_ptd as ipw_ptd_module
from glide.confidence_intervals import BootstrapConfidenceInterval
from glide.estimators import IPWPTDMeanEstimator
from glide.mean_inference_results import PredictionPoweredMeanInferenceResult

# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def estimator() -> IPWPTDMeanEstimator:
    return IPWPTDMeanEstimator()


@pytest.fixture
def y_arrays() -> Tuple[NDArray, NDArray, NDArray]:
    y_true = np.array([5.0, 6.0, 7.0, np.nan, np.nan, np.nan])
    y_proxy = np.array([4.5, 5.5, 6.5, 6.0, 7.0, 8.0])
    pi = np.array([0.4, 0.6, 0.5, 0.5, 0.2, 0.8])
    return y_true, y_proxy, pi


# ── _preprocess ───────────────────────────────────────────────────────────────


def test_preprocess_valid_output(estimator, y_arrays):
    y_true_all, y_proxy_all, pi_all = y_arrays
    y_true_filled, y_proxy, xi, pi = estimator._preprocess(y_true_all, y_proxy_all, pi_all)
    assert len(y_true_filled) == 6
    assert len(y_proxy) == 6
    assert len(pi) == 6
    assert np.sum(np.isnan(y_true_filled)) == 0
    assert len(xi) == 6
    assert np.isin(xi, [0.0, 1.0]).all()
    assert np.sum(xi) == 3


def test_preprocess_delegates_to_validation(estimator):
    y_true = np.array([1.0, 2.0, np.nan, np.nan])
    y_proxy = np.array([1.0, 2.0, 3.0, 4.0])
    pi = np.array([0.5, 0.5, 0.5, 0.5])

    with (
        patch.object(ipw_ptd_module, "_validate_equal_lengths") as mock_equal_lengths,
        patch.object(ipw_ptd_module, "_validate_probabilities") as mock_sampling_probs,
        patch.object(ipw_ptd_module, "_validate_y_proxy") as mock_y_proxy,
        patch.object(ipw_ptd_module, "_validate_y_true") as mock_y_true,
        patch.object(ipw_ptd_module, "_validate_label_prob_consistency") as mock_pi_consistency,
        patch.object(ipw_ptd_module, "_validate_sample_sizes") as mock_sample_sizes,
    ):
        estimator._preprocess(y_true, y_proxy, pi)

        mock_equal_lengths.assert_called_with(y_true, y_proxy, pi, names=["y_true", "y_proxy", "pi"])
        mock_sampling_probs.assert_called_with(pi)
        mock_y_proxy.assert_called_with(y_proxy)
        mock_y_true.assert_called_with(y_true)
        y_true_non_nan_mask = ~np.isnan(y_true)
        np.testing.assert_array_equal(mock_pi_consistency.call_args[0][0], y_true_non_nan_mask)
        np.testing.assert_array_equal(mock_pi_consistency.call_args[0][1], pi)
        assert mock_sample_sizes.call_count == 1
        np.testing.assert_array_equal(mock_sample_sizes.call_args[0][0], y_true_non_nan_mask)


# ── estimate ──────────────────────────────────────────────────────────────────


def test_estimate_returns_valid_inference_result(estimator, y_arrays):
    y_true, y_proxy, pi = y_arrays
    result = estimator.estimate(y_true, y_proxy, pi, n_bootstrap=5, random_seed=0)
    assert isinstance(result, PredictionPoweredMeanInferenceResult)
    assert isinstance(result.confidence_interval, BootstrapConfidenceInterval)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "IPWPTDMeanEstimator"


def test_estimate_metadata(estimator, y_arrays):
    y_true, y_proxy, pi = y_arrays
    result = estimator.estimate(y_true, y_proxy, pi, n_bootstrap=5, metric_name="accuracy", random_seed=2)
    assert result.metric_name == "accuracy"
    assert result.estimator_name == "IPWPTDMeanEstimator"
    assert result.n_true == 3
    assert result.n_proxy == 6
    assert result.effective_sample_size == 13


def test_estimate_custom_confidence_level(estimator, y_arrays):
    y_true, y_proxy, pi = y_arrays
    result = estimator.estimate(
        y_true, y_proxy, pi, metric_name="perf", confidence_level=0.90, n_bootstrap=5, random_seed=0
    )

    expected_mean = 4.586
    expected_std = 2.118
    expected_lower = 1.666
    expected_upper = 5.950

    assert result.confidence_interval.confidence_level == 0.90
    assert result.confidence_interval.mean == pytest.approx(expected_mean, abs=0.01)
    assert result.std == pytest.approx(expected_std, abs=0.01)
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=0.01)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=0.01)


def test_estimate_reproducibility(estimator, y_arrays):
    y_true, y_proxy, pi = y_arrays
    result_a = estimator.estimate(y_true, y_proxy, pi, n_bootstrap=5, random_seed=7)
    result_b = estimator.estimate(y_true, y_proxy, pi, n_bootstrap=5, random_seed=7)
    assert result_a.confidence_interval.lower_bound == result_b.confidence_interval.lower_bound
    assert result_a.confidence_interval.upper_bound == result_b.confidence_interval.upper_bound


def test_estimate_warns_on_zero_pi(estimator, y_arrays):
    y_true, y_proxy, _ = y_arrays
    pi = np.array([0.4, 0.6, 0.5, 0.0, 0.2, 0.8])
    with pytest.warns(UserWarning, match="Some observations have pi=0"):
        estimator.estimate(y_true, y_proxy, pi, n_bootstrap=5, random_seed=7)


def test_estimate_warns_on_one_pi(estimator, y_arrays):
    y_true, y_proxy, _ = y_arrays
    pi = np.array([0.4, 0.6, 1.0, 0.5, 0.2, 0.8])
    with pytest.warns(UserWarning, match="Some observations have pi=1"):
        estimator.estimate(y_true, y_proxy, pi, n_bootstrap=5, random_seed=7)


# ── __str__ / __repr__ ────────────────────────────────────────────────────────


def test_str_format(estimator, y_arrays):
    y_true, y_proxy, pi = y_arrays
    result = estimator.estimate(y_true, y_proxy, pi, n_bootstrap=5, metric_name="performance", random_seed=0)
    output = str(result)
    expected = (
        "Metric: performance\n"
        "Point Estimate: 4.587\n"
        "Confidence Interval (95%): [1.278, 5.968]\n"
        "Estimator : IPWPTDMeanEstimator\n"
        "n_true: 3\n"
        "n_proxy: 6\n"
        "Effective Sample Size: 5"
    )
    assert output == expected


def test_repr_equals_str(estimator, y_arrays):
    y_true, y_proxy, pi = y_arrays
    result = estimator.estimate(y_true, y_proxy, pi, n_bootstrap=5, metric_name="performance", random_seed=0)
    assert repr(result) == str(result)
