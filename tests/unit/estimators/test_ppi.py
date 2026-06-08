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
def y_arrays(n_true: int = 2, n_proxy: int = 2, seed: int = 42) -> Tuple[NDArray, NDArray]:
    rng = np.random.default_rng(seed)
    y_true = rng.normal(loc=5.0, scale=1.0, size=n_true)
    y_proxy_labeled = y_true + rng.normal(0, 0.5, size=n_true)
    y_proxy_unlabeled = rng.normal(loc=5.0, scale=1.0, size=n_proxy) + rng.normal(0, 0.5, size=n_proxy)
    y_true_all = np.hstack([y_true, np.full(n_proxy, np.nan)])
    y_proxy_all = np.hstack([y_proxy_labeled, y_proxy_unlabeled])
    return y_true_all, y_proxy_all


@pytest.fixture
def estimator() -> PPIMeanEstimator:
    return PPIMeanEstimator()


# --- _preprocess ---


def test_preprocess_delegates_to_validation(estimator):
    y_true = np.array([1.0, 2.0, np.nan, np.nan])
    y_proxy = np.array([1.0, 2.0, 3.0, 4.0])

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
    assert len(y_true) == 2
    assert len(y_proxy_labeled) == 2
    assert len(y_proxy_unlabeled) == 2
    assert not np.any(np.isnan(y_true))


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
    assert result.effective_sample_size == 4


def test_estimate_custom_confidence_level(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy, metric_name="perf", confidence_level=0.90)

    expected_mean = 4.07
    expected_std = 0.47
    expected_lower = 3.29
    expected_upper = 4.85

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
        "Point Estimate: 4.068\n"
        "Confidence Interval (95%): [3.140, 4.996]\n"
        "Estimator : PPIMeanEstimator\n"
        "n_true: 2\n"
        "n_proxy: 4\n"
        "Effective Sample Size: 4"
    )
    assert output == expected


def test_repr_equals_str(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy, metric_name="perf")
    assert repr(result) == str(result)
