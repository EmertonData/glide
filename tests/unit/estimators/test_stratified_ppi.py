from math import floor

import numpy as np
import pytest

from glide.confidence_intervals import CLTConfidenceInterval
from glide.estimators import StratifiedPPIMeanEstimator
from glide.mean_inference_results import PredictionPoweredMeanInferenceResult

# ── helpers ────────────────────────────────────────────────────────────────────


@pytest.fixture
def y_true() -> np.ndarray:
    return np.array([5.0, 6.0, np.nan, np.nan, 5.0, 6.0, np.nan, np.nan])


@pytest.fixture
def y_proxy() -> np.ndarray:
    return np.array([4.9, 6.1, 5.2, 6.1, 4.9, 6.1, 5.2, 6.1])


@pytest.fixture
def groups() -> np.ndarray:
    return np.array(["A", "A", "A", "A", "B", "B", "B", "B"])


@pytest.fixture
def estimator() -> StratifiedPPIMeanEstimator:
    return StratifiedPPIMeanEstimator()


# --- estimate ---


def test_estimate_returns_valid_inference_result(estimator, y_true, y_proxy, groups):
    result = estimator.estimate(y_true, y_proxy, groups)
    assert isinstance(result, PredictionPoweredMeanInferenceResult)
    assert isinstance(result.confidence_interval, CLTConfidenceInterval)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "StratifiedPPIMeanEstimator"


def test_estimate_metadata(estimator, y_true, y_proxy, groups):
    result = estimator.estimate(y_true, y_proxy, groups, metric_name="performance")
    assert result.metric_name == "performance"
    assert result.estimator_name == estimator.__class__.__name__
    assert result.n_true == 4
    assert result.n_proxy == 8
    assert result.effective_sample_size == 7


def test_estimate_custom_confidence_level(estimator, y_true, y_proxy, groups):
    result = estimator.estimate(y_true, y_proxy, groups, confidence_level=0.85)

    expected_mean = 5.618
    expected_std = 0.250
    expected_lower = 5.257
    expected_upper = 5.978

    assert result.confidence_interval.confidence_level == 0.85
    assert result.confidence_interval.mean == pytest.approx(expected_mean, abs=0.01)
    assert result.std == pytest.approx(expected_std, abs=0.01)
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=0.01)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=0.01)


# --- __str__ / __repr__ ---


def test_str_format(estimator, y_true, y_proxy, groups):
    result = estimator.estimate(y_true, y_proxy, groups, metric_name="performance")
    output = str(result)
    expected = (
        "Metric: performance\n"
        "Point Estimate: 5.618\n"
        "Confidence Interval (95%): [5.127, 6.108]\n"
        "Estimator : StratifiedPPIMeanEstimator\n"
        "n_true: 4\n"
        "n_proxy: 8\n"
        "Effective Sample Size: 7"
    )
    assert output == expected


def test_repr_equals_str(estimator, y_true, y_proxy, groups):
    result = estimator.estimate(y_true, y_proxy, groups, metric_name="perf")
    assert repr(result) == str(result)


def test_estimate_ess_uses_stratified_baseline(estimator, groups):
    y_true = np.array([1.0, 9.0, np.nan, np.nan, 5.0, 6.0, np.nan, np.nan])
    y_proxy = np.array([1.5, 8.5, 5.0, 7.0, 4.9, 6.1, 5.2, 6.1])

    result = estimator.estimate(y_true, y_proxy, groups)

    y_labeled = np.array([1.0, 9.0, 5.0, 6.0])
    naive_ess = floor(np.var(y_labeled, ddof=1) / result.confidence_interval.var)

    assert result.effective_sample_size != naive_ess


def test_estimate_ess_uses_total_proportional_weights(estimator):
    y_true = np.array([1.0, 9.0, np.nan, np.nan, 5.0, 6.0, np.nan, np.nan, np.nan, np.nan])
    y_proxy = np.array([1.1, 8.9, 2.0, 3.0, 5.1, 5.9, 6.0, 7.0, 8.0, 9.0])
    groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B", "B", "B"])

    result = estimator.estimate(y_true, y_proxy, groups)

    labeled_weights_ess = floor(4 * 4.0625 / result.confidence_interval.var)

    assert result.effective_sample_size != labeled_weights_ess
