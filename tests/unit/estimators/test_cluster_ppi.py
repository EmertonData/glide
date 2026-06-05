import numpy as np
import pytest
from numpy.typing import NDArray

from glide.confidence_intervals import CLTConfidenceInterval
from glide.estimators import ClusterPPIMeanEstimator
from glide.mean_inference_results import PredictionPoweredMeanInferenceResult


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
def estimator() -> ClusterPPIMeanEstimator:
    return ClusterPPIMeanEstimator()


# --- estimate ----


def test_estimate_returns_valid_inference_result(estimator, y_true, y_proxy, clusters):
    result = estimator.estimate(y_true, y_proxy, clusters)
    assert isinstance(result, PredictionPoweredMeanInferenceResult)
    assert isinstance(result.confidence_interval, CLTConfidenceInterval)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "ClusterPPIMeanEstimator"


def test_estimate_metadata(estimator, y_true, y_proxy, clusters):
    result = estimator.estimate(y_true, y_proxy, clusters, metric_name="performance")
    assert result.metric_name == "performance"
    assert result.estimator_name == estimator.__class__.__name__
    assert result.n_true == 2
    assert result.n_proxy == 4
    assert result.effective_sample_size == 6


def test_estimate_custom_confidence_level(estimator, y_true, y_proxy, clusters):
    result = estimator.estimate(y_true, y_proxy, clusters, confidence_level=0.90)

    expected_mean = 61 / 11
    expected_std = np.sqrt(37) / 11
    expected_lower = 4.636
    expected_upper = 6.455

    assert result.confidence_interval.confidence_level == 0.90
    assert result.confidence_interval.mean == pytest.approx(expected_mean, abs=1e-10)
    assert result.std == pytest.approx(expected_std, abs=1e-10)
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=1e-3)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=1e-3)


# --- __str__ / __repr__ ---


def test_str_format(estimator, y_true, y_proxy, clusters):
    result = estimator.estimate(y_true, y_proxy, clusters, metric_name="performance")
    output = str(result)
    expected = (
        "Metric: performance\n"
        "Point Estimate: 5.545\n"
        "Confidence Interval (95%): [4.462, 6.629]\n"
        "Estimator : ClusterPPIMeanEstimator\n"
        "n_true: 2\n"
        "n_proxy: 4\n"
        "Effective Sample Size: 6"
    )
    assert output == expected


def test_repr_equals_str(estimator, y_true, y_proxy, clusters):
    result = estimator.estimate(y_true, y_proxy, clusters, metric_name="perf")
    assert repr(result) == str(result)
