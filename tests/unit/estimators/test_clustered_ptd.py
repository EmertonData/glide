import numpy as np
import pytest
from numpy.typing import NDArray

from glide.confidence_intervals import BootstrapConfidenceInterval
from glide.estimators import ClusteredPTDMeanEstimator
from glide.mean_inference_results import PredictionPoweredMeanInferenceResult


@pytest.fixture
def y_true() -> NDArray:
    return np.array([1.0, 2.0, 3.0, 4.0, np.nan, np.nan, np.nan, np.nan])


@pytest.fixture
def y_proxy() -> NDArray:
    return np.array([1.1, 2.2, 3.1, 3.9, 1.5, 1.8, 4.5, 4.8])


@pytest.fixture
def clusters() -> NDArray:
    return np.array(["A", "A", "B", "B", "C", "C", "D", "D"])


@pytest.fixture
def estimator() -> ClusteredPTDMeanEstimator:
    return ClusteredPTDMeanEstimator()


# --- estimate ---


def test_estimate_returns_valid_inference_result(estimator, y_true, y_proxy, clusters):
    result = estimator.estimate(y_true, y_proxy, clusters, n_bootstrap=5, random_seed=0)
    assert isinstance(result, PredictionPoweredMeanInferenceResult)
    assert isinstance(result.confidence_interval, BootstrapConfidenceInterval)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "ClusteredPTDMeanEstimator"


def test_estimate_metadata(estimator, y_true, y_proxy, clusters):
    result = estimator.estimate(y_true, y_proxy, clusters, metric_name="accuracy", n_bootstrap=5, random_seed=0)
    assert result.metric_name == "accuracy"
    assert result.estimator_name == "ClusteredPTDMeanEstimator"
    assert result.n_true == 4
    assert result.n_proxy == 8
    assert result.effective_sample_size == 6


def test_estimate_custom_confidence_level(estimator, y_true, y_proxy, clusters):
    result = estimator.estimate(y_true, y_proxy, clusters, n_bootstrap=5, confidence_level=0.85, random_seed=0)

    expected_mean = 2.516
    expected_std = 0.779
    expected_lower = 1.769
    expected_upper = 3.403

    assert result.confidence_interval.confidence_level == 0.85
    assert result.confidence_interval.mean == pytest.approx(expected_mean, abs=0.01)
    assert result.std == pytest.approx(expected_std, abs=0.01)
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=0.01)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=0.01)


def test_estimate_reproducibility(estimator, y_true, y_proxy, clusters):
    result_a = estimator.estimate(y_true, y_proxy, clusters, n_bootstrap=5, random_seed=7)
    result_b = estimator.estimate(y_true, y_proxy, clusters, n_bootstrap=5, random_seed=7)
    assert result_a.confidence_interval.lower_bound == result_b.confidence_interval.lower_bound
    assert result_a.confidence_interval.upper_bound == result_b.confidence_interval.upper_bound


def test_estimate_different_seeds_results_differ(estimator, y_true, y_proxy, clusters):
    result_a = estimator.estimate(y_true, y_proxy, clusters, n_bootstrap=5, random_seed=0)
    result_b = estimator.estimate(y_true, y_proxy, clusters, n_bootstrap=5, random_seed=1)
    assert (
        result_a.confidence_interval.lower_bound != result_b.confidence_interval.lower_bound
        or result_a.confidence_interval.upper_bound != result_b.confidence_interval.upper_bound
    )


# --- __str__ / __repr__ ---


def test_str_format(estimator, y_true, y_proxy, clusters):
    result = estimator.estimate(y_true, y_proxy, clusters, n_bootstrap=5, random_seed=0)
    output = str(result)
    expected = (
        "Metric: Metric\n"
        "Point Estimate: 2.517\n"
        "Confidence Interval (95%): [1.657, 3.497]\n"
        "Estimator : ClusteredPTDMeanEstimator\n"
        "n_true: 4\n"
        "n_proxy: 8\n"
        "Effective Sample Size: 6"
    )
    assert output == expected


def test_repr_equals_str(estimator, y_true, y_proxy, clusters):
    result = estimator.estimate(y_true, y_proxy, clusters, n_bootstrap=5, random_seed=0)
    assert repr(result) == str(result)
