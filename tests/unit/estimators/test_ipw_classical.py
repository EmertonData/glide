import numpy as np
import pytest
from numpy.typing import NDArray

from glide.core.mean_inference_result import ClassicalMeanInferenceResult
from glide.estimators.ipw_classical import IPWClassicalMeanEstimator


@pytest.fixture
def estimator() -> IPWClassicalMeanEstimator:
    return IPWClassicalMeanEstimator()


@pytest.fixture
def y() -> NDArray:
    return np.array([2.0, 4.0, np.nan, np.nan])


@pytest.fixture
def sampling_probability() -> NDArray:
    return np.array([0.5, 0.5, 0.5, 0.5])


# --- _compute_ipw_weighted_values ---


def test_compute_ipw_weighted_values_known_values(estimator):
    y = np.array([2.0, np.nan])
    pi = np.array([0.5, 0.5])
    result = estimator._compute_ipw_weighted_values(y, pi)
    np.testing.assert_array_equal(result, np.array([4.0, 0.0]))


# --- estimate ---


def test_estimate_is_valid_inference_result(estimator, y, sampling_probability):
    result = estimator.estimate(y, sampling_probability)
    assert isinstance(result, ClassicalMeanInferenceResult)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "IPWClassicalMeanEstimator"


def test_estimate_metadata(estimator, y, sampling_probability):
    result = estimator.estimate(y, sampling_probability, metric_name="score")
    assert result.metric_name == "score"
    assert result.estimator_name == estimator.__class__.__name__
    assert result.n == 2


def test_estimate_custom_confidence_level(estimator, y, sampling_probability):
    result = estimator.estimate(y, sampling_probability, confidence_level=0.85)
    assert result.confidence_interval.confidence_level == 0.85
    assert result.confidence_interval.mean == pytest.approx(3.0)
    assert result.std == pytest.approx(1.915, abs=0.001)
    assert result.confidence_interval.lower_bound == pytest.approx(0.243, abs=0.001)
    assert result.confidence_interval.upper_bound == pytest.approx(5.756, abs=0.001)


@pytest.mark.parametrize("bad_pi", [2.0, -0.5])
def test_estimate_raises_on_non_positive_sampling_probability(estimator, y, bad_pi):
    pi = np.array([0.5, 0.5, 0.5, bad_pi])
    with pytest.raises(ValueError, match="Sampling probabilities should be in \\(0, 1]"):
        estimator.estimate(y, pi)


# --- __str__ / __repr__ ---


def test_str_format(estimator, y, sampling_probability):
    result = estimator.estimate(y, sampling_probability, metric_name="accuracy")
    output = str(result)
    expected = (
        "Metric: accuracy\n"
        "Point Estimate: 3.000\n"
        "Confidence Interval (95%): [-0.753, 6.753]\n"
        "Estimator : IPWClassicalMeanEstimator\n"
        "n: 2"
    )
    assert output == expected


def test_repr_equals_str(estimator, y, sampling_probability):
    result = estimator.estimate(y, sampling_probability, metric_name="perf")
    assert repr(result) == str(result)
