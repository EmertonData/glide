from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

import glide.estimators.ipw_classical as ipw_classical_module
from glide.estimators import IPWClassicalMeanEstimator
from glide.mean_inference_results import ClassicalMeanInferenceResult


@pytest.fixture
def estimator() -> IPWClassicalMeanEstimator:
    return IPWClassicalMeanEstimator()


@pytest.fixture
def y() -> NDArray:
    return np.array([2.0, 4.0, np.nan, np.nan])


@pytest.fixture
def sampling_probability() -> NDArray:
    return np.array([0.5, 0.5, 0.5, 0.5])


# --- preprocessing ---


def test_preprocess_valid_output(estimator):
    y = np.array([1.0, 2.0, np.nan])
    sampling_probability = np.array([0.5, 0.5, 0.5])
    y_out, pi_out = estimator._preprocess(y, sampling_probability)
    assert len(y_out) == 3
    assert len(pi_out) == 3
    np.testing.assert_array_equal(y_out, np.array([1.0, 2.0, np.nan]))


def test_preprocess_delegates_to_validation(estimator):
    y = np.array([1.0, np.nan])
    pi = np.array([0.5, 0.5])

    with patch.object(ipw_classical_module, "_validate_sampling_probabilities") as mock_sampling_probs:
        estimator._preprocess(y, pi)
        mock_sampling_probs.assert_called_once_with(pi)


# --- estimate ---


def test_estimate_warns_on_zero_pi(estimator):
    y = np.array([1.0, 2.0, np.nan, np.nan, np.nan])
    pi = np.array([0.5, 0.5, 0.5, 0.5, 0.0])
    with pytest.warns(UserWarning, match="Some observations have pi=0"):
        estimator.estimate(y, pi)


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
