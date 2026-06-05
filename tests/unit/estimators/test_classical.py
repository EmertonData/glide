from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from glide.estimators import ClassicalMeanEstimator
from glide.mean_inference_results import ClassicalMeanInferenceResult


@pytest.fixture
def y_array(n: int = 4, seed: int = 42) -> NDArray:
    rng = np.random.default_rng(seed)
    y = rng.normal(loc=5.0, scale=1.0, size=n)
    return y


@pytest.fixture
def estimator() -> ClassicalMeanEstimator:
    return ClassicalMeanEstimator()


# --- _preprocess ---


def test_preprocess_removes_nan(estimator):
    y = np.array([2.0, np.nan, 4.0])
    result = estimator._preprocess(y)
    np.testing.assert_array_equal(result, np.array([2.0, 4.0]))


def test_preprocess_delegate_to_validation(estimator):
    y_valid = np.array([1.0])
    with patch("glide.estimators.classical._validate_min_samples") as mock_validate_min_samples:
        estimator._preprocess(y_valid)
    mock_validate_min_samples.assert_called_once()
    np.testing.assert_array_equal(mock_validate_min_samples.call_args[0][0], y_valid)
    assert mock_validate_min_samples.call_args[0][1] == "y"


# --- estimate ---


def test_estimate_is_valid_inference_result(estimator, y_array):
    result = estimator.estimate(y_array)
    assert isinstance(result, ClassicalMeanInferenceResult)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "ClassicalMeanEstimator"


def test_estimate_metadata(estimator, y_array):
    result = estimator.estimate(y_array, metric_name="performance")
    assert result.metric_name == "performance"
    assert result.estimator_name == estimator.__class__.__name__
    assert result.n == 4


def test_estimate_custom_confidence_level(estimator, y_array):
    result = estimator.estimate(y_array, confidence_level=0.90)
    assert result.confidence_interval.confidence_level == 0.90

    expected_mean = 5.24
    expected_std = 0.45
    expected_lower = 4.50
    expected_upper = 5.97

    assert result.confidence_interval.confidence_level == 0.90
    assert result.confidence_interval.mean == pytest.approx(expected_mean, abs=0.01)
    assert result.std == pytest.approx(expected_std, abs=0.01)
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=0.01)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=0.01)


# --- __str__ / __repr__ ---


def test_str_format(estimator, y_array):
    result = estimator.estimate(y_array, metric_name="performance")
    output = str(result)
    expected = (
        "Metric: performance\n"
        "Point Estimate: 5.239\n"
        "Confidence Interval (95%): [4.364, 6.114]\n"
        "Estimator : ClassicalMeanEstimator\n"
        "n: 4"
    )
    assert output == expected


def test_repr_equals_str(estimator, y_array):
    result = estimator.estimate(y_array, metric_name="perf")
    assert repr(result) == str(result)
