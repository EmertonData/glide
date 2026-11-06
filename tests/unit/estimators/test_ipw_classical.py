from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from glide.engines.ipw_classical import IPWClassicalMeanEngine
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


# --- __init__ ---


def test_init_sets_engine(estimator):
    assert isinstance(estimator._engine, IPWClassicalMeanEngine)


# --- estimate ---


def test_estimate_delegates(estimator, y, sampling_probability):
    dataset = estimator._engine._preprocess(y, sampling_probability)
    with (
        patch.object(estimator._engine, "_preprocess", wraps=estimator._engine._preprocess) as mock_preprocess,
        patch.object(
            estimator._engine, "_fit_tuning_parameter", wraps=estimator._engine._fit_tuning_parameter
        ) as mock_fit_tuning_parameter,
        patch.object(
            estimator._engine, "_compute_mean_and_std", wraps=estimator._engine._compute_mean_and_std
        ) as mock_compute_mean_and_std,
    ):
        estimator.estimate(y, sampling_probability)

    mock_preprocess.assert_called_once()
    np.testing.assert_array_equal(mock_preprocess.call_args[0][0], y)
    np.testing.assert_array_equal(mock_preprocess.call_args[0][1], sampling_probability)
    mock_fit_tuning_parameter.assert_called_once()
    for received, expected in zip(mock_fit_tuning_parameter.call_args[0][0], dataset, strict=True):
        np.testing.assert_array_equal(received, expected)
    assert mock_fit_tuning_parameter.call_args[1] == {"power_tuning": False}
    mock_compute_mean_and_std.assert_called_once()
    for received, expected in zip(mock_compute_mean_and_std.call_args[0][0], dataset, strict=True):
        np.testing.assert_array_equal(received, expected)
    assert mock_compute_mean_and_std.call_args[0][1] is None


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
