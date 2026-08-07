from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from glide.engines.classical import ClassicalMeanEngine
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


# --- __init__ ---


def test_init_sets_engine(estimator):
    assert isinstance(estimator._engine, ClassicalMeanEngine)


# --- estimate ---


def test_estimate_delegates(estimator, y_array):
    with (
        patch.object(estimator._engine, "preprocess", wraps=estimator._engine.preprocess) as mock_preprocess,
        patch.object(
            estimator._engine, "compute_mean_and_std", wraps=estimator._engine.compute_mean_and_std
        ) as mock_compute_mean_and_std,
    ):
        estimator.estimate(y_array, confidence_level=0.90)

        mock_preprocess.assert_called_once()
        np.testing.assert_array_equal(mock_preprocess.call_args[0][0], y_array)
        mock_compute_mean_and_std.assert_called_once()
        np.testing.assert_array_equal(mock_compute_mean_and_std.call_args[0][0], y_array)
        assert mock_compute_mean_and_std.call_args[0][1] is None


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
