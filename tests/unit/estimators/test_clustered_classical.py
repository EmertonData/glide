from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from glide.engines.clustered_classical import ClusteredClassicalMeanEngine
from glide.estimators import ClusteredClassicalMeanEstimator
from glide.mean_inference_results import ClassicalMeanInferenceResult


@pytest.fixture
def y() -> NDArray:
    return np.array([5.0, 5.0, 7.0, 7.0])


@pytest.fixture
def clusters() -> NDArray:
    return np.array(["A", "A", "B", "B"])


@pytest.fixture
def estimator() -> ClusteredClassicalMeanEstimator:
    return ClusteredClassicalMeanEstimator()


# --- __init__ ---


def test_init_sets_engine(estimator):
    assert isinstance(estimator._engine, ClusteredClassicalMeanEngine)


# --- estimate ---


def test_estimate_delegates(estimator, y, clusters):
    with (
        patch.object(estimator._engine, "preprocess", wraps=estimator._engine.preprocess) as mock_preprocess,
        patch.object(
            estimator._engine, "compute_mean_and_std", wraps=estimator._engine.compute_mean_and_std
        ) as mock_compute_mean_and_std,
    ):
        estimator.estimate(y, clusters)

        mock_preprocess.assert_called_once()
        np.testing.assert_array_equal(mock_preprocess.call_args[0][0], y)
        np.testing.assert_array_equal(mock_preprocess.call_args[0][1], clusters)
        mock_compute_mean_and_std.assert_called_once()
        np.testing.assert_array_equal(mock_compute_mean_and_std.call_args[0][0], np.array([5.0, 7.0]))
        assert mock_compute_mean_and_std.call_args[0][1] is None


def test_estimate_is_valid_inference_result(estimator, y, clusters):
    result = estimator.estimate(y, clusters)
    assert isinstance(result, ClassicalMeanInferenceResult)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "ClusteredClassicalMeanEstimator"


def test_estimate_metadata(estimator, y, clusters):
    result = estimator.estimate(y, clusters, metric_name="performance")
    assert result.metric_name == "performance"
    assert result.estimator_name == estimator.__class__.__name__
    assert result.n == 4


def test_estimate_custom_confidence_level(estimator, y, clusters):
    result = estimator.estimate(y, clusters, confidence_level=0.85)

    expected_mean = 6.0
    expected_std = 1.0
    expected_lower = 4.560
    expected_upper = 7.440

    assert result.confidence_interval.mean == pytest.approx(expected_mean)
    assert result.std == pytest.approx(expected_std)
    assert result.confidence_interval.confidence_level == 0.85
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=0.001)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=0.001)


# --- __str__ / __repr__ ---


def test_str_format(estimator, y, clusters):
    result = estimator.estimate(y, clusters, metric_name="performance")
    output = str(result)
    expected = (
        "Metric: performance\n"
        "Point Estimate: 6.000\n"
        "Confidence Interval (95%): [4.040, 7.960]\n"
        "Estimator : ClusteredClassicalMeanEstimator\n"
        "n: 4"
    )
    assert output == expected


def test_repr_equals_str(estimator, y, clusters):
    result = estimator.estimate(y, clusters, metric_name="perf")
    assert repr(result) == str(result)
