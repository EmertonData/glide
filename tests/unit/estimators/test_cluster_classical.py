from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

import glide.estimators.cluster_classical as cluster_classical_module
from glide.estimators import ClusterClassicalMeanEstimator
from glide.mean_inference_results import ClassicalMeanInferenceResult


@pytest.fixture
def y() -> NDArray:
    return np.array([5.0, 5.0, 7.0, 7.0])


@pytest.fixture
def clusters() -> NDArray:
    return np.array(["A", "A", "B", "B"])


@pytest.fixture
def estimator() -> ClusterClassicalMeanEstimator:
    return ClusterClassicalMeanEstimator()


# --- _preprocess ---


def test_preprocess_delegates_to_validation(estimator, y, clusters):

    with (
        patch.object(cluster_classical_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(cluster_classical_module, "_validate_has_no_nan") as mock_validate_has_no_nan,
        patch.object(cluster_classical_module, "_validate_bounds") as mock_validate_bounds,
    ):
        estimator._preprocess(y, clusters)

        mock_validate_equal_lengths.assert_called_once()
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], y)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], clusters)
        assert mock_validate_equal_lengths.call_args[1] == {"names": ["y", "clusters"]}

        mock_validate_has_no_nan.assert_called_once()
        np.testing.assert_array_equal(mock_validate_has_no_nan.call_args[0][0], clusters)
        assert mock_validate_has_no_nan.call_args[0][1] == "clusters"

        mock_validate_bounds.assert_called_once_with(
            2,
            "n_valid_clusters",
            lower=2,
            error_message="Need at least 2 clusters with non-NaN observations; got 2.",
        )


def test_preprocess_returns_filtered_arrays(estimator):
    y = np.array([2.0, np.nan, 4.0, np.nan, np.nan])
    clusters = np.array(["A", "A", "B", "C", "C"])
    y_valid, cluster_indices, n_valid_clusters = estimator._preprocess(y, clusters)
    assert n_valid_clusters == 2
    np.testing.assert_array_equal(y_valid, np.array([2.0, 4.0]))
    np.testing.assert_array_equal(cluster_indices, np.array([0, 1]))


# --- estimate ---


def test_estimate_is_valid_inference_result(estimator, y, clusters):
    result = estimator.estimate(y, clusters)
    assert isinstance(result, ClassicalMeanInferenceResult)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "ClusterClassicalMeanEstimator"


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
        "Estimator : ClusterClassicalMeanEstimator\n"
        "n: 4"
    )
    assert output == expected


def test_repr_equals_str(estimator, y, clusters):
    result = estimator.estimate(y, clusters, metric_name="perf")
    assert repr(result) == str(result)
