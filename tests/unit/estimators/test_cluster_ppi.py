from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

import glide.estimators.cluster_ppi as cluster_ppi_module
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


# --- estimate ---


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


def test_estimate_known_values_power_tuning_false(estimator, y_true, y_proxy, clusters):
    result = estimator.estimate(y_true, y_proxy, clusters, power_tuning=False)
    assert result.confidence_interval.mean == pytest.approx(6.0, abs=1e-10)
    assert result.std == pytest.approx(np.sqrt(2), abs=1e-10)


def test_estimate_known_values_power_tuning_true(estimator, y_true, y_proxy, clusters):
    result = estimator.estimate(y_true, y_proxy, clusters, power_tuning=True)
    assert result.confidence_interval.mean == pytest.approx(61 / 11, abs=1e-10)
    assert result.std == pytest.approx(np.sqrt(37) / 11, abs=1e-10)


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


def test_estimate_delegates_to_validation(estimator, y_true, y_proxy, clusters):
    with patch.object(cluster_ppi_module, "_preprocess") as mock_preprocess:
        mock_preprocess.return_value = (
            np.array([4.0, 6.0]),
            np.array([2.0, 6.0]),
            np.array([4.0, 6.0]),
            np.array([1, 1]),
            np.array([1, 1]),
        )
        estimator.estimate(y_true, y_proxy, clusters)

        mock_preprocess.assert_called_once()
        np.testing.assert_array_equal(mock_preprocess.call_args[0][0], y_true)
        np.testing.assert_array_equal(mock_preprocess.call_args[0][1], y_proxy)
        np.testing.assert_array_equal(mock_preprocess.call_args[0][2], clusters)


def test_estimate_raises_on_partial_cluster(estimator):
    y_t = np.array([5.0, np.nan, 6.0, np.nan])
    y_p = np.array([4.9, 5.1, 6.0, 6.0])
    cls = np.array(["A", "A", "B", "C"])

    with pytest.raises(ValueError, match="Cluster 'A'"):
        estimator.estimate(y_t, y_p, cls)


def test_estimate_raises_on_zero_variance_proxy(estimator):
    y_t = np.array([4.0, np.nan, 6.0, np.nan])
    y_p = np.array([5.0, 5.0, 5.0, 5.0])
    cls = np.array(["A", "C", "B", "D"])

    with pytest.raises(ValueError, match="zero variance"):
        estimator.estimate(y_t, y_p, cls)


def test_estimate_raises_on_zero_variance_proxy_power_tuning_false_succeeds(estimator):
    y_t = np.array([4.0, np.nan, 6.0, np.nan])
    y_p = np.array([5.0, 5.0, 5.0, 5.0])
    cls = np.array(["A", "C", "B", "D"])

    result = estimator.estimate(y_t, y_p, cls, power_tuning=False)
    assert isinstance(result, PredictionPoweredMeanInferenceResult)


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
