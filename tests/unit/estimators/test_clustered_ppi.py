from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

import glide.estimators.clustered_ppi as clustered_ppi_module
from glide.confidence_intervals import CLTConfidenceInterval
from glide.engines.clustered_ppi import ClusteredPPIMeanEngine
from glide.estimators import ClusteredPPIMeanEstimator
from glide.mean_inference_results import PredictionPoweredMeanInferenceResult


@pytest.fixture
def y_true() -> NDArray:
    return np.array([4.0, np.nan, 6.0, np.nan])


@pytest.fixture
def y_proxy() -> NDArray:
    return np.array([2.0, 4.0, 6.0, 6.0])


@pytest.fixture
def clusters() -> NDArray:
    return np.array(["A", "B", "C", "D"])


@pytest.fixture
def estimator() -> ClusteredPPIMeanEstimator:
    return ClusteredPPIMeanEstimator()


# --- __init__ ---


def test_init_sets_engine(estimator):
    assert isinstance(estimator._engine, ClusteredPPIMeanEngine)


# --- estimate ---


def test_estimate_delegates(estimator, y_true, y_proxy, clusters):
    expected_labeled_true_means = np.array([4.0, 6.0])
    with (
        patch.object(clustered_ppi_module, "_validate_y_true") as mock_validate_y_true,
        patch.object(estimator._engine, "preprocess", wraps=estimator._engine.preprocess) as mock_preprocess,
        patch.object(
            estimator._engine, "fit_tuning_parameter", wraps=estimator._engine.fit_tuning_parameter
        ) as mock_fit_tuning_parameter,
        patch.object(
            estimator._engine, "compute_mean_and_std", wraps=estimator._engine.compute_mean_and_std
        ) as mock_compute_mean_and_std,
    ):
        estimator.estimate(y_true, y_proxy, clusters)

        mock_validate_y_true.assert_called_once()
        np.testing.assert_array_equal(mock_validate_y_true.call_args[0][0], y_true)

        mock_preprocess.assert_called_once()
        np.testing.assert_array_equal(mock_preprocess.call_args[0][0], y_true)
        np.testing.assert_array_equal(mock_preprocess.call_args[0][1], y_proxy)
        np.testing.assert_array_equal(mock_preprocess.call_args[0][2], clusters)

        mock_fit_tuning_parameter.assert_called_once()
        fit_labeled_true_means, _, _ = mock_fit_tuning_parameter.call_args[0][0]
        np.testing.assert_array_equal(fit_labeled_true_means, expected_labeled_true_means)
        assert mock_fit_tuning_parameter.call_args.kwargs["power_tuning"] is True

        mock_compute_mean_and_std.assert_called_once()
        compute_labeled_true_means, _, _ = mock_compute_mean_and_std.call_args[0][0]
        np.testing.assert_array_equal(compute_labeled_true_means, expected_labeled_true_means)


def test_estimate_returns_valid_inference_result(estimator, y_true, y_proxy, clusters):
    result = estimator.estimate(y_true, y_proxy, clusters)
    assert isinstance(result, PredictionPoweredMeanInferenceResult)
    assert isinstance(result.confidence_interval, CLTConfidenceInterval)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "ClusteredPPIMeanEstimator"


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
        "Estimator : ClusteredPPIMeanEstimator\n"
        "n_true: 2\n"
        "n_proxy: 4\n"
        "Effective Sample Size: 6"
    )
    assert output == expected


def test_repr_equals_str(estimator, y_true, y_proxy, clusters):
    result = estimator.estimate(y_true, y_proxy, clusters, metric_name="perf")
    assert repr(result) == str(result)
