from typing import Tuple
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from glide.confidence_intervals import CLTConfidenceInterval
from glide.estimators import ASIMeanEstimator
from glide.mean_inference_results import PredictionPoweredMeanInferenceResult

# --- helpers ---


@pytest.fixture
def y_arrays() -> Tuple[NDArray, NDArray, NDArray]:
    y_true = np.array([3.0, 5.0, np.nan, np.nan])
    y_proxy = np.array([2.0, 4.0, 5.0, 7.0])
    sampling_probability = np.array([0.5, 0.5, 0.5, 0.5])
    return y_true, y_proxy, sampling_probability


@pytest.fixture
def estimator() -> ASIMeanEstimator:
    return ASIMeanEstimator()


# --- estimate ---


def test_estimate_delegates(estimator, y_arrays):
    y_true, y_proxy, sampling_probability = y_arrays
    with (
        patch.object(estimator._engine, "_preprocess", wraps=estimator._engine._preprocess) as mock_preprocess,
        patch.object(
            estimator._engine, "_fit_tuning_parameter", wraps=estimator._engine._fit_tuning_parameter
        ) as mock_fit_tuning_parameter,
        patch.object(
            estimator._engine, "_compute_mean_and_std", wraps=estimator._engine._compute_mean_and_std
        ) as mock_compute_mean_and_std,
        patch.object(
            estimator._classical_engine,
            "_compute_mean_and_std",
            wraps=estimator._classical_engine._compute_mean_and_std,
        ) as mock_classical_engine_compute_mean_and_std,
    ):
        estimator.estimate(y_true, y_proxy, sampling_probability)

        mock_preprocess.assert_called_once()
        np.testing.assert_array_equal(mock_preprocess.call_args[0][0], y_true)
        np.testing.assert_array_equal(mock_preprocess.call_args[0][1], y_proxy)
        np.testing.assert_array_equal(mock_preprocess.call_args[0][2], sampling_probability)

        mock_fit_tuning_parameter.assert_called_once()
        asi_dataset = mock_fit_tuning_parameter.call_args[0][0]
        np.testing.assert_array_equal(asi_dataset[0], np.array([3.0, 5.0, np.nan, np.nan]))
        assert mock_fit_tuning_parameter.call_args.kwargs == {"power_tuning": True}

        mock_compute_mean_and_std.assert_called_once()
        assert mock_compute_mean_and_std.call_args[0][0] is asi_dataset
        assert mock_compute_mean_and_std.call_args[0][1] == pytest.approx(0.894, abs=1e-3)

        mock_classical_engine_compute_mean_and_std.assert_called_once()
        ipw_classical_dataset = mock_classical_engine_compute_mean_and_std.call_args[0][0]
        np.testing.assert_array_equal(ipw_classical_dataset[0], np.array([3.0, 5.0, np.nan, np.nan]))
        np.testing.assert_array_equal(ipw_classical_dataset[1], sampling_probability)
        assert mock_classical_engine_compute_mean_and_std.call_args[0][1] is None


def test_estimate_is_valid_inference_result(estimator, y_arrays):
    y_true, y_proxy, sampling_probability = y_arrays
    result = estimator.estimate(y_true, y_proxy, sampling_probability)
    assert isinstance(result, PredictionPoweredMeanInferenceResult)
    assert isinstance(result.confidence_interval, CLTConfidenceInterval)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "ASIMeanEstimator"


def test_estimate_metadata(estimator, y_arrays):
    y_true, y_proxy, sampling_probability = y_arrays
    result = estimator.estimate(y_true, y_proxy, sampling_probability, metric_name="TestMetric")
    assert result.metric_name == "TestMetric"
    assert result.estimator_name == estimator.__class__.__name__
    assert result.n_true == 2
    assert result.n_proxy == 4
    assert result.effective_sample_size == 35


def test_estimate_custom_confidence_level(estimator, y_arrays):
    y_true, y_proxy, sampling_probability = y_arrays
    result = estimator.estimate(y_true, y_proxy, sampling_probability, confidence_level=0.95)

    expected_mean = 5.34
    expected_std = 0.58
    expected_lower = 4.20
    expected_upper = 6.48

    assert result.confidence_interval.confidence_level == 0.95
    assert result.confidence_interval.mean == pytest.approx(expected_mean, abs=0.01)
    assert result.std == pytest.approx(expected_std, abs=0.01)
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=0.01)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=0.01)


# --- __str__ / __repr__ ---


def test_str_format(estimator, y_arrays):
    y_true, y_proxy, sampling_probability = y_arrays
    result = estimator.estimate(y_true, y_proxy, sampling_probability, metric_name="accuracy")
    output = str(result)
    expected = (
        "Metric: accuracy\n"
        "Point Estimate: 5.341\n"
        "Confidence Interval (95%): [4.203, 6.479]\n"
        "Estimator : ASIMeanEstimator\n"
        "n_true: 2\n"
        "n_proxy: 4\n"
        "Effective Sample Size: 35"
    )
    assert output == expected


def test_repr_equals_str(estimator, y_arrays):
    y_true, y_proxy, sampling_probability = y_arrays
    result = estimator.estimate(y_true, y_proxy, sampling_probability, metric_name="perf")
    assert repr(result) == str(result)
