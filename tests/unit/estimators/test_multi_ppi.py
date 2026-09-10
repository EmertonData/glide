from typing import Tuple
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

import glide.estimators.multi_ppi as multi_ppi_module
from glide.confidence_intervals import CLTConfidenceInterval
from glide.engines.classical import ClassicalMeanEngine
from glide.engines.multi_ppi import MultiPPIMeanEngine
from glide.estimators import MultiPPIMeanEstimator
from glide.mean_inference_results import PredictionPoweredMeanInferenceResult

# ── helpers ────────────────────────────────────────────────────────────────────


@pytest.fixture
def y_arrays() -> Tuple[NDArray, NDArray]:
    y_true = np.array([1.0, 2.0, np.nan, np.nan])
    y_proxies = np.array([[1.0, 0.0], [2.0, 2.0], [3.0, 1.0], [4.0, 3.0]])
    return y_true, y_proxies


@pytest.fixture
def estimator() -> MultiPPIMeanEstimator:
    return MultiPPIMeanEstimator()


# --- __init__ ---


def test_init_sets_engines(estimator):
    assert isinstance(estimator._engine, MultiPPIMeanEngine)
    assert isinstance(estimator._classical_engine, ClassicalMeanEngine)


# --- estimate ---


def test_estimate_delegates(estimator, y_arrays):
    y_true, y_proxies = y_arrays
    with (
        patch.object(multi_ppi_module, "_validate_non_constant") as mock_validate_non_constant,
        patch.object(estimator._engine, "preprocess", wraps=estimator._engine.preprocess) as mock_preprocess,
        patch.object(
            estimator._engine, "fit_tuning_parameter", wraps=estimator._engine.fit_tuning_parameter
        ) as mock_fit_tuning_parameter,
        patch.object(
            estimator._engine, "compute_mean_and_std", wraps=estimator._engine.compute_mean_and_std
        ) as mock_compute_mean_and_std,
        patch.object(
            estimator._classical_engine,
            "compute_mean_and_std",
            wraps=estimator._classical_engine.compute_mean_and_std,
        ) as mock_classical_engine_compute_mean_and_std,
    ):
        estimator.estimate(y_true, y_proxies)

        mock_preprocess.assert_called_once()
        np.testing.assert_array_equal(mock_preprocess.call_args[0][0], y_true)
        np.testing.assert_array_equal(mock_preprocess.call_args[0][1], y_proxies)

        mock_validate_non_constant.assert_called_once()
        np.testing.assert_array_equal(mock_validate_non_constant.call_args[0][0], np.array([1.0, 2.0]))
        assert mock_validate_non_constant.call_args[0][1] == "'y_true' labeled values are constant."

        mock_fit_tuning_parameter.assert_called_once()
        multi_ppi_dataset = mock_fit_tuning_parameter.call_args[0][0]
        np.testing.assert_array_equal(multi_ppi_dataset[0], np.array([1.0, 2.0]))
        assert mock_fit_tuning_parameter.call_args[0][1] is True

        mock_compute_mean_and_std.assert_called_once()
        np.testing.assert_array_equal(mock_compute_mean_and_std.call_args[0][0][0], np.array([1.0, 2.0]))

        mock_classical_engine_compute_mean_and_std.assert_called_once()
        np.testing.assert_array_equal(mock_classical_engine_compute_mean_and_std.call_args[0][0], np.array([1.0, 2.0]))
        assert mock_classical_engine_compute_mean_and_std.call_args[0][1] is None


def test_estimate_returns_valid_inference_result(estimator, y_arrays):
    y_true, y_proxies = y_arrays
    result = estimator.estimate(y_true, y_proxies)
    assert isinstance(result, PredictionPoweredMeanInferenceResult)
    assert isinstance(result.confidence_interval, CLTConfidenceInterval)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "MultiPPIMeanEstimator"


def test_estimate_metadata(estimator, y_arrays):
    y_true, y_proxies = y_arrays
    result = estimator.estimate(y_true, y_proxies, metric_name="accuracy")
    assert result.metric_name == "accuracy"
    assert result.estimator_name == "MultiPPIMeanEstimator"
    assert result.n_true == 2
    assert result.n_proxy == 4
    assert result.effective_sample_size == 3


def test_estimate_custom_confidence_level(estimator, y_arrays):
    y_true, y_proxies = y_arrays
    result = estimator.estimate(y_true, y_proxies, confidence_level=0.85)

    expected_mean = 1.5
    expected_std = 0.395
    expected_lower = 0.931
    expected_upper = 2.069

    assert result.confidence_interval.confidence_level == 0.85
    assert result.confidence_interval.mean == pytest.approx(expected_mean, abs=0.01)
    assert result.std == pytest.approx(expected_std, abs=0.01)
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=0.01)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=0.01)


# --- __str__ / __repr__ ---


def test_str_format(estimator, y_arrays):
    y_true, y_proxies = y_arrays
    result = estimator.estimate(y_true, y_proxies, metric_name="performance")

    output = str(result)
    expected = (
        "Metric: performance\n"
        "Point Estimate: 1.500\n"
        "Confidence Interval (95%): [0.725, 2.275]\n"
        "Estimator : MultiPPIMeanEstimator\n"
        "n_true: 2\n"
        "n_proxy: 4\n"
        "Effective Sample Size: 3"
    )
    assert output == expected


def test_repr_equals_str(estimator, y_arrays):
    y_true, y_proxies = y_arrays
    result = estimator.estimate(y_true, y_proxies, metric_name="perf")
    assert repr(result) == str(result)
