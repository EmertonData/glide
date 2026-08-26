from typing import Tuple
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

import glide.estimators.ppi as ppi_module
from glide.confidence_intervals import CLTConfidenceInterval
from glide.engines.classical import ClassicalMeanEngine
from glide.engines.ppi import PPIMeanEngine
from glide.estimators import PPIMeanEstimator
from glide.mean_inference_results import PredictionPoweredMeanInferenceResult

# ── helpers ────────────────────────────────────────────────────────────────────


@pytest.fixture
def y_arrays() -> Tuple[NDArray, NDArray]:
    y_true = np.array([1.0, 2.0, np.nan, np.nan])
    y_proxy = np.array([1.0, 2.0, 3.0, 4.0])
    return y_true, y_proxy


@pytest.fixture
def estimator() -> PPIMeanEstimator:
    return PPIMeanEstimator()


# --- __init__ ---


def test_init_sets_engines(estimator):
    assert isinstance(estimator._engine, PPIMeanEngine)
    assert isinstance(estimator._classical_engine, ClassicalMeanEngine)


# --- estimate ---


def test_estimate_delegates(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    with (
        patch.object(ppi_module, "_validate_non_constant") as mock_validate_non_constant,
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
        estimator.estimate(y_true, y_proxy)

        mock_preprocess.assert_called_once()
        np.testing.assert_array_equal(mock_preprocess.call_args[0][0], y_true)
        np.testing.assert_array_equal(mock_preprocess.call_args[0][1], y_proxy)

        mock_validate_non_constant.assert_called_once()
        np.testing.assert_array_equal(mock_validate_non_constant.call_args[0][0], np.array([1.0, 2.0]))
        assert mock_validate_non_constant.call_args[0][1] == "'y_true' labeled values are constant."

        mock_fit_tuning_parameter.assert_called_once()
        ppi_dataset = mock_fit_tuning_parameter.call_args[0][0]
        np.testing.assert_array_equal(ppi_dataset[0], np.array([1.0, 2.0]))
        assert mock_fit_tuning_parameter.call_args[0][1] is True

        mock_compute_mean_and_std.assert_called_once()
        np.testing.assert_array_equal(mock_compute_mean_and_std.call_args[0][0][0], np.array([1.0, 2.0]))
        assert mock_compute_mean_and_std.call_args[0][1] == pytest.approx(0.15)

        mock_classical_engine_compute_mean_and_std.assert_called_once()
        np.testing.assert_array_equal(mock_classical_engine_compute_mean_and_std.call_args[0][0], np.array([1.0, 2.0]))
        assert mock_classical_engine_compute_mean_and_std.call_args[0][1] is None


def test_estimate_is_valid_inference_result(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy)
    assert isinstance(result, PredictionPoweredMeanInferenceResult)
    assert isinstance(result.confidence_interval, CLTConfidenceInterval)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "PPIMeanEstimator"


def test_estimate_metadata(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy, metric_name="performance")
    assert result.metric_name == "performance"
    assert result.estimator_name == estimator.__class__.__name__
    assert result.n_true == 2
    assert result.n_proxy == 4
    assert result.effective_sample_size == 2


def test_estimate_custom_confidence_level(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy, metric_name="perf", confidence_level=0.90)

    expected_mean = 1.8
    expected_std = 0.431
    expected_lower = 1.09
    expected_upper = 2.51

    assert result.confidence_interval.confidence_level == 0.90
    assert result.confidence_interval.mean == pytest.approx(expected_mean, abs=0.01)
    assert result.std == pytest.approx(expected_std, abs=0.01)
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=0.01)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=0.01)


# --- __str__ / __repr__ ---


def test_str_format(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy, metric_name="performance")
    output = str(result)
    expected = (
        "Metric: performance\n"
        "Point Estimate: 1.800\n"
        "Confidence Interval (95%): [0.954, 2.646]\n"
        "Estimator : PPIMeanEstimator\n"
        "n_true: 2\n"
        "n_proxy: 4\n"
        "Effective Sample Size: 2"
    )
    assert output == expected


def test_repr_equals_str(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy, metric_name="perf")
    assert repr(result) == str(result)
