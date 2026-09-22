from unittest.mock import patch

import numpy as np
import pytest

import glide.estimators.stratified_ppi as stratified_ppi_module
from glide.confidence_intervals import CLTConfidenceInterval
from glide.engines.stratified_ppi import StratifiedPPIMeanEngine
from glide.estimators import StratifiedPPIMeanEstimator
from glide.mean_inference_results import PredictionPoweredMeanInferenceResult

# ── helpers ────────────────────────────────────────────────────────────────────


@pytest.fixture
def y_true() -> np.ndarray:
    return np.array([5.0, 6.0, np.nan, np.nan, 5.0, 6.0, np.nan, np.nan])


@pytest.fixture
def y_proxy() -> np.ndarray:
    return np.array([4.9, 6.1, 5.2, 6.1, 4.9, 6.1, 5.2, 6.1])


@pytest.fixture
def groups() -> np.ndarray:
    return np.array(["A", "A", "A", "A", "B", "B", "B", "B"])


@pytest.fixture
def estimator() -> StratifiedPPIMeanEstimator:
    return StratifiedPPIMeanEstimator()


# --- __init__ ---


def test_init_sets_engine(estimator):
    assert isinstance(estimator._engine, StratifiedPPIMeanEngine)


# --- estimate ---


def test_estimate_delegates(estimator, y_true, y_proxy, groups):
    with (
        patch.object(stratified_ppi_module, "_validate_non_constant") as mock_validate_non_constant,
        patch.object(estimator._engine, "preprocess", wraps=estimator._engine.preprocess) as mock_preprocess,
        patch.object(
            estimator._engine, "fit_tuning_parameter", wraps=estimator._engine.fit_tuning_parameter
        ) as mock_fit_tuning_parameter,
        patch.object(
            estimator._engine, "compute_mean_and_std", wraps=estimator._engine.compute_mean_and_std
        ) as mock_compute_mean_and_std,
    ):
        estimator.estimate(y_true, y_proxy, groups)

        mock_preprocess.assert_called_once()
        np.testing.assert_array_equal(mock_preprocess.call_args[0][0], y_true)
        np.testing.assert_array_equal(mock_preprocess.call_args[0][1], y_proxy)
        np.testing.assert_array_equal(mock_preprocess.call_args[0][2], groups)

        mock_validate_non_constant.assert_called_once()
        np.testing.assert_array_equal(mock_validate_non_constant.call_args[0][0], np.array([5.0, 6.0, 5.0, 6.0]))
        assert mock_validate_non_constant.call_args[0][1] == "'y_true' labeled values are constant."

        mock_fit_tuning_parameter.assert_called_once()
        stratified_dataset = mock_fit_tuning_parameter.call_args[0][0]
        assert set(stratified_dataset.keys()) == {"A", "B"}
        assert mock_fit_tuning_parameter.call_args[0][1] is True

        mock_compute_mean_and_std.assert_called_once()
        assert set(mock_compute_mean_and_std.call_args[0][0].keys()) == {"A", "B"}
        tuning_parameter = mock_compute_mean_and_std.call_args[0][1]
        assert tuning_parameter["A"] == pytest.approx(0.7843137254901965)
        assert tuning_parameter["B"] == pytest.approx(0.7843137254901965)


def test_estimate_returns_valid_inference_result(estimator, y_true, y_proxy, groups):
    result = estimator.estimate(y_true, y_proxy, groups)
    assert isinstance(result, PredictionPoweredMeanInferenceResult)
    assert isinstance(result.confidence_interval, CLTConfidenceInterval)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "StratifiedPPIMeanEstimator"


def test_estimate_metadata(estimator, y_true, y_proxy, groups):
    result = estimator.estimate(y_true, y_proxy, groups, metric_name="performance")
    assert result.metric_name == "performance"
    assert result.estimator_name == estimator.__class__.__name__
    assert result.n_true == 4
    assert result.n_proxy == 8
    assert result.effective_sample_size == 7


def test_estimate_custom_confidence_level(estimator, y_true, y_proxy, groups):
    result = estimator.estimate(y_true, y_proxy, groups, confidence_level=0.85)

    expected_mean = 5.618
    expected_std = 0.250
    expected_lower = 5.257
    expected_upper = 5.978

    assert result.confidence_interval.confidence_level == 0.85
    assert result.confidence_interval.mean == pytest.approx(expected_mean, abs=0.01)
    assert result.std == pytest.approx(expected_std, abs=0.01)
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=0.01)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=0.01)


# --- __str__ / __repr__ ---


def test_str_format(estimator, y_true, y_proxy, groups):
    result = estimator.estimate(y_true, y_proxy, groups, metric_name="performance")
    output = str(result)
    expected = (
        "Metric: performance\n"
        "Point Estimate: 5.618\n"
        "Confidence Interval (95%): [5.127, 6.108]\n"
        "Estimator : StratifiedPPIMeanEstimator\n"
        "n_true: 4\n"
        "n_proxy: 8\n"
        "Effective Sample Size: 7"
    )
    assert output == expected


def test_repr_equals_str(estimator, y_true, y_proxy, groups):
    result = estimator.estimate(y_true, y_proxy, groups, metric_name="perf")
    assert repr(result) == str(result)
