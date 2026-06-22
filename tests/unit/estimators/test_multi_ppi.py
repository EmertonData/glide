from typing import Tuple
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

import glide.estimators.multi_ppi as multi_ppi_module
from glide.confidence_intervals import CLTConfidenceInterval
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


# --- _preprocess ---


def test_preprocess_delegates_to_validation(estimator, y_arrays):
    y_true, y_proxies = y_arrays
    with (
        patch.object(multi_ppi_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(multi_ppi_module, "_validate_y_proxies") as mock_validate_y_proxies,
        patch.object(multi_ppi_module, "_validate_y_true") as mock_validate_y_true,
        patch.object(multi_ppi_module, "_validate_sample_sizes") as mock_validate_sample_sizes,
    ):
        estimator._preprocess(y_true, y_proxies)

        mock_validate_equal_lengths.assert_called_once_with(y_true, y_proxies, names=["y_true", "y_proxies"])
        mock_validate_y_proxies.assert_called_once_with(y_proxies)
        mock_validate_y_true.assert_called_once_with(y_true)
        mock_validate_sample_sizes.assert_called_once()
        labeled_mask_arg = mock_validate_sample_sizes.call_args[0][0]
        np.testing.assert_array_equal(labeled_mask_arg, np.array([True, True, False, False]))


def test_preprocess_valid_output(estimator, y_arrays):
    y_true_all, y_proxies_all = y_arrays
    y_true, y_proxies_labeled, y_proxies_unlabeled = estimator._preprocess(y_true_all, y_proxies_all)
    np.testing.assert_array_equal(y_true, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(y_proxies_labeled, np.array([[1.0, 0.0], [2.0, 2.0]]))
    np.testing.assert_array_equal(y_proxies_unlabeled, np.array([[3.0, 1.0], [4.0, 3.0]]))


# --- estimate ---


def test_estimate_is_valid_inference_result(estimator, y_arrays):
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


def test_estimate_known_values(estimator, y_arrays):
    y_true, y_proxies = y_arrays
    result = estimator.estimate(y_true, y_proxies, confidence_level=0.95)
    assert result.confidence_interval.mean == pytest.approx(1.5, abs=1e-10)
    assert result.std == pytest.approx(np.sqrt(0.15625), abs=1e-10)
    assert result.confidence_interval.lower_bound == pytest.approx(0.725, abs=0.01)
    assert result.confidence_interval.upper_bound == pytest.approx(2.275, abs=0.01)


def test_estimate_power_tuning_false(estimator, y_arrays):
    y_true, y_proxies = y_arrays
    result_tuned = estimator.estimate(y_true, y_proxies, power_tuning=True)
    result_untuned = estimator.estimate(y_true, y_proxies, power_tuning=False)
    assert isinstance(result_untuned, PredictionPoweredMeanInferenceResult)
    assert result_untuned.estimator_name == "MultiPPIMeanEstimator"
    assert result_untuned.confidence_interval.mean != pytest.approx(result_tuned.confidence_interval.mean, abs=1e-10)
