from unittest.mock import patch

import numpy as np
import pytest

import glide.engines.ipw_classical as ipw_classical_engine_module
from glide.engines.ipw_classical import IPWClassicalMeanEngine


@pytest.fixture
def y():
    return np.array([2.0, np.nan, np.nan, 4.0])


@pytest.fixture
def sampling_probability():
    return np.array([0.5, 0.25, 0.0, 0.8])


@pytest.fixture
def engine():
    return IPWClassicalMeanEngine()


@pytest.fixture
def dataset(engine, y, sampling_probability):
    return engine._preprocess(y, sampling_probability)


# --- _preprocess ---


def test_preprocess_delegates_to_validation(engine, y, sampling_probability):
    with (
        patch.object(ipw_classical_engine_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(ipw_classical_engine_module, "_validate_probabilities") as mock_validate_probabilities,
        patch.object(
            ipw_classical_engine_module, "_validate_label_prob_consistency"
        ) as mock_validate_label_prob_consistency,
        patch.object(ipw_classical_engine_module, "_validate_min_samples") as mock_validate_min_samples,
        patch.object(
            ipw_classical_engine_module, "_get_non_zero_mask", return_value=np.ones(4, dtype=bool)
        ) as mock_get_non_zero_mask,
    ):
        engine._preprocess(y, sampling_probability)

        mock_validate_equal_lengths.assert_called_once()
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], y)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], sampling_probability)
        assert mock_validate_equal_lengths.call_args.kwargs == {"names": ["y", "sampling_probability"]}

        mock_validate_probabilities.assert_called_once()
        np.testing.assert_array_equal(mock_validate_probabilities.call_args[0][0], sampling_probability)

        mock_validate_label_prob_consistency.assert_called_once()
        np.testing.assert_array_equal(
            mock_validate_label_prob_consistency.call_args[0][0], np.array([True, False, False, True])
        )
        np.testing.assert_array_equal(mock_validate_label_prob_consistency.call_args[0][1], sampling_probability)

        mock_validate_min_samples.assert_called_once()
        np.testing.assert_array_equal(mock_validate_min_samples.call_args[0][0], y)
        assert mock_validate_min_samples.call_args[0][1] == "y"

        mock_get_non_zero_mask.assert_called_once()
        np.testing.assert_array_equal(mock_get_non_zero_mask.call_args[0][0], sampling_probability)


def test_preprocess_valid_output(engine, y, sampling_probability):
    y_filtered, sampling_probability_filtered = engine._preprocess(y, sampling_probability)
    np.testing.assert_array_equal(y_filtered, np.array([2.0, np.nan, 4.0]))
    np.testing.assert_array_equal(sampling_probability_filtered, np.array([0.5, 0.25, 0.8]))


# --- _fit_tuning_parameter ---


def test_fit_tuning_parameter_returns_none(engine, dataset):
    assert engine._fit_tuning_parameter(dataset, power_tuning=True) is None
    assert engine._fit_tuning_parameter(dataset, power_tuning=False) is None


# --- _compute_mean_and_std ---


def test_compute_mean_and_std(engine, dataset):
    mean, std = engine._compute_mean_and_std(dataset, tuning_parameter=None)
    assert mean == pytest.approx(3.0)
    assert std == pytest.approx(1.5275, abs=1e-4)
