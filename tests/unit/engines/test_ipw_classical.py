from unittest.mock import patch

import numpy as np
import pytest

from glide.engines.ipw_classical import IPWClassicalMeanEngine


@pytest.fixture
def y():
    return np.array([2.0, np.nan, 4.0])


@pytest.fixture
def sampling_probability():
    return np.array([0.5, 0.5, 0.5])


@pytest.fixture
def engine():
    return IPWClassicalMeanEngine()


@pytest.fixture
def dataset(engine, y, sampling_probability):
    return engine._preprocess(y, sampling_probability)


# --- _preprocess ---


def test_preprocess_valid_output(engine, y, sampling_probability):
    y_filled, labeled_indicator, sampling_probability_filtered = engine._preprocess(y, sampling_probability)
    np.testing.assert_array_equal(y_filled, np.array([2.0, 0.0, 4.0]))
    np.testing.assert_array_equal(labeled_indicator, np.array([1.0, 0.0, 1.0]))
    np.testing.assert_array_equal(sampling_probability_filtered, np.array([0.5, 0.5, 0.5]))


def test_preprocess_drops_zero_probability_rows(engine):
    y = np.array([1.0, 2.0, np.nan, np.nan])
    sampling_probability = np.array([0.5, 0.5, 0.0, 0.5])
    y_filled, labeled_indicator, sampling_probability_filtered = engine._preprocess(y, sampling_probability)
    np.testing.assert_array_equal(y_filled, np.array([1.0, 2.0, 0.0]))
    np.testing.assert_array_equal(labeled_indicator, np.array([1.0, 1.0, 0.0]))
    np.testing.assert_array_equal(sampling_probability_filtered, np.array([0.5, 0.5, 0.5]))


def test_preprocess_unequal_lengths(engine, y):
    with pytest.raises(ValueError, match="'y' and 'sampling_probability' must have the same length; got 3 and 2."):
        engine._preprocess(y, np.array([0.5, 0.5]))


def test_preprocess_too_few_labeled_values(engine):
    with pytest.raises(ValueError, match="'y' must have at least 2 non-NaN values; got 1."):
        engine._preprocess(np.array([1.0, np.nan]), np.array([0.5, 0.5]))


def test_preprocess_delegates_to_validation(engine, y, sampling_probability):
    with (
        patch("glide.engines.ipw_classical._validate_equal_lengths") as mock_validate_equal_lengths,
        patch("glide.engines.ipw_classical._validate_probabilities") as mock_validate_probabilities,
        patch("glide.engines.ipw_classical._validate_label_prob_consistency") as mock_validate_label_prob_consistency,
        patch("glide.engines.ipw_classical._validate_min_samples") as mock_validate_min_samples,
        patch(
            "glide.engines.ipw_classical._get_non_zero_mask", return_value=np.ones(3, dtype=bool)
        ) as mock_get_non_zero_mask,
    ):
        engine._preprocess(y, sampling_probability)

    mock_validate_equal_lengths.assert_called_once()
    np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], y)
    np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], sampling_probability)
    assert mock_validate_equal_lengths.call_args[1] == {"names": ["y", "sampling_probability"]}
    mock_validate_probabilities.assert_called_once()
    np.testing.assert_array_equal(mock_validate_probabilities.call_args[0][0], sampling_probability)
    mock_validate_label_prob_consistency.assert_called_once()
    np.testing.assert_array_equal(mock_validate_label_prob_consistency.call_args[0][0], np.array([True, False, True]))
    np.testing.assert_array_equal(mock_validate_label_prob_consistency.call_args[0][1], sampling_probability)
    mock_validate_min_samples.assert_called_once()
    np.testing.assert_array_equal(mock_validate_min_samples.call_args[0][0], np.array([2.0, 4.0]))
    assert mock_validate_min_samples.call_args[0][1] == "y"
    mock_get_non_zero_mask.assert_called_once()
    np.testing.assert_array_equal(mock_get_non_zero_mask.call_args[0][0], sampling_probability)


# --- _fit_tuning_parameter ---


def test_fit_tuning_parameter_returns_none(engine, dataset):
    assert engine._fit_tuning_parameter(dataset, power_tuning=True) is None
    assert engine._fit_tuning_parameter(dataset, power_tuning=False) is None


# --- _compute_mean_and_std ---


def test_compute_mean_and_std(engine):
    dataset = (np.array([2.0, 4.0]), np.array([1.0, 1.0]), np.array([0.5, 0.5]))
    mean, std = engine._compute_mean_and_std(dataset, tuning_parameter=None)
    assert mean == pytest.approx(6.0)
    assert std == pytest.approx(2.0)
