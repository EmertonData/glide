from unittest.mock import patch

import numpy as np
import pytest

import glide.engines.asi as asi_engine_module
from glide.engines.asi import ASIMeanEngine


@pytest.fixture
def y_true():
    return np.array([3.0, 5.0, np.nan, np.nan, np.nan])


@pytest.fixture
def y_proxy():
    return np.array([2.0, 4.0, 5.0, 7.0, 9.0])


@pytest.fixture
def sampling_probability():
    return np.array([0.5, 0.5, 0.5, 0.5, 0.0])


@pytest.fixture
def engine():
    return ASIMeanEngine()


@pytest.fixture
def dataset(engine, y_true, y_proxy, sampling_probability):
    return engine._preprocess(y_true, y_proxy, sampling_probability)


# --- _preprocess ---


def test_preprocess_delegates_to_validation(engine, y_true, y_proxy, sampling_probability):
    non_zero_probability_mask = np.array([True, True, True, True, False])
    with (
        patch.object(asi_engine_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(asi_engine_module, "_validate_y_proxy") as mock_validate_y_proxy,
        patch.object(asi_engine_module, "_validate_probabilities") as mock_validate_probabilities,
        patch.object(asi_engine_module, "_validate_label_prob_consistency") as mock_validate_label_prob_consistency,
        patch.object(asi_engine_module, "_validate_min_samples") as mock_validate_min_samples,
        patch.object(
            asi_engine_module, "_get_non_zero_mask", return_value=non_zero_probability_mask
        ) as mock_get_non_zero_mask,
    ):
        engine._preprocess(y_true, y_proxy, sampling_probability)

        mock_validate_equal_lengths.assert_called_once()
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], y_true)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], y_proxy)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][2], sampling_probability)
        assert mock_validate_equal_lengths.call_args.kwargs == {"names": ["y_true", "y_proxy", "sampling_probability"]}

        mock_validate_y_proxy.assert_called_once()
        np.testing.assert_array_equal(mock_validate_y_proxy.call_args[0][0], y_proxy)

        mock_validate_probabilities.assert_called_once()
        np.testing.assert_array_equal(mock_validate_probabilities.call_args[0][0], sampling_probability)

        mock_validate_label_prob_consistency.assert_called_once()
        np.testing.assert_array_equal(
            mock_validate_label_prob_consistency.call_args[0][0], np.array([True, True, False, False, False])
        )
        np.testing.assert_array_equal(mock_validate_label_prob_consistency.call_args[0][1], sampling_probability)

        mock_validate_min_samples.assert_called_once()
        np.testing.assert_array_equal(mock_validate_min_samples.call_args[0][0], y_true)
        assert mock_validate_min_samples.call_args[0][1] == "y_true"

        mock_get_non_zero_mask.assert_called_once()
        np.testing.assert_array_equal(mock_get_non_zero_mask.call_args[0][0], sampling_probability)


def test_preprocess_valid_output(engine, y_true, y_proxy, sampling_probability):
    y_true_filtered, y_proxy_filtered, sampling_probability_filtered = engine._preprocess(
        y_true, y_proxy, sampling_probability
    )
    np.testing.assert_array_equal(y_true_filtered, np.array([3.0, 5.0, np.nan, np.nan]))
    np.testing.assert_array_equal(y_proxy_filtered, np.array([2.0, 4.0, 5.0, 7.0]))
    np.testing.assert_array_equal(sampling_probability_filtered, np.array([0.5, 0.5, 0.5, 0.5]))


# --- _fit_tuning_parameter ---


def test_fit_tuning_parameter_delegates_to_validation(engine, dataset):
    with patch.object(asi_engine_module, "_validate_non_constant") as mock_validate_non_constant:
        engine._fit_tuning_parameter(dataset, power_tuning=True)

        mock_validate_non_constant.assert_called_once()
        np.testing.assert_allclose(mock_validate_non_constant.call_args[0][0], np.array([2.0, 4.0, -5.0, -7.0]))
        assert mock_validate_non_constant.call_args[0][1] == "'y_proxy' values lead to constant rectifiers."


def test_fit_tuning_parameter_power_tuning_true(engine, dataset):
    tuning_parameter = engine._fit_tuning_parameter(dataset, power_tuning=True)
    assert tuning_parameter == pytest.approx(0.894, abs=1e-3)


def test_fit_tuning_parameter_power_tuning_false(engine, dataset):
    tuning_parameter = engine._fit_tuning_parameter(dataset, power_tuning=False)
    assert tuning_parameter == 1.0


# --- _compute_mean_and_std ---


def test_compute_mean_and_std(engine, dataset):
    mean, std = engine._compute_mean_and_std(dataset, tuning_parameter=0.5)
    assert mean == pytest.approx(4.75, abs=1e-4)
    assert std == pytest.approx(1.1989, abs=1e-4)
