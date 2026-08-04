from unittest.mock import patch

import numpy as np
import pytest

import glide.engines.ppi as ppi_engine_module
from glide.engines.ppi import PPIMeanEngine


@pytest.fixture
def y_true():
    return np.array([1.0, 2.0, np.nan, np.nan])


@pytest.fixture
def y_proxy():
    return np.array([1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def engine():
    return PPIMeanEngine()


# --- preprocess ---


def test_preprocess_delegates_to_validation(engine, y_true, y_proxy):
    labeled_mask = np.array([True, True, False, False])
    with (
        patch.object(ppi_engine_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(ppi_engine_module, "_validate_has_no_nan") as mock_validate_has_no_nan,
        patch.object(ppi_engine_module, "_split_labeled_unlabeled") as mock_split_labeled_unlabeled,
        patch.object(ppi_engine_module, "_validate_sample_sizes") as mock_validate_sample_sizes,
    ):
        mock_split_labeled_unlabeled.return_value = (
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            labeled_mask,
        )
        engine.preprocess(y_true, y_proxy)

        mock_validate_equal_lengths.assert_called_once()
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], y_true)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], y_proxy)
        assert mock_validate_equal_lengths.call_args[1] == {"names": ["y_true", "y_proxy"]}
        mock_validate_has_no_nan.assert_called_once()
        np.testing.assert_array_equal(mock_validate_has_no_nan.call_args[0][0], y_proxy)
        assert mock_validate_has_no_nan.call_args[0][1] == "y_proxy"
        mock_split_labeled_unlabeled.assert_called_once()
        np.testing.assert_array_equal(mock_split_labeled_unlabeled.call_args[0][0], y_true)
        np.testing.assert_array_equal(mock_split_labeled_unlabeled.call_args[0][1], y_proxy)
        mock_validate_sample_sizes.assert_called_once()
        np.testing.assert_array_equal(mock_validate_sample_sizes.call_args[0][0], labeled_mask)


def test_preprocess_valid_output(engine, y_true, y_proxy):
    y_true_labeled, y_proxy_labeled, y_proxy_unlabeled = engine.preprocess(y_true, y_proxy)
    np.testing.assert_array_equal(y_true_labeled, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(y_proxy_labeled, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(y_proxy_unlabeled, np.array([3.0, 4.0]))


# --- fit_tuning_parameter ---


def test_fit_tuning_parameter_power_tuning_true(engine, y_true, y_proxy):
    dataset = engine.preprocess(y_true, y_proxy)
    tuning_parameter = engine.fit_tuning_parameter(dataset, power_tuning=True)
    assert tuning_parameter == pytest.approx(0.15)


def test_fit_tuning_parameter_power_tuning_false(engine, y_true, y_proxy):
    dataset = engine.preprocess(y_true, y_proxy)
    tuning_parameter = engine.fit_tuning_parameter(dataset, power_tuning=False)
    assert tuning_parameter == pytest.approx(1.0)


# --- compute_mean_and_std ---


def test_compute_mean_and_std(engine, y_true, y_proxy):
    dataset = engine.preprocess(y_true, y_proxy)
    mean, std = engine.compute_mean_and_std(dataset, tuning_parameter=0.15)
    assert mean == pytest.approx(1.8)
    assert std == pytest.approx(0.4316, abs=1e-4)
