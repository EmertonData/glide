from unittest.mock import patch

import numpy as np
import pytest

import glide.engines.multi_ppi as multi_ppi_engine_module
from glide.engines.multi_ppi import MultiPPIMeanEngine


@pytest.fixture
def y_true():
    return np.array([1.0, 2.0, np.nan, np.nan])


@pytest.fixture
def y_proxies():
    return np.array([[1.0, 0.0], [2.0, 2.0], [3.0, 1.0], [4.0, 3.0]])


@pytest.fixture
def engine():
    return MultiPPIMeanEngine()


@pytest.fixture
def dataset(engine, y_true, y_proxies):
    return engine.preprocess(y_true, y_proxies)


# --- preprocess ---


def test_preprocess_delegates_to_validation(engine, y_true, y_proxies):
    labeled_mask = np.array([True, True, False, False])
    with (
        patch.object(multi_ppi_engine_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(multi_ppi_engine_module, "_validate_is_2d") as mock_validate_is_2d,
        patch.object(multi_ppi_engine_module, "_validate_has_no_nan") as mock_validate_has_no_nan,
        patch.object(multi_ppi_engine_module, "_split_labeled_unlabeled") as mock_split_labeled_unlabeled,
        patch.object(multi_ppi_engine_module, "_validate_sample_sizes") as mock_validate_sample_sizes,
    ):
        mock_split_labeled_unlabeled.return_value = (
            np.array([1.0, 2.0]),
            np.array([[1.0, 0.0], [2.0, 2.0]]),
            np.array([[3.0, 1.0], [4.0, 3.0]]),
            labeled_mask,
        )
        engine.preprocess(y_true, y_proxies)

        mock_validate_equal_lengths.assert_called_once()
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], y_true)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], y_proxies)

        assert mock_validate_equal_lengths.call_args[1] == {"names": ["y_true", "y_proxies"]}

        mock_validate_is_2d.assert_called_once()
        np.testing.assert_array_equal(mock_validate_is_2d.call_args[0][0], y_proxies)
        assert mock_validate_is_2d.call_args[0][1] == "y_proxies"

        mock_validate_has_no_nan.assert_called_once()
        np.testing.assert_array_equal(mock_validate_has_no_nan.call_args[0][0], y_proxies)
        assert mock_validate_has_no_nan.call_args[0][1] == "y_proxies"

        mock_split_labeled_unlabeled.assert_called_once()
        np.testing.assert_array_equal(mock_split_labeled_unlabeled.call_args[0][0], y_true)
        np.testing.assert_array_equal(mock_split_labeled_unlabeled.call_args[0][1], y_proxies)

        mock_validate_sample_sizes.assert_called_once()
        np.testing.assert_array_equal(mock_validate_sample_sizes.call_args[0][0], labeled_mask)


def test_preprocess_valid_output(engine, y_true, y_proxies):
    y_true_labeled, y_proxies_labeled, y_proxies_unlabeled = engine.preprocess(y_true, y_proxies)
    np.testing.assert_array_equal(y_true_labeled, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(y_proxies_labeled, np.array([[1.0, 0.0], [2.0, 2.0]]))
    np.testing.assert_array_equal(y_proxies_unlabeled, np.array([[3.0, 1.0], [4.0, 3.0]]))


# --- fit_tuning_parameter ---


def test_fit_tuning_parameter_power_tuning_true(engine, dataset):
    tuning_parameter = engine.fit_tuning_parameter(dataset, power_tuning=True)
    expected = np.array([-0.25, 0.5])
    np.testing.assert_allclose(tuning_parameter, expected)


def test_fit_tuning_parameter_power_tuning_false(engine, dataset):
    tuning_parameter = engine.fit_tuning_parameter(dataset, power_tuning=False)
    expected = np.full(2, 1 / np.sqrt(2))
    np.testing.assert_allclose(tuning_parameter, expected)


# --- compute_mean_and_std ---


def test_compute_mean_and_std(engine, dataset):
    mean, std = engine.compute_mean_and_std(dataset, tuning_parameter=np.array([0.5, 0.5]))
    assert mean == pytest.approx(3.0)
    assert std == pytest.approx(0.7906, abs=1e-4)
