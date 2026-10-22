from unittest.mock import patch

import numpy as np
import pytest

from glide.engines.classical import ClassicalMeanEngine


@pytest.fixture
def y():
    return np.array([2.0, np.nan, 4.0])


@pytest.fixture
def engine():
    return ClassicalMeanEngine()


@pytest.fixture
def dataset(engine, y):
    return engine._preprocess(y)


# --- _preprocess ---


def test_preprocess_valid_output(engine, y):
    dataset = engine._preprocess(y)
    np.testing.assert_array_equal(dataset, np.array([2.0, 4.0]))


def test_preprocess_delegates_to_validation(engine, y):
    with patch("glide.engines.classical._validate_min_samples") as mock_validate_min_samples:
        engine._preprocess(y)
    mock_validate_min_samples.assert_called_once()
    np.testing.assert_array_equal(mock_validate_min_samples.call_args[0][0], np.array([2.0, 4.0]))
    assert mock_validate_min_samples.call_args[0][1] == "y"


# --- _fit_tuning_parameter ---


def test_fit_tuning_parameter_returns_none(engine, dataset):
    assert engine._fit_tuning_parameter(dataset, power_tuning=True) is None
    assert engine._fit_tuning_parameter(dataset, power_tuning=False) is None


# --- _compute_mean_and_std ---


def test_compute_mean_and_std(engine, dataset):
    mean, std = engine._compute_mean_and_std(dataset, tuning_parameter=None)
    assert mean == pytest.approx(3.0)
    assert std == pytest.approx(1.0)
