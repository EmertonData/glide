from unittest.mock import patch

import numpy as np
import pytest

import glide.monitors.core as core_module
from glide.monitors.core import _scale_from_unit_risk, _scale_to_unit_risk, _unique_ordered_batches


@pytest.fixture
def values_unscaled():
    return np.array([0.0, 5.0, 10.0])


@pytest.fixture
def values_scaled():
    return np.array([0.0, 0.5, 1.0])


# --- _unique_ordered_batches ---


def test_unique_ordered_batches_delegates_to_validation():
    batches = np.array([0, 0, 1])
    with patch.object(core_module, "_validate_has_no_nan") as mock_validate_has_no_nan:
        _unique_ordered_batches(batches)
    mock_validate_has_no_nan.assert_called_once()
    np.testing.assert_array_equal(mock_validate_has_no_nan.call_args[0][0], batches)
    assert mock_validate_has_no_nan.call_args[0][1] == "batches"


def test_unique_ordered_batches_integers():
    batches = np.array([2, 2, 0, 0, 0, 1])
    batch_identifiers, batch_codes = _unique_ordered_batches(batches)

    np.testing.assert_array_equal(batch_identifiers, np.array([2, 0, 1]))
    np.testing.assert_array_equal(batch_codes, np.array([0, 0, 1, 1, 1, 2]))


def test_unique_ordered_batches_interleaved_raises():
    batches = np.array(["A", "B", "A", "B"])
    with pytest.raises(ValueError, match="'batches' must be grouped into contiguous blocks"):
        _unique_ordered_batches(batches)


# --- _scale_to_unit_risk ---


def test_scale_to_unit_risk_lower_is_better(values_unscaled):
    result = _scale_to_unit_risk(
        values_unscaled, metric_lower_bound=0.0, metric_upper_bound=10.0, higher_is_better=False
    )
    np.testing.assert_allclose(result, np.array([0.0, 0.5, 1.0]))


def test_scale_to_unit_risk_higher_is_better(values_unscaled):
    result = _scale_to_unit_risk(
        values_unscaled, metric_lower_bound=0.0, metric_upper_bound=10.0, higher_is_better=True
    )
    np.testing.assert_allclose(result, np.array([1.0, 0.5, 0.0]))


# --- _scale_from_unit_risk ---


def test_scale_from_unit_risk_lower_is_better(values_scaled):
    result = _scale_from_unit_risk(
        values_scaled, metric_lower_bound=0.0, metric_upper_bound=10.0, higher_is_better=False
    )
    np.testing.assert_allclose(result, np.array([0.0, 5.0, 10.0]))


def test_scale_from_unit_risk_higher_is_better(values_scaled):
    result = _scale_from_unit_risk(
        values_scaled, metric_lower_bound=0.0, metric_upper_bound=10.0, higher_is_better=True
    )
    np.testing.assert_allclose(result, np.array([10.0, 5.0, 0.0]))
