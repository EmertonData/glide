from unittest.mock import patch

import numpy as np
import pytest

import glide.monitors.core as core_module
from glide.monitors.core import _postprocess, _scale_from_unit_risk, _scale_to_unit_risk, _unique_ordered_batches


@pytest.fixture
def values_unscaled():
    return np.array([0.0, 5.0, 10.0])


@pytest.fixture
def values_scaled():
    return np.array([0.0, 0.5, 1.0])


# --- _unique_ordered_batches ---


def test_unique_ordered_batches_integers():
    batches = np.array([2, 2, 0, 0, 0, 1])
    expected_batch_identifiers = np.array([2, 0, 1])
    expected_batch_codes = np.array([0, 0, 1, 1, 1, 2])

    batch_identifiers, batch_codes = _unique_ordered_batches(batches)

    np.testing.assert_array_equal(batch_identifiers, expected_batch_identifiers)
    np.testing.assert_array_equal(batch_codes, expected_batch_codes)


def test_unique_ordered_batches_interleaved_raises():
    batches = np.array(["A", "B", "A", "B"])
    with pytest.raises(ValueError, match="'batches' must be grouped into contiguous blocks"):
        _unique_ordered_batches(batches)


# --- _scale_to_unit_risk ---


def test_scale_to_unit_risk_lower_is_better(values_unscaled):
    expected = np.array([0.0, 0.5, 1.0])
    result = _scale_to_unit_risk(
        values_unscaled, metric_lower_bound=0.0, metric_upper_bound=10.0, higher_is_better=False
    )
    np.testing.assert_allclose(result, expected)


def test_scale_to_unit_risk_higher_is_better(values_unscaled):
    expected = np.array([1.0, 0.5, 0.0])
    result = _scale_to_unit_risk(
        values_unscaled, metric_lower_bound=0.0, metric_upper_bound=10.0, higher_is_better=True
    )
    np.testing.assert_allclose(result, expected)


# --- _scale_from_unit_risk ---


def test_scale_from_unit_risk_lower_is_better(values_scaled):
    expected = np.array([0.0, 5.0, 10.0])
    result = _scale_from_unit_risk(
        values_scaled, metric_lower_bound=0.0, metric_upper_bound=10.0, higher_is_better=False
    )
    np.testing.assert_allclose(result, expected)


def test_scale_from_unit_risk_higher_is_better(values_scaled):
    expected = np.array([10.0, 5.0, 0.0])
    result = _scale_from_unit_risk(
        values_scaled, metric_lower_bound=0.0, metric_upper_bound=10.0, higher_is_better=True
    )
    np.testing.assert_allclose(result, expected)


# --- _postprocess ---


def test_postprocess_delegates_to_scaling():
    risk_running_means = np.array([0.2, 0.25])
    risk_confidence_bounds = np.array([0.1, 0.2])
    risk_batch_mean_estimates = np.array([0.2, 0.3])

    with patch.object(core_module, "_scale_from_unit_risk") as mock_scale_from_unit_risk:
        _postprocess(
            risk_running_means,
            risk_confidence_bounds,
            risk_batch_mean_estimates,
            higher_is_better=True,
            metric_lower_bound=0.0,
            metric_upper_bound=1.0,
        )

    assert mock_scale_from_unit_risk.call_count == 3
    np.testing.assert_array_equal(mock_scale_from_unit_risk.call_args_list[0][0][0], risk_running_means)
    np.testing.assert_array_equal(mock_scale_from_unit_risk.call_args_list[1][0][0], risk_confidence_bounds)
    np.testing.assert_array_equal(mock_scale_from_unit_risk.call_args_list[2][0][0], risk_batch_mean_estimates)
