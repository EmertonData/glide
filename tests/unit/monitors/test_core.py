from unittest.mock import patch

import numpy as np
import pytest

import glide.monitors.core as core_module
from glide.monitors.core import _postprocess, _reorient, _unique_ordered_batches


@pytest.fixture
def risk_running_means():
    return np.array([0.2, 0.25])


@pytest.fixture
def risk_confidence_bounds():
    return np.array([0.1, 0.2])


@pytest.fixture
def risk_batch_mean_estimates():
    return np.array([0.2, 0.3])


# --- _reorient ---


def test_reorient_lower_is_better(risk_running_means):
    reoriented_value = _reorient(risk_running_means, higher_is_better=False)

    np.testing.assert_array_equal(reoriented_value, risk_running_means)


def test_reorient_higher_is_better(risk_running_means):
    reoriented_value = _reorient(risk_running_means, higher_is_better=True)

    np.testing.assert_array_equal(reoriented_value, -risk_running_means)


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


# --- _postprocess ---


def test_postprocess_delegates_to_reorient(risk_running_means, risk_confidence_bounds, risk_batch_mean_estimates):
    with patch.object(core_module, "_reorient") as mock_reorient:
        _postprocess(
            risk_running_means,
            risk_confidence_bounds,
            risk_batch_mean_estimates,
            higher_is_better=True,
        )

    assert mock_reorient.call_count == 3
    np.testing.assert_array_equal(mock_reorient.call_args_list[0][0][0], risk_running_means)
    assert mock_reorient.call_args_list[0][0][1] is True
    np.testing.assert_array_equal(mock_reorient.call_args_list[1][0][0], risk_confidence_bounds)
    assert mock_reorient.call_args_list[1][0][1] is True
    np.testing.assert_array_equal(mock_reorient.call_args_list[2][0][0], risk_batch_mean_estimates)
    assert mock_reorient.call_args_list[2][0][1] is True
