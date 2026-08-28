from unittest.mock import patch

import numpy as np
import pytest

import glide.monitors.core as core_module
from glide.monitors.core import _postprocess, _preprocess, _reorient, _unique_ordered_batches


@pytest.fixture
def risk_running_means():
    return np.array([0.2, 0.25])


@pytest.fixture
def risk_confidence_bounds():
    return np.array([0.1, 0.2])


@pytest.fixture
def risk_batch_mean_estimates():
    return np.array([0.2, 0.3])


@pytest.fixture
def fields():
    return [np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.5, 0.6, 0.7, 0.8])]


@pytest.fixture
def field_names():
    return ["field_a", "field_b"]


@pytest.fixture
def preprocess_batches():
    return np.array([0, 0, 1, 1])


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


# --- _preprocess ---


def test_preprocess_delegates_to_validation(fields, field_names, preprocess_batches):
    with (
        patch.object(core_module, "_validate_bounds") as mock_validate_bounds,
        patch.object(core_module, "_validate_non_empty") as mock_validate_non_empty,
        patch.object(core_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(core_module, "_validate_has_no_nan") as mock_validate_has_no_nan,
    ):
        _preprocess(fields, field_names, preprocess_batches, higher_is_better=False, confidence_level=0.8)

    mock_validate_bounds.assert_called_once_with(
        0.8,
        "confidence_level",
        lower=0.5,
        upper=1,
        left_inclusive=False,
        right_inclusive=False,
        error_message="'confidence_level' must be in (0.5, 1) for the asymptotic monitor; got 0.8.",
    )
    mock_validate_non_empty.assert_called_once()
    np.testing.assert_array_equal(mock_validate_non_empty.call_args[0][0], preprocess_batches)
    assert mock_validate_non_empty.call_args[0][1] == "batches"
    mock_validate_equal_lengths.assert_called_once()
    np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], fields[0])
    np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], fields[1])
    np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][2], preprocess_batches)
    assert mock_validate_equal_lengths.call_args[1] == {"names": ["field_a", "field_b", "batches"]}
    mock_validate_has_no_nan.assert_called_once()
    np.testing.assert_array_equal(mock_validate_has_no_nan.call_args[0][0], preprocess_batches)
    assert mock_validate_has_no_nan.call_args[0][1] == "batches"


def test_preprocess_known_output(fields, field_names, preprocess_batches):
    expected_batch_identifiers = np.array([0, 1])
    expected_batch_codes = np.array([0, 0, 1, 1])

    risk_fields, batch_identifiers, batch_codes = _preprocess(
        fields, field_names, preprocess_batches, higher_is_better=True, confidence_level=0.8
    )

    np.testing.assert_array_equal(risk_fields[0], -fields[0])
    np.testing.assert_array_equal(risk_fields[1], -fields[1])
    np.testing.assert_array_equal(batch_identifiers, expected_batch_identifiers)
    np.testing.assert_array_equal(batch_codes, expected_batch_codes)


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
