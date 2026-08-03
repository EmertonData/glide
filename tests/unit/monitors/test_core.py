import numpy as np
import pytest

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


def test_postprocess_lower_is_better(risk_running_means, risk_confidence_bounds, risk_batch_mean_estimates):
    running_means, confidence_bounds, batch_mean_estimates = _postprocess(
        risk_running_means, risk_confidence_bounds, risk_batch_mean_estimates, higher_is_better=False
    )

    np.testing.assert_allclose(running_means, risk_running_means)
    np.testing.assert_allclose(confidence_bounds, risk_confidence_bounds)
    np.testing.assert_allclose(batch_mean_estimates, risk_batch_mean_estimates)


def test_postprocess_higher_is_better(risk_running_means, risk_confidence_bounds, risk_batch_mean_estimates):
    running_means, confidence_bounds, batch_mean_estimates = _postprocess(
        risk_running_means, risk_confidence_bounds, risk_batch_mean_estimates, higher_is_better=True
    )

    np.testing.assert_allclose(running_means, -risk_running_means)
    np.testing.assert_allclose(confidence_bounds, -risk_confidence_bounds)
    np.testing.assert_allclose(batch_mean_estimates, -risk_batch_mean_estimates)
