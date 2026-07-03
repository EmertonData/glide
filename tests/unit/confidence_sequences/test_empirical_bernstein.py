from unittest.mock import patch

import numpy as np
import pytest

import glide.confidence_sequences.empirical_bernstein as empirical_bernstein_module
from glide.confidence_sequences import EmpiricalBernsteinConfidenceSequence
from glide.confidence_sequences.empirical_bernstein import (
    _compute_empirical_bernstein_bounds,
    _compute_mixture_boundary,
    _compute_mixture_wealth,
)

EXPECTED_LOWER_BOUNDS = np.array([-2.27458313, -0.86521862, -0.41014575])


# --- _compute_mixture_wealth ---


def test_compute_mixture_wealth_known_value():
    wealth = _compute_mixture_wealth(0.5, 0.1)
    assert wealth == pytest.approx(1.227, abs=1e-3)


# --- _compute_mixture_boundary ---


@pytest.mark.parametrize("miscoverage", [0.1, 0.2, 0.5])
def test_compute_mixture_boundary_zero_variance(miscoverage):
    boundary = _compute_mixture_boundary(0.0, miscoverage)
    assert (np.exp(boundary) - 1) / boundary == pytest.approx(1 / miscoverage, abs=1e-6)


def test_compute_mixture_boundary_increases_with_variance():
    small = _compute_mixture_boundary(0.5, 0.1)
    large = _compute_mixture_boundary(2.0, 0.1)
    assert small == pytest.approx(4.298, abs=1e-3)
    assert large == pytest.approx(5.780, abs=1e-3)


def test_compute_mixture_boundary_increases_with_confidence():
    loose = _compute_mixture_boundary(0.5, 0.2)
    tight = _compute_mixture_boundary(0.5, 0.05)
    assert loose == pytest.approx(3.283, abs=1e-3)
    assert tight == pytest.approx(5.249, abs=1e-3)


# --- _compute_empirical_bernstein_bounds ---


@pytest.fixture
def batch_estimates():
    return np.array([0.4, 0.6, 0.5])


def test_compute_empirical_bernstein_bounds_delegates_to_validation(batch_estimates):
    with patch.object(empirical_bernstein_module, "_validate_bounds") as mock_validate_bounds:
        _compute_empirical_bernstein_bounds(batch_estimates, seed_center=0.5, miscoverage=0.2)

        mock_validate_bounds.assert_called_once_with(
            0.2, "miscoverage", lower=0.0, upper=1.0, left_inclusive=False, right_inclusive=False
        )


def test_compute_empirical_bernstein_bounds(batch_estimates):
    running_mean_estimates, lower_bounds = _compute_empirical_bernstein_bounds(
        batch_estimates, seed_center=0.5, miscoverage=0.2
    )
    np.testing.assert_allclose(running_mean_estimates, np.array([0.4, 0.5, 0.5]))
    np.testing.assert_allclose(lower_bounds, EXPECTED_LOWER_BOUNDS, atol=1e-6)


# --- EmpiricalBernsteinConfidenceSequence ---


@pytest.fixture
def sequence():
    return EmpiricalBernsteinConfidenceSequence(
        running_mean_estimates=np.array([0.4, 0.6]),
        confidence_bounds=np.array([0.1, 0.55]),
    )


def test_sequence_attributes(sequence):
    np.testing.assert_array_equal(sequence.running_mean_estimates, np.array([0.4, 0.6]))
    np.testing.assert_array_equal(sequence.confidence_bounds, np.array([0.1, 0.55]))


def test_null_hypothesis_delegates_to_validation(sequence):
    with patch.object(empirical_bernstein_module, "_validate_literal") as mock_validate_literal:
        sequence.test_null_hypothesis(0.5, alternative="larger")

        mock_validate_literal.assert_called_once_with("larger", "alternative", ["larger", "smaller"])


def test_null_hypothesis_larger(sequence):
    alarms = sequence.test_null_hypothesis(0.5, alternative="larger")
    np.testing.assert_array_equal(alarms, np.array([False, True]))


def test_null_hypothesis_smaller(sequence):
    alarms = sequence.test_null_hypothesis(0.3, alternative="smaller")
    np.testing.assert_array_equal(alarms, np.array([True, False]))
