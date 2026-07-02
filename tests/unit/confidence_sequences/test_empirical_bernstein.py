import numpy as np
import pytest

from glide.confidence_sequences import EmpiricalBernsteinConfidenceSequence
from glide.confidence_sequences.empirical_bernstein import (
    _compute_empirical_bernstein_bounds,
    _compute_mixture_boundary,
    _compute_mixture_wealth,
)

EXPECTED_LOWER_BOUNDS = np.array([-2.27458313, -0.86521862, -0.41014575])


# --- _compute_mixture_wealth ---


def test_compute_mixture_wealth_positive():
    wealth = _compute_mixture_wealth(0.5, 0.1)
    assert wealth > 0


# --- _compute_mixture_boundary ---


@pytest.mark.parametrize("miscoverage", [0.1, 0.2, 0.5])
def test_compute_mixture_boundary_zero_variance(miscoverage):
    boundary = _compute_mixture_boundary(0.0, miscoverage)
    assert (np.exp(boundary) - 1) / boundary == pytest.approx(1 / miscoverage, abs=1e-6)


def test_compute_mixture_boundary_increases_with_variance():
    small = _compute_mixture_boundary(0.5, 0.1)
    large = _compute_mixture_boundary(2.0, 0.1)
    assert small < large


def test_compute_mixture_boundary_increases_with_confidence():
    loose = _compute_mixture_boundary(0.5, 0.2)
    tight = _compute_mixture_boundary(0.5, 0.05)
    assert loose < tight


# --- _compute_empirical_bernstein_bounds ---


def test_compute_empirical_bernstein_bounds():
    batch_estimates = np.array([0.4, 0.6, 0.5])
    running_mean_estimates, lower_bounds = _compute_empirical_bernstein_bounds(
        batch_estimates, seed_center=0.5, miscoverage=0.2
    )
    np.testing.assert_allclose(running_mean_estimates, np.array([0.4, 0.5, 0.5]))
    np.testing.assert_allclose(lower_bounds, EXPECTED_LOWER_BOUNDS, atol=1e-6)


def test_compute_empirical_bernstein_bounds_single_batch():
    batch_estimates = np.array([0.7])
    running_mean_estimates, lower_bounds = _compute_empirical_bernstein_bounds(
        batch_estimates, seed_center=0.5, miscoverage=0.1
    )
    np.testing.assert_array_equal(running_mean_estimates, np.array([0.7]))
    assert len(lower_bounds) == 1


# --- EmpiricalBernsteinConfidenceSequence ---


@pytest.fixture
def sequence():
    return EmpiricalBernsteinConfidenceSequence(
        running_mean_estimates=np.array([0.4, 0.6]),
        confidence_bounds=np.array([0.1, 0.55]),
    )


def test_running_mean_estimates_stored(sequence):
    np.testing.assert_array_equal(sequence.running_mean_estimates, np.array([0.4, 0.6]))


def test_confidence_bounds_stored(sequence):
    np.testing.assert_array_equal(sequence.confidence_bounds, np.array([0.1, 0.55]))


def test_null_hypothesis_larger(sequence):
    alarms = sequence.test_null_hypothesis(0.5, alternative="larger")
    np.testing.assert_array_equal(alarms, np.array([False, True]))


def test_null_hypothesis_smaller(sequence):
    alarms = sequence.test_null_hypothesis(0.3, alternative="smaller")
    np.testing.assert_array_equal(alarms, np.array([True, False]))


def test_null_hypothesis_invalid_alternative(sequence):
    with pytest.raises(ValueError, match="'alternative' must be"):
        sequence.test_null_hypothesis(0.5, alternative="two-sided")
