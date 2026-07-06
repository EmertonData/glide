"""Functional tests for ClassicalMeanMonitor.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

import numpy as np
import pytest

from glide.estimators import ClassicalMeanEstimator
from glide.monitors import ClassicalMeanMonitor
from glide.simulators import generate_binary_dataset

# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def y():
    y_true, _ = generate_binary_dataset(n_total=20, true_mean=0.5, random_seed=0)
    return y_true


@pytest.fixture
def batches():
    return np.repeat(np.arange(5), 4)


# ── tests ──────────────────────────────────────────────────────────────────────


def test_detect_batch_mean_estimates_match_classical_estimator(y, batches):
    """Each batch estimate equals the classical estimator computed on that batch alone."""
    result = ClassicalMeanMonitor().detect(y, batches, higher_is_better=False, threshold=0.5)
    batch_mask = batches == 0
    estimator_result = ClassicalMeanEstimator().estimate(y[batch_mask])

    assert result.batch_mean_estimates[0] == pytest.approx(estimator_result.mean)


def test_detect_prefix_consistency():
    """Detecting on a growing history is prefix-consistent with detecting on the full history.

    Every batch is monitored (there is no reference batch to exclude), so restricting
    the call to the first k batches must reproduce exactly the first k entries of the
    arrays returned by the call on the full dataset. This is the property that makes
    repeated calls on a growing history jointly valid.
    """
    first_block = np.array([0.1, 0.2, 0.15, 0.25])
    later_block = np.array([0.3, 0.4, 0.35, 0.45])
    y = np.hstack([first_block, np.tile(later_block, 3)])
    batches = np.repeat(np.arange(4), 4)

    monitor = ClassicalMeanMonitor()
    full = monitor.detect(y, batches, higher_is_better=False, threshold=0.5)
    prefix_mask = batches <= 2
    prefix = monitor.detect(y[prefix_mask], batches[prefix_mask], higher_is_better=False, threshold=0.5)

    np.testing.assert_allclose(prefix.running_means, full.running_means[:3])
    np.testing.assert_allclose(prefix.confidence_bounds, full.confidence_bounds[:3])


def test_detect_higher_is_better_symmetry(y, batches):
    """Monitoring a performance is the mirror image of monitoring its complement as a risk."""
    risk = ClassicalMeanMonitor().detect(y, batches, higher_is_better=False, threshold=0.3)
    performance = ClassicalMeanMonitor().detect(1.0 - y, batches, higher_is_better=True, threshold=0.7)

    np.testing.assert_array_equal(performance.alarms, risk.alarms)
    np.testing.assert_allclose(performance.confidence_bounds, 1.0 - risk.confidence_bounds)


def test_detect_no_alarm_on_stationary_stream_below_threshold():
    """A stream stationary well below the threshold never raises a drift alarm."""
    y, _ = generate_binary_dataset(n_total=200, true_mean=0.2, proxy_mean=0.2, correlation=0.0, random_seed=1)
    batches = np.repeat(np.arange(20), 10)

    result = ClassicalMeanMonitor().detect(y, batches, higher_is_better=False, threshold=0.5)

    assert result.drift_detected is False


def test_detect_alarm_on_stream_above_threshold():
    """A stream stationary well above the threshold eventually raises a drift alarm."""
    y, _ = generate_binary_dataset(n_total=200, true_mean=0.9, proxy_mean=0.9, correlation=0.0, random_seed=1)
    batches = np.repeat(np.arange(20), 10)

    result = ClassicalMeanMonitor().detect(y, batches, higher_is_better=False, threshold=0.5)

    assert result.drift_detected is True
