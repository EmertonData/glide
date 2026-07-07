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
    n_batches = len(np.unique(batches))
    estimator_means = np.zeros(n_batches)
    for batch_id in range(n_batches):
        batch_mask = batches == batch_id
        estimator_result = ClassicalMeanEstimator().estimate(y[batch_mask])
        estimator_means[batch_id] = estimator_result.mean

    np.testing.assert_allclose(result.batch_mean_estimates, estimator_means)


def test_detect_prefix_consistency(y, batches):
    """Detecting on a growing history is prefix-consistent with detecting on the full history.

    Every batch is monitored (there is no reference batch to exclude), so restricting
    the call to the first k batches must reproduce exactly the first k entries of the
    arrays returned by the call on the full dataset. This is the property that makes
    repeated calls on a growing history jointly valid.
    """
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
