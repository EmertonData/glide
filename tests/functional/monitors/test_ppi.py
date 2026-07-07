"""Functional tests for PPIMeanMonitor.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

import numpy as np
import pytest

from glide.estimators import PPIMeanEstimator
from glide.monitors import PPIMeanMonitor
from glide.simulators import generate_gaussian_dataset, simulate_annotation

# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def batches():
    return np.repeat(np.arange(5), 8)


@pytest.fixture
def y_true_oracle_and_proxy():
    return generate_gaussian_dataset(
        n_total=40, true_mean=0.5, true_std=1.0, proxy_mean=0.5, proxy_std=1.0, correlation=0.8, random_seed=0
    )


@pytest.fixture
def y_true(y_true_oracle_and_proxy):
    y_true_oracle, _ = y_true_oracle_and_proxy
    xi = np.tile(np.array([1, 1, 0, 0, 0, 0, 0, 0]), 5)
    return simulate_annotation(y_true_oracle, xi)


@pytest.fixture
def y_proxy(y_true_oracle_and_proxy):
    _, y_proxy = y_true_oracle_and_proxy
    return y_proxy


# ── tests ──────────────────────────────────────────────────────────────────────


def test_detect_batch_estimates_match_ppi_estimator(y_true, y_proxy, batches):
    """Each batch estimate equals the PPI estimator computed on that batch alone.

    Power tuning must be disabled on both sides: the monitor fits its tuning
    parameter on the batches strictly preceding the current one, so with power
    tuning enabled the two estimates would only agree by coincidence. Checking
    every batch (not just the first) exercises the ``power_tuning=False`` code
    path on batches that would otherwise be power-tuned.
    """
    result = PPIMeanMonitor().detect(
        y_true,
        y_proxy,
        batches,
        higher_is_better=False,
        threshold=0.5,
        metric_lower_bound=-5.0,
        metric_upper_bound=5.0,
        power_tuning=False,
    )
    n_batches = len(np.unique(batches))
    estimator_means = np.zeros(n_batches)
    for batch_id in range(n_batches):
        batch_mask = batches == batch_id
        estimator_result = PPIMeanEstimator().estimate(y_true[batch_mask], y_proxy[batch_mask], power_tuning=False)
        estimator_means[batch_id] = estimator_result.mean

    np.testing.assert_allclose(result.batch_mean_estimates, estimator_means)


def test_detect_prefix_consistency(y_true, y_proxy, batches):
    """Detecting on a growing history is prefix-consistent with detecting on the full history.

    Every batch is monitored (there is no reference batch to exclude), so restricting
    the call to the first k batches must reproduce exactly the first k entries of the
    arrays returned by the call on the full dataset. This is the property that makes
    repeated calls on a growing history jointly valid.
    """
    monitor = PPIMeanMonitor()
    full = monitor.detect(
        y_true,
        y_proxy,
        batches,
        higher_is_better=False,
        threshold=0.5,
        metric_lower_bound=-5.0,
        metric_upper_bound=5.0,
    )
    prefix_mask = batches <= 2
    prefix = monitor.detect(
        y_true[prefix_mask],
        y_proxy[prefix_mask],
        batches[prefix_mask],
        higher_is_better=False,
        threshold=0.5,
        metric_lower_bound=-5.0,
        metric_upper_bound=5.0,
    )

    np.testing.assert_allclose(prefix.running_means, full.running_means[:3])
    np.testing.assert_allclose(prefix.confidence_bounds, full.confidence_bounds[:3])


def test_detect_higher_is_better_symmetry(y_true, y_proxy, batches):
    """Monitoring a performance is the mirror image of monitoring its negation as a risk."""
    risk = PPIMeanMonitor().detect(
        y_true,
        y_proxy,
        batches,
        higher_is_better=False,
        threshold=0.3,
        metric_lower_bound=-5.0,
        metric_upper_bound=5.0,
    )
    performance = PPIMeanMonitor().detect(
        -y_true,
        -y_proxy,
        batches,
        higher_is_better=True,
        threshold=-0.3,
        metric_lower_bound=-5.0,
        metric_upper_bound=5.0,
    )

    np.testing.assert_array_equal(performance.alarms, risk.alarms)
    np.testing.assert_allclose(performance.confidence_bounds, -risk.confidence_bounds)


def test_detect_no_alarm_on_stationary_stream_below_threshold():
    """A stationary stream well below the threshold never triggers an alarm."""
    n_batches = 15
    y_true_oracle, y_proxy = generate_gaussian_dataset(
        n_total=n_batches * 8,
        true_mean=0.1,
        true_std=0.02,
        proxy_mean=0.1,
        proxy_std=0.02,
        correlation=0.8,
        random_seed=1,
    )
    xi = np.tile(np.array([1, 1, 0, 0, 0, 0, 0, 0]), n_batches)
    y_true = simulate_annotation(y_true_oracle, xi)
    batches = np.repeat(np.arange(n_batches), 8)

    result = PPIMeanMonitor().detect(
        y_true,
        y_proxy,
        batches,
        higher_is_better=False,
        threshold=0.5,
        metric_lower_bound=0.0,
        metric_upper_bound=1.0,
    )

    assert not result.drift_detected


def test_detect_alarm_on_stream_well_above_threshold():
    """A stream well above the threshold eventually triggers an alarm."""
    n_batches = 30
    y_true_oracle, y_proxy = generate_gaussian_dataset(
        n_total=n_batches * 8,
        true_mean=0.9,
        true_std=0.02,
        proxy_mean=0.9,
        proxy_std=0.02,
        correlation=0.8,
        random_seed=1,
    )
    xi = np.tile(np.array([1, 1, 0, 0, 0, 0, 0, 0]), n_batches)
    y_true = simulate_annotation(y_true_oracle, xi)
    batches = np.repeat(np.arange(n_batches), 8)

    result = PPIMeanMonitor().detect(
        y_true,
        y_proxy,
        batches,
        higher_is_better=False,
        threshold=0.5,
        metric_lower_bound=0.0,
        metric_upper_bound=1.0,
    )

    assert result.drift_detected
