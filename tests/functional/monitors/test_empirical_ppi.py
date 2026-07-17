"""Functional tests for EmpiricalPPIMeanMonitor.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

import numpy as np
import pytest

from glide.estimators import PPIMeanEstimator
from glide.monitors import EmpiricalPPIMeanMonitor
from glide.simulators import generate_stratified_binary_dataset, simulate_annotation

# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def dataset():
    n_batches = 5
    batch_size = 20
    n_labeled_per_batch = 8

    y_true_oracle, y_proxy, batches = generate_stratified_binary_dataset(
        n_total=[batch_size] * n_batches,
        true_mean=[0.5] * n_batches,
        proxy_mean=[0.6] * n_batches,
        correlation=[0.8] * n_batches,
        random_seed=0,
    )
    rng = np.random.default_rng(seed=1)
    xis = []
    for _ in range(n_batches):
        xi_batch = np.zeros(batch_size)
        labeled_indices = rng.choice(batch_size, size=n_labeled_per_batch)
        xi_batch[labeled_indices] = 1
        xis.append(xi_batch)
    xi = np.hstack(xis)
    y_true = simulate_annotation(y_true_oracle, xi)
    return y_true, y_proxy, batches


# ── tests ──────────────────────────────────────────────────────────────────────


def test_detect_batch_estimates_match_ppi_estimator(dataset):
    """Each batch estimate equals the PPI estimator computed on that batch alone when power tuning is disabled."""
    y_true, y_proxy, batches = dataset
    monitor_result = EmpiricalPPIMeanMonitor().detect(
        y_true,
        y_proxy,
        batches,
        higher_is_better=False,
        threshold=0.5,
        power_tuning=False,
    )
    n_batches = len(np.unique(batches))
    estimator_means = np.zeros(n_batches)
    for batch_id in range(n_batches):
        batch_mask = batches == batch_id
        estimator_result = PPIMeanEstimator().estimate(y_true[batch_mask], y_proxy[batch_mask], power_tuning=False)
        estimator_means[batch_id] = estimator_result.mean

    np.testing.assert_allclose(monitor_result.batch_mean_estimates, estimator_means)


def test_detect_prefix_consistency(dataset):
    """Detecting on a growing history is prefix-consistent with detecting on the full history."""
    y_true, y_proxy, batches = dataset
    monitor = EmpiricalPPIMeanMonitor()
    full = monitor.detect(
        y_true,
        y_proxy,
        batches,
        higher_is_better=False,
        threshold=0.5,
    )
    prefix_mask = batches <= 2
    prefix = monitor.detect(
        y_true[prefix_mask],
        y_proxy[prefix_mask],
        batches[prefix_mask],
        higher_is_better=False,
        threshold=0.5,
    )

    np.testing.assert_allclose(prefix.running_means, full.running_means[:3])
    np.testing.assert_allclose(prefix.confidence_bounds, full.confidence_bounds[:3])


def test_detect_higher_is_better_symmetry(dataset):
    """Monitoring a performance is the mirror image of monitoring its negation as a risk."""
    y_true, y_proxy, batches = dataset
    risk = EmpiricalPPIMeanMonitor().detect(
        y_true,
        y_proxy,
        batches,
        higher_is_better=False,
        threshold=0.3,
    )
    performance = EmpiricalPPIMeanMonitor().detect(
        1 - y_true,
        1 - y_proxy,
        batches,
        higher_is_better=True,
        threshold=0.7,
    )

    np.testing.assert_array_equal(performance.alarms, risk.alarms)
    np.testing.assert_allclose(performance.confidence_bounds, 1 - risk.confidence_bounds)
