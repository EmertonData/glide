"""Functional tests for AsymptoticMultiPPRM.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

import numpy as np
import pytest

from glide.estimators import MultiPPIMeanEstimator
from glide.monitors import AsymptoticMultiPPRM
from glide.simulators import generate_stratified_multi_binary_dataset, simulate_annotation

# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def dataset():
    n_batches = 5
    batch_size = 20
    n_labeled_per_batch = 8

    y_true_oracle, y_proxies, batches = generate_stratified_multi_binary_dataset(
        n_samples=[batch_size] * n_batches,
        true_mean=[0.5] * n_batches,
        proxy_means=[[0.6, 0.55]] * n_batches,
        correlations=[[0.8, 0.7]] * n_batches,
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
    return y_true, y_proxies, batches


# ── tests ──────────────────────────────────────────────────────────────────────


def test_detect_batch_estimates_match_multi_ppi_estimator(dataset):
    """Each batch estimate equals the Multi-PPI estimator computed on that batch alone when power tuning is disabled."""
    y_true, y_proxies, batches = dataset
    monitor_result = AsymptoticMultiPPRM().detect(
        y_true,
        y_proxies,
        batches,
        higher_is_better=False,
        threshold=0.5,
        power_tuning=False,
    )
    n_batches = len(np.unique(batches))
    estimator_means = np.zeros(n_batches)
    for batch_id in range(n_batches):
        batch_mask = batches == batch_id
        estimator_result = MultiPPIMeanEstimator().estimate(
            y_true[batch_mask], y_proxies[batch_mask], power_tuning=False
        )
        estimator_means[batch_id] = estimator_result.mean

    np.testing.assert_allclose(monitor_result.batch_mean_estimates, estimator_means)


def test_detect_prefix_consistency(dataset):
    """Detecting on a growing history is prefix-consistent with detecting on the full history."""
    y_true, y_proxies, batches = dataset
    monitor = AsymptoticMultiPPRM()
    full = monitor.detect(y_true, y_proxies, batches, higher_is_better=False, threshold=0.5, tightest_at_batch=2)
    prefix_mask = batches <= 2
    prefix = monitor.detect(
        y_true[prefix_mask],
        y_proxies[prefix_mask],
        batches[prefix_mask],
        higher_is_better=False,
        threshold=0.5,
        tightest_at_batch=2,
    )

    np.testing.assert_allclose(prefix.running_means, full.running_means[:3])
    np.testing.assert_allclose(prefix.confidence_bounds, full.confidence_bounds[:3])


def test_detect_higher_is_better_symmetry(dataset):
    """Monitoring a performance is the mirror image of monitoring its negation as a risk."""
    y_true, y_proxies, batches = dataset
    risk = AsymptoticMultiPPRM().detect(y_true, y_proxies, batches, higher_is_better=False, threshold=0.3)
    performance = AsymptoticMultiPPRM().detect(-y_true, -y_proxies, batches, higher_is_better=True, threshold=-0.3)

    np.testing.assert_array_equal(performance.alarms, risk.alarms)
    np.testing.assert_allclose(performance.confidence_bounds, -risk.confidence_bounds)
