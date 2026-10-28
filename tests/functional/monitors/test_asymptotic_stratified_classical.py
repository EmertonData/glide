"""Functional tests for AsymptoticStratifiedClassicalRM.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

import numpy as np
import pytest

from glide.estimators import StratifiedClassicalMeanEstimator
from glide.monitors import AsymptoticClassicalRM, AsymptoticStratifiedClassicalRM
from glide.samplers import StratifiedSampler
from glide.simulators import generate_batched_stratified_binary_dataset, simulate_annotation

# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def dataset():
    n_batches = 5
    n_labeled_per_batch = 14

    y_true_oracle, y_proxy, batches, groups = generate_batched_stratified_binary_dataset(
        n_samples=[[16, 12]] * n_batches,
        true_mean=[[0.5, 0.4]] * n_batches,
        proxy_mean=[[0.6, 0.65]] * n_batches,
        correlation=[[0.6, 0.5]] * n_batches,
        random_seed=0,
    )
    xi = np.zeros(len(y_true_oracle))
    for batch_id in np.unique(batches):
        batch_mask = batches == batch_id
        xi[batch_mask] = StratifiedSampler().sample(
            y_proxy[batch_mask],
            groups[batch_mask],
            n_samples=n_labeled_per_batch,
            strategy="proportional",
            random_seed=batch_id,
        )
    y = simulate_annotation(y_true_oracle, xi)
    return y, groups, batches


# ── tests ──────────────────────────────────────────────────────────────────────


def test_detect_single_stratum_equals_asymptotic_classical_rm(dataset):
    """With one stratum, the stratified monitor reduces to the unstratified classical monitor."""
    y, _, batches = dataset
    groups = np.zeros(len(y), dtype=int)

    stratified_result = AsymptoticStratifiedClassicalRM().detect(
        y, groups, batches, higher_is_better=False, threshold=0.5, tightest_at_batch=2
    )
    classical_result = AsymptoticClassicalRM().detect(
        y, batches, higher_is_better=False, threshold=0.5, tightest_at_batch=2
    )

    np.testing.assert_allclose(stratified_result.running_means, classical_result.running_means)
    np.testing.assert_allclose(stratified_result.confidence_bounds, classical_result.confidence_bounds)


def test_detect_batch_estimates_match_stratified_classical_estimator(dataset):
    """Each batch estimate equals the Stratified classical estimator on that batch alone."""
    y, groups, batches = dataset
    monitor_result = AsymptoticStratifiedClassicalRM().detect(y, groups, batches, higher_is_better=False, threshold=0.5)
    n_batches = len(np.unique(batches))
    estimator_means = np.zeros(n_batches)
    for batch_id in range(n_batches):
        batch_mask = batches == batch_id
        estimator_means[batch_id] = StratifiedClassicalMeanEstimator().estimate(y[batch_mask], groups[batch_mask]).mean

    np.testing.assert_allclose(monitor_result.batch_mean_estimates, estimator_means)


def test_detect_prefix_consistency(dataset):
    """Detecting on a growing history is prefix-consistent with detecting on the full history."""
    y, groups, batches = dataset
    monitor = AsymptoticStratifiedClassicalRM()
    full = monitor.detect(y, groups, batches, higher_is_better=False, threshold=0.5, tightest_at_batch=2)
    prefix_mask = batches <= 2
    prefix = monitor.detect(
        y[prefix_mask],
        groups[prefix_mask],
        batches[prefix_mask],
        higher_is_better=False,
        threshold=0.5,
        tightest_at_batch=2,
    )

    np.testing.assert_allclose(prefix.running_means, full.running_means[:3])
    np.testing.assert_allclose(prefix.confidence_bounds, full.confidence_bounds[:3])


def test_detect_higher_is_better_symmetry(dataset):
    """Monitoring a performance is the mirror image of monitoring its negation as a risk."""
    y, groups, batches = dataset
    risk = AsymptoticStratifiedClassicalRM().detect(y, groups, batches, higher_is_better=False, threshold=0.3)
    performance = AsymptoticStratifiedClassicalRM().detect(-y, groups, batches, higher_is_better=True, threshold=-0.3)

    np.testing.assert_array_equal(performance.alarms, risk.alarms)
    np.testing.assert_allclose(performance.confidence_bounds, -risk.confidence_bounds)
