"""Functional tests for AsymptoticClusteredClassicalRM.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

import numpy as np
import pytest

from glide.estimators import ClusteredClassicalMeanEstimator
from glide.monitors import AsymptoticClusteredClassicalRM
from glide.samplers import UniformClusteredSampler
from glide.simulators import generate_batched_clustered_binary_dataset, simulate_annotation

# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def dataset():
    n_batches = 5
    n_samples_per_batch = 40
    n_clusters_per_batch = 8
    n_labeled_clusters_per_batch = 3

    y_true_oracle, y_proxy, batches, clusters = generate_batched_clustered_binary_dataset(
        n_samples=[n_samples_per_batch] * n_batches,
        n_clusters=[n_clusters_per_batch] * n_batches,
        true_mean=[0.5] * n_batches,
        proxy_mean=[0.6] * n_batches,
        correlation=[0.8] * n_batches,
        random_seed=0,
    )
    xi = np.zeros(len(y_true_oracle))
    for batch_id in range(n_batches):
        batch_mask = batches == batch_id
        xi[batch_mask] = UniformClusteredSampler().sample(
            clusters[batch_mask], n_clusters=n_labeled_clusters_per_batch, random_seed=batch_id
        )
    y = simulate_annotation(y_true_oracle, xi)
    return y, clusters, batches


# ── tests ──────────────────────────────────────────────────────────────────────


def test_detect_batch_estimates_match_clustered_classical_estimator(dataset):
    """Each batch estimate equals the Clustered Classical estimator on that batch alone."""
    y, clusters, batches = dataset
    monitor_result = AsymptoticClusteredClassicalRM().detect(
        y, clusters, batches, higher_is_better=False, threshold=0.5
    )
    n_batches = len(np.unique(batches))
    estimator_means = np.zeros(n_batches)
    for batch_id in range(n_batches):
        batch_mask = batches == batch_id
        estimator_means[batch_id] = ClusteredClassicalMeanEstimator().estimate(y[batch_mask], clusters[batch_mask]).mean

    np.testing.assert_allclose(monitor_result.batch_mean_estimates, estimator_means)


def test_detect_prefix_consistency(dataset):
    """Detecting on a growing history is prefix-consistent with detecting on the full history."""
    y, clusters, batches = dataset
    monitor = AsymptoticClusteredClassicalRM()
    full = monitor.detect(y, clusters, batches, higher_is_better=False, threshold=0.5, tightest_at_batch=2)
    prefix_mask = batches <= 2
    prefix = monitor.detect(
        y[prefix_mask],
        clusters[prefix_mask],
        batches[prefix_mask],
        higher_is_better=False,
        threshold=0.5,
        tightest_at_batch=2,
    )

    np.testing.assert_allclose(prefix.running_means, full.running_means[:3])
    np.testing.assert_allclose(prefix.confidence_bounds, full.confidence_bounds[:3])


def test_detect_higher_is_better_symmetry(dataset):
    """Monitoring a performance is the mirror image of monitoring its negation as a risk."""
    y, clusters, batches = dataset
    risk = AsymptoticClusteredClassicalRM().detect(y, clusters, batches, higher_is_better=False, threshold=0.3)
    performance = AsymptoticClusteredClassicalRM().detect(-y, clusters, batches, higher_is_better=True, threshold=-0.3)

    np.testing.assert_array_equal(performance.alarms, risk.alarms)
    np.testing.assert_allclose(performance.confidence_bounds, -risk.confidence_bounds)
