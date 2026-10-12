"""Functional tests for AsymptoticClusteredPPRM.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

import numpy as np
import pytest

from glide.estimators import ClusteredPPIMeanEstimator
from glide.monitors import AsymptoticClusteredPPRM
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
    y_true = simulate_annotation(y_true_oracle, xi)
    return y_true, y_proxy, clusters, batches


# ── tests ──────────────────────────────────────────────────────────────────────


def test_detect_batch_estimates_match_clustered_ppi_estimator(dataset):
    """Each batch estimate equals the Clustered PPI estimator on that batch alone when power tuning is disabled."""
    y_true, y_proxy, clusters, batches = dataset
    monitor_result = AsymptoticClusteredPPRM().detect(
        y_true,
        y_proxy,
        clusters,
        batches,
        higher_is_better=False,
        threshold=0.5,
        power_tuning=False,
    )
    n_batches = len(np.unique(batches))
    estimator_means = np.zeros(n_batches)
    for batch_id in range(n_batches):
        batch_mask = batches == batch_id
        estimator_result = ClusteredPPIMeanEstimator().estimate(
            y_true[batch_mask], y_proxy[batch_mask], clusters[batch_mask], power_tuning=False
        )
        estimator_means[batch_id] = estimator_result.mean

    np.testing.assert_allclose(monitor_result.batch_mean_estimates, estimator_means)


def test_detect_prefix_consistency(dataset):
    """Detecting on a growing history is prefix-consistent with detecting on the full history."""
    y_true, y_proxy, clusters, batches = dataset
    monitor = AsymptoticClusteredPPRM()
    full = monitor.detect(
        y_true, y_proxy, clusters, batches, higher_is_better=False, threshold=0.5, tightest_at_batch=2
    )
    prefix_mask = batches <= 2
    prefix = monitor.detect(
        y_true[prefix_mask],
        y_proxy[prefix_mask],
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
    y_true, y_proxy, clusters, batches = dataset
    risk = AsymptoticClusteredPPRM().detect(y_true, y_proxy, clusters, batches, higher_is_better=False, threshold=0.3)
    performance = AsymptoticClusteredPPRM().detect(
        -y_true, -y_proxy, clusters, batches, higher_is_better=True, threshold=-0.3
    )

    np.testing.assert_array_equal(performance.alarms, risk.alarms)
    np.testing.assert_allclose(performance.confidence_bounds, -risk.confidence_bounds)
