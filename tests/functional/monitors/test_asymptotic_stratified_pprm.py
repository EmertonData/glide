"""Functional tests for AsymptoticStratifiedPPRM.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

import numpy as np
import pytest

from glide.estimators import StratifiedPPIMeanEstimator
from glide.monitors import AsymptoticStratifiedPPRM
from glide.simulators import generate_batched_stratified_binary_dataset, simulate_annotation

# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def dataset():
    n_batches = 5
    batch_size = 10
    n_labeled_per_stratum = 4

    y_true_oracle, y_proxy, batches, groups = generate_batched_stratified_binary_dataset(
        n_samples=[[batch_size, batch_size]] * n_batches,
        true_mean=[[0.5, 0.5]] * n_batches,
        proxy_mean=[[0.6, 0.65]] * n_batches,
        correlation=[[0.6, 0.6]] * n_batches,
        random_seed=0,
    )
    rng = np.random.default_rng(seed=1)
    xi = np.zeros(len(y_true_oracle))
    for batch_id in np.unique(batches):
        for stratum_id in np.unique(groups):
            stratum_indices = np.flatnonzero((batches == batch_id) & (groups == stratum_id))
            labeled_indices = rng.choice(stratum_indices, size=n_labeled_per_stratum, replace=False)
            xi[labeled_indices] = 1
    y_true = simulate_annotation(y_true_oracle, xi)
    return y_true, y_proxy, groups, batches


# ── tests ──────────────────────────────────────────────────────────────────────


def test_detect_batch_estimates_match_stratified_ppi_estimator(dataset):
    """Each batch estimate equals the Stratified PPI estimator on that batch alone when power tuning is disabled."""
    y_true, y_proxy, groups, batches = dataset
    monitor_result = AsymptoticStratifiedPPRM().detect(
        y_true,
        y_proxy,
        groups,
        batches,
        higher_is_better=False,
        threshold=0.5,
        power_tuning=False,
    )
    n_batches = len(np.unique(batches))
    estimator_means = np.zeros(n_batches)
    for batch_id in range(n_batches):
        batch_mask = batches == batch_id
        estimator_result = StratifiedPPIMeanEstimator().estimate(
            y_true[batch_mask], y_proxy[batch_mask], groups[batch_mask], power_tuning=False
        )
        estimator_means[batch_id] = estimator_result.mean

    np.testing.assert_allclose(monitor_result.batch_mean_estimates, estimator_means)


def test_detect_prefix_consistency(dataset):
    """Detecting on a growing history is prefix-consistent with detecting on the full history."""
    y_true, y_proxy, groups, batches = dataset
    monitor = AsymptoticStratifiedPPRM()
    full = monitor.detect(y_true, y_proxy, groups, batches, higher_is_better=False, threshold=0.5, tightest_at_batch=2)
    prefix_mask = batches <= 2
    prefix = monitor.detect(
        y_true[prefix_mask],
        y_proxy[prefix_mask],
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
    y_true, y_proxy, groups, batches = dataset
    risk = AsymptoticStratifiedPPRM().detect(y_true, y_proxy, groups, batches, higher_is_better=False, threshold=0.3)
    performance = AsymptoticStratifiedPPRM().detect(
        -y_true, -y_proxy, groups, batches, higher_is_better=True, threshold=-0.3
    )

    np.testing.assert_array_equal(performance.alarms, risk.alarms)
    np.testing.assert_allclose(performance.confidence_bounds, -risk.confidence_bounds)


def test_detect_stratum_relabeling_only_affects_subsequent_batches():
    """Data reassigned from one stratum to another partway through the stream changes the confidence bounds
    from that batch onward, even though the pooled data across strata is unchanged; earlier batches are unaffected."""
    y_true_oracle, y_proxy, batches, local_groups = generate_batched_stratified_binary_dataset(
        n_samples=[[20, 20, 20]] * 3,
        true_mean=[[0.5, 0.5, 0.5]] * 3,
        proxy_mean=[[0.6, 0.65, 0.7]] * 3,
        correlation=[[0.6, 0.6, 0.6]] * 3,
        random_seed=0,
    )
    rng = np.random.default_rng(seed=1)
    xi = np.zeros(len(y_true_oracle))
    for batch_id in np.unique(batches):
        for local_stratum_id in np.unique(local_groups):
            stratum_indices = np.flatnonzero((batches == batch_id) & (local_groups == local_stratum_id))
            labeled_indices = rng.choice(stratum_indices, size=8, replace=False)
            xi[labeled_indices] = 1
    y_true = simulate_annotation(y_true_oracle, xi)

    stable_groups = np.array(["a", "b", "c"])[local_groups.astype(int)]
    reassigned_groups = stable_groups.copy()
    reassigned_batch = 1
    reassigned_slot_mask = (local_groups == 2) & (batches == reassigned_batch)
    reassigned_groups[reassigned_slot_mask] = "d"

    monitor = AsymptoticStratifiedPPRM()
    stable_result = monitor.detect(
        y_true, y_proxy, stable_groups, batches, higher_is_better=False, threshold=0.5, tightest_at_batch=1
    )
    reassigned_result = monitor.detect(
        y_true, y_proxy, reassigned_groups, batches, higher_is_better=False, threshold=0.5, tightest_at_batch=1
    )

    np.testing.assert_allclose(stable_result.confidence_bounds[0], reassigned_result.confidence_bounds[0])
    assert not np.allclose(stable_result.confidence_bounds[1:], reassigned_result.confidence_bounds[1:])
