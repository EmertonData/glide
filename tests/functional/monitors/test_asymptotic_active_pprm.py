"""Functional tests for AsymptoticActivePPRM.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

import numpy as np
import pytest

from glide.estimators import ASIMeanEstimator
from glide.monitors import AsymptoticActivePPRM, AsymptoticPPRM
from glide.samplers import ActiveSampler
from glide.simulators import (
    generate_batched_binary_dataset_with_oracle_sampling,
    generate_gaussian_dataset,
    simulate_annotation,
)

N_BATCHES = 8
BATCH_SIZE = 100
N_LABELED_PER_BATCH = 40

# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def dataset():
    y_true_oracle, y_proxy, uncertainty, batches = generate_batched_binary_dataset_with_oracle_sampling(
        n_samples=[BATCH_SIZE] * N_BATCHES,
        true_mean=[0.3] * N_BATCHES,
        proxy_mean=[0.35] * N_BATCHES,
        correlation=[0.6] * N_BATCHES,
        random_seed=0,
    )
    samples = [
        ActiveSampler().sample(uncertainty[batches == batch_id], n_samples=N_LABELED_PER_BATCH, random_seed=batch_id)
        for batch_id in range(N_BATCHES)
    ]
    sampling_probability = np.hstack([sample[0] for sample in samples])
    xi = np.hstack([sample[1] for sample in samples])
    y_true = simulate_annotation(y_true_oracle, xi)
    return y_true, y_proxy, sampling_probability, batches


# ── tests ──────────────────────────────────────────────────────────────────────


def test_detect_batch_estimates_match_asi_estimator(dataset):
    """Each batch estimate equals the ASI estimator computed on that batch alone when power tuning is disabled."""
    y_true, y_proxy, sampling_probability, batches = dataset
    monitor_result = AsymptoticActivePPRM().detect(
        y_true, y_proxy, sampling_probability, batches, higher_is_better=False, threshold=0.5, power_tuning=False
    )
    estimator_means = np.zeros(N_BATCHES)
    for batch_id in range(N_BATCHES):
        batch_mask = batches == batch_id
        estimator_result = ASIMeanEstimator().estimate(
            y_true[batch_mask], y_proxy[batch_mask], sampling_probability[batch_mask], power_tuning=False
        )
        estimator_means[batch_id] = estimator_result.mean

    np.testing.assert_allclose(monitor_result.batch_mean_estimates, estimator_means)


def test_detect_equal_probabilities_match_asymptotic_pprm():
    """Equal sampling probabilities reduce ASI to PPI++, up to a small finite-sample gap between the two monitors."""
    n_batches = 3
    n_labeled, n_unlabeled = 100, 400
    batch_size = n_labeled + n_unlabeled
    y_true_oracle_and_proxy = [
        generate_gaussian_dataset(
            n_samples=batch_size, true_mean=3, true_std=0.1, proxy_mean=2, proxy_std=0.1, random_seed=batch_id
        )
        for batch_id in range(n_batches)
    ]
    y_true_oracle = np.hstack([pair[0] for pair in y_true_oracle_and_proxy])
    y_proxy = np.hstack([pair[1] for pair in y_true_oracle_and_proxy])
    xi = np.tile(np.hstack([np.ones(n_labeled), np.zeros(n_unlabeled)]), n_batches)
    y_true = simulate_annotation(y_true_oracle, xi)
    sampling_probability = np.full(n_batches * batch_size, n_labeled / batch_size)
    batches = np.repeat(np.arange(n_batches), batch_size)

    active_result = AsymptoticActivePPRM().detect(
        y_true, y_proxy, sampling_probability, batches, higher_is_better=False, threshold=3.5, power_tuning=False
    )
    ppi_result = AsymptoticPPRM().detect(
        y_true, y_proxy, batches, higher_is_better=False, threshold=3.5, power_tuning=False
    )

    np.testing.assert_allclose(active_result.batch_mean_estimates, ppi_result.batch_mean_estimates, atol=9e-3)


def test_detect_prefix_consistency(dataset):
    """Detecting on a growing history is prefix-consistent with detecting on the full history."""
    y_true, y_proxy, sampling_probability, batches = dataset
    monitor = AsymptoticActivePPRM()
    full = monitor.detect(
        y_true, y_proxy, sampling_probability, batches, higher_is_better=False, threshold=0.5, tightest_at_batch=2
    )
    prefix_mask = batches <= 2
    prefix = monitor.detect(
        y_true[prefix_mask],
        y_proxy[prefix_mask],
        sampling_probability[prefix_mask],
        batches[prefix_mask],
        higher_is_better=False,
        threshold=0.5,
        tightest_at_batch=2,
    )

    np.testing.assert_allclose(prefix.running_means, full.running_means[:3])
    np.testing.assert_allclose(prefix.confidence_bounds, full.confidence_bounds[:3])


def test_detect_higher_is_better_symmetry(dataset):
    """Monitoring a performance is the mirror image of monitoring its negation as a risk."""
    y_true, y_proxy, sampling_probability, batches = dataset
    risk = AsymptoticActivePPRM().detect(
        y_true, y_proxy, sampling_probability, batches, higher_is_better=False, threshold=0.3
    )
    performance = AsymptoticActivePPRM().detect(
        -y_true, -y_proxy, sampling_probability, batches, higher_is_better=True, threshold=-0.3
    )

    np.testing.assert_array_equal(performance.alarms, risk.alarms)
    np.testing.assert_allclose(performance.confidence_bounds, -risk.confidence_bounds)
