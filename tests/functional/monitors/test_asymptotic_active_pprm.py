"""Functional tests for AsymptoticActivePPRM.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

from typing import List, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

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

# ── helpers ────────────────────────────────────────────────────────────────────


def _build_active_stream(true_mean: List[float], proxy_mean: List[float]) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    y_true_oracle, y_proxy, uncertainty, batches = generate_batched_binary_dataset_with_oracle_sampling(
        n_samples=[BATCH_SIZE] * N_BATCHES,
        true_mean=true_mean,
        proxy_mean=proxy_mean,
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


# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def dataset():
    return _build_active_stream(true_mean=[0.3] * N_BATCHES, proxy_mean=[0.35] * N_BATCHES)


@pytest.fixture
def drifted_dataset():
    return _build_active_stream(
        true_mean=[0.3] * 2 + [0.7] * (N_BATCHES - 2), proxy_mean=[0.35] * 2 + [0.65] * (N_BATCHES - 2)
    )


@pytest.fixture
def constant_probability_dataset():
    n_labeled_per_batch = 25
    y_true_oracle_per_batch = []
    y_proxy_per_batch = []
    for batch_id in range(N_BATCHES):
        y_true_oracle, y_proxy = generate_gaussian_dataset(
            n_samples=BATCH_SIZE, true_mean=3, true_std=0.1, proxy_mean=2, proxy_std=0.1, random_seed=batch_id
        )
        y_true_oracle_per_batch.append(y_true_oracle)
        y_proxy_per_batch.append(y_proxy)
    xi = np.tile(np.hstack([np.ones(n_labeled_per_batch), np.zeros(BATCH_SIZE - n_labeled_per_batch)]), N_BATCHES)
    y_true = simulate_annotation(np.hstack(y_true_oracle_per_batch), xi)
    sampling_probability = np.full(N_BATCHES * BATCH_SIZE, n_labeled_per_batch / BATCH_SIZE)
    batches = np.repeat(np.arange(N_BATCHES), BATCH_SIZE)
    return y_true, np.hstack(y_proxy_per_batch), sampling_probability, batches


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


def test_detect_constant_probability_matches_asymptotic_pprm(constant_probability_dataset):
    """With a constant sampling probability, ASI reduces to PPI++ so both monitors return the same batch estimates."""
    y_true, y_proxy, sampling_probability, batches = constant_probability_dataset
    active_result = AsymptoticActivePPRM().detect(
        y_true, y_proxy, sampling_probability, batches, higher_is_better=False, threshold=3.5, power_tuning=False
    )
    ppi_result = AsymptoticPPRM().detect(
        y_true, y_proxy, batches, higher_is_better=False, threshold=3.5, power_tuning=False
    )

    np.testing.assert_allclose(active_result.batch_mean_estimates, ppi_result.batch_mean_estimates, atol=1e-2)


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


def test_detect_stationary_stream_below_threshold_does_not_alarm(dataset):
    """No false alarm is raised on a stationary stream whose true mean stays below the threshold."""
    y_true, y_proxy, sampling_probability, batches = dataset
    result = AsymptoticActivePPRM().detect(
        y_true, y_proxy, sampling_probability, batches, higher_is_better=False, threshold=0.5, tightest_at_batch=4
    )

    assert not result.drift_detected


def test_detect_injected_drift_alarms(drifted_dataset):
    """An alarm is raised once the true mean of the stream drifts above the threshold."""
    y_true, y_proxy, sampling_probability, batches = drifted_dataset
    result = AsymptoticActivePPRM().detect(
        y_true, y_proxy, sampling_probability, batches, higher_is_better=False, threshold=0.5, tightest_at_batch=4
    )

    assert result.drift_detected
