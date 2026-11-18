"""Functional tests for AsymptoticIPWClassicalRM.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

import numpy as np
import pytest

from glide.estimators import IPWClassicalMeanEstimator
from glide.monitors import AsymptoticClassicalRM, AsymptoticIPWClassicalRM
from glide.simulators import generate_binary_dataset

# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def y_true():
    y_true, _ = generate_binary_dataset(n_samples=20, true_mean=0.5, random_seed=0)
    return y_true


@pytest.fixture
def batches():
    return np.repeat(np.arange(5), 4)


@pytest.fixture
def sampling_probability():
    return np.tile(np.array([0.9, 0.8, 0.7, 0.6]), 5)


@pytest.fixture
def y(y_true, sampling_probability):
    labeled = np.random.default_rng(0).random(len(y_true)) < sampling_probability
    labeled[::4] = True
    labeled[1::4] = True
    y = np.where(labeled, y_true, np.nan)
    return y


# ── tests ──────────────────────────────────────────────────────────────────────


def test_detect_constant_probability_matches_classical_monitor():
    """With a constant probability labeling exactly that fraction of each batch, IPW and classical agree."""
    n_batches, batch_size, probability = 4, 10, 0.5
    y_full = np.random.default_rng(0).binomial(1, 0.4, size=n_batches * batch_size).astype(float)
    labeled = np.tile(np.arange(batch_size) < batch_size * probability, n_batches)
    y = np.where(labeled, y_full, np.nan)
    batches = np.repeat(np.arange(n_batches), batch_size)
    sampling_probability = np.full(n_batches * batch_size, probability)

    ipw_result = AsymptoticIPWClassicalRM().detect(
        y, sampling_probability, batches, higher_is_better=False, threshold=0.5
    )
    classical_result = AsymptoticClassicalRM().detect(y, batches, higher_is_better=False, threshold=0.5)

    np.testing.assert_allclose(ipw_result.batch_mean_estimates, classical_result.batch_mean_estimates)


def test_detect_batch_mean_estimates_match_ipw_classical_estimator(y, sampling_probability, batches):
    """Each batch estimate equals the IPW classical estimator computed on that batch alone."""
    result = AsymptoticIPWClassicalRM().detect(y, sampling_probability, batches, higher_is_better=False, threshold=0.5)
    n_batches = len(np.unique(batches))
    estimator_means = np.zeros(n_batches)
    for batch_id in range(n_batches):
        batch_mask = batches == batch_id
        estimator_result = IPWClassicalMeanEstimator().estimate(y[batch_mask], sampling_probability[batch_mask])
        estimator_means[batch_id] = estimator_result.mean

    np.testing.assert_allclose(result.batch_mean_estimates, estimator_means)


def test_detect_prefix_consistency(y, sampling_probability, batches):
    """Detecting on a growing history is prefix-consistent with detecting on the full history."""
    monitor = AsymptoticIPWClassicalRM()
    full = monitor.detect(y, sampling_probability, batches, higher_is_better=False, threshold=0.5, tightest_at_batch=2)
    prefix_mask = batches <= 2
    prefix = monitor.detect(
        y[prefix_mask],
        sampling_probability[prefix_mask],
        batches[prefix_mask],
        higher_is_better=False,
        threshold=0.5,
        tightest_at_batch=2,
    )

    np.testing.assert_allclose(prefix.running_means, full.running_means[:3])
    np.testing.assert_allclose(prefix.confidence_bounds, full.confidence_bounds[:3])


def test_detect_higher_is_better_symmetry(y, sampling_probability, batches):
    """Monitoring a performance is the mirror image of monitoring its complement as a risk."""
    risk = AsymptoticIPWClassicalRM().detect(y, sampling_probability, batches, higher_is_better=False, threshold=0.3)
    performance = AsymptoticIPWClassicalRM().detect(
        -y, sampling_probability, batches, higher_is_better=True, threshold=-0.3
    )

    np.testing.assert_array_equal(performance.alarms, risk.alarms)
    np.testing.assert_allclose(performance.confidence_bounds, -risk.confidence_bounds)
