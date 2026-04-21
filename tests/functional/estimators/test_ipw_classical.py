"""Functional tests for IPWClassicalMeanEstimator.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

import numpy as np
import pytest

from glide.estimators.classical import ClassicalMeanEstimator
from glide.estimators.ipw_classical import IPWClassicalMeanEstimator
from glide.simulators import generate_gaussian_dataset

# ── tests ──────────────────────────────────────────────────────────────────────


def test_ipw_mean_matches_expected():
    """IPW estimator recovers the true mean under non-uniform subsampling.

    Observations are dropped with probability (1 - π_i), where π_i is drawn
    uniformly from [0.5, 1]. The IPW estimator should compensate for the
    non-uniform missingness and produce a point estimate close to the true
    population mean, and n must equal the number of non-missing observations.
    """
    n = 500
    true_mean = 3
    true_std = 0.1
    rng = np.random.default_rng(seed=1)

    y_true, _ = generate_gaussian_dataset(
        n_labeled=n,
        n_unlabeled=0,
        true_mean=true_mean,
        true_std=true_std,
        random_seed=0,
    )
    sampling_probability = np.clip(rng.random(n), 0.5, 1)
    y_true[rng.random(n) > sampling_probability] = np.nan

    result = IPWClassicalMeanEstimator().estimate(y_true, sampling_probability)

    assert result.mean == pytest.approx(true_mean, abs=0.2)
    assert result.std == pytest.approx(0.11, abs=0.01)
    assert result.n == np.sum(~np.isnan(y_true))


def test_uniform_sampling_probability_matches_classical():
    """IPW with all sampling probabilities equal to 1 on fully observed data
    reduces to classical mean estimator.

    When every unit is sampled with probability 1 and no observations are
    missing, the IPW weights are all 1 and the estimator is equivalent to the
    classical sample mean. Both the point estimate and the standard error must
    therefore agree exactly with ClassicalMeanEstimator on the same data.
    """
    n_labeled = 40

    y_true, _ = generate_gaussian_dataset(n_labeled=n_labeled, n_unlabeled=0, random_seed=0)
    sampling_probability = np.ones(n_labeled)

    ipw_result = IPWClassicalMeanEstimator().estimate(y_true, sampling_probability)
    classical_result = ClassicalMeanEstimator().estimate(y_true)

    assert ipw_result.mean == pytest.approx(classical_result.mean, abs=1e-10)
    assert ipw_result.std == pytest.approx(classical_result.std, abs=1e-10)
    assert ipw_result.n == classical_result.n
