"""Functional tests for ClusterClassicalMeanEstimator.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

import numpy as np
import pytest

from glide.estimators import ClassicalMeanEstimator, ClusterClassicalMeanEstimator

# ── tests ──────────────────────────────────────────────────────────────────────


def test_single_observation_clusters_equals_classical():
    """Single-observation clusters produce identical results to ClassicalMeanEstimator.

    When every cluster contains exactly one observation, cluster sums equal y,
    and Var(theta) = K * Var(y, ddof=1) / K^2 = Var(y, ddof=1) / K, which is
    exactly what ClassicalMeanEstimator computes (std(y, ddof=1) / sqrt(K)).
    """
    y = np.array([5.0, 7.0, 4.0, 8.0, 6.0, 3.0, 9.0, 2.0])
    clusters = np.array(["A", "B", "C", "D", "E", "F", "G", "H"])

    cluster_result = ClusterClassicalMeanEstimator().estimate(y, clusters)
    classical_result = ClassicalMeanEstimator().estimate(y)

    assert cluster_result.confidence_interval.lower_bound == pytest.approx(
        classical_result.confidence_interval.lower_bound, abs=1e-10
    )
    assert cluster_result.confidence_interval.upper_bound == pytest.approx(
        classical_result.confidence_interval.upper_bound, abs=1e-10
    )


def test_equal_size_clusters_equals_classical():
    """Equal-size clusters with equal cluster means equal the unweighted mean of cluster means.

    When all clusters have the same size s, the cluster estimator simplifies to
    Var(mu_k, ddof=1) / K, which is the standard mean-of-means estimator.
    With single observations per cluster this is identical to ClassicalMeanEstimator.
    With two observations per cluster the point estimate still equals the grand mean,
    but the standard error reflects K effective sampling units rather than 2K.
    """
    rng = np.random.default_rng(42)
    n_clusters = 6
    cluster_size = 2

    y_blocks = [rng.normal(loc=5.0, scale=0.1, size=cluster_size) for _ in range(n_clusters)]
    y = np.hstack(y_blocks)
    clusters = np.repeat(np.arange(n_clusters), cluster_size)

    cluster_result = ClusterClassicalMeanEstimator().estimate(y, clusters)
    cluster_means = np.array([block.mean() for block in y_blocks])
    expected_mean = cluster_means.mean()
    expected_std = cluster_means.std(ddof=1) / np.sqrt(n_clusters)

    assert cluster_result.confidence_interval.mean == pytest.approx(expected_mean, abs=1e-10)
    assert cluster_result.std == pytest.approx(expected_std, abs=1e-10)
