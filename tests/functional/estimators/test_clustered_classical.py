"""Functional tests for ClusteredClassicalMeanEstimator.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

import numpy as np
import pytest

from glide.estimators import ClassicalMeanEstimator, ClusteredClassicalMeanEstimator


def test_single_observation_clusters_equals_classical():
    # Single-observation clusters produce identical results to ClassicalMeanEstimator.

    y = np.array([5.0, 7.0, 4.0, 8.0, 6.0, 3.0, 9.0, 2.0])
    clusters = np.array(["A", "B", "C", "D", "E", "F", "G", "H"])

    cluster_result = ClusteredClassicalMeanEstimator().estimate(y, clusters)
    classical_result = ClassicalMeanEstimator().estimate(y)

    assert cluster_result.confidence_interval.mean == pytest.approx(
        classical_result.confidence_interval.mean, abs=1e-10
    )
    assert cluster_result.confidence_interval.std == pytest.approx(classical_result.confidence_interval.std, abs=1e-10)
