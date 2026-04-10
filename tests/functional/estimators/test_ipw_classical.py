"""Functional tests for IPWClassicalMeanEstimator.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

import numpy as np
import pytest

from glide.core.simulated_datasets import generate_gaussian_dataset
from glide.estimators.classical import ClassicalMeanEstimator
from glide.estimators.ipw_classical import IPWClassicalMeanEstimator

# ── tests ──────────────────────────────────────────────────────────────────────


def test_uniform_sampling_probability_matches_classical():
    """IPW with all sampling probabilities equal to 1 on fully observed data reduces to classical mean estimator.

    When every unit is sampled with probability 1 and no observations are
    missing, the IPW weights are all 1 and the estimator is equivalent to the
    classical sample mean. Both the point estimate and the standard error must
    therefore agree exactly with ClassicalMeanEstimator on the same data.
    """
    n_labeled = 40

    labeled_dataset, _ = generate_gaussian_dataset(n_labeled, 0, random_seed=0)
    y = np.array(labeled_dataset["y_true"])
    sampling_probability = np.ones(n_labeled)

    ipw_result = IPWClassicalMeanEstimator().estimate(y, sampling_probability)
    classical_result = ClassicalMeanEstimator().estimate(labeled_dataset, y_field="y_true")

    assert ipw_result.mean == pytest.approx(classical_result.mean, abs=1e-10)
    assert ipw_result.std == pytest.approx(classical_result.std, abs=1e-10)
    assert ipw_result.n == classical_result.n
