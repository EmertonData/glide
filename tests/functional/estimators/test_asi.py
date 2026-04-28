"""Functional tests for ASIMeanEstimator.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

import numpy as np
import pytest

from glide.estimators import ASIMeanEstimator, PPIMeanEstimator
from glide.simulators import generate_gaussian_dataset

# ── tests ──────────────────────────────────────────────────────────────────────


def test_equal_probabilities_match_simple_ppi():
    """When all π_i are equal, IPW weights are such that the estimator reduces
    to PPIMeanEstimator.

    All sampling probabilities are set to the same constant value
    π = n_labeled / (n_labeled + n_unlabeled). Because the IPW correction 1/π_i is
    identical for every observation, it factors out, leaving an estimate identical
    to the plain PPIMeanEstimator on the same data.
    """
    n_labeled, n_unlabeled = 100, 400
    pi_value = n_labeled / (n_labeled + n_unlabeled)
    true_mean = 3
    true_std = 0.1
    proxy_mean = 2
    proxy_std = 0.1
    random_seed = 0

    y_true, y_proxy = generate_gaussian_dataset(
        n_labeled=n_labeled,
        n_unlabeled=n_unlabeled,
        true_mean=true_mean,
        true_std=true_std,
        proxy_mean=proxy_mean,
        proxy_std=proxy_std,
        random_seed=random_seed,
    )
    pi = pi_value * np.ones(n_labeled + n_unlabeled)

    asi_result = ASIMeanEstimator().estimate(y_true, y_proxy, pi)
    ppi_result = PPIMeanEstimator().estimate(y_true, y_proxy)

    assert asi_result.mean == pytest.approx(ppi_result.mean, abs=0.01)
    assert asi_result.std == pytest.approx(ppi_result.std, abs=0.01)
