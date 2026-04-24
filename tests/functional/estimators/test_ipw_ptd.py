"""Functional tests for IPWPTDMeanEstimator.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

import numpy as np
import pytest

from glide.estimators.asi import ASIMeanEstimator
from glide.estimators.ipw_ptd import IPWPTDMeanEstimator
from glide.estimators.ptd import PTDMeanEstimator
from glide.simulators import generate_gaussian_dataset

# ── tests ──────────────────────────────────────────────────────────────────────


def test_deterministic_probabilities_match_simple_ptd():
    """When all π_i are deterministic, IPW weights induce the same result as in PTDMeanEstimator."""
    n_labeled, n_unlabeled = 100, 400
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
        random_seed=0,
    )
    pi = (~np.isnan(y_true)).astype(float)

    ipw_ptd_result = IPWPTDMeanEstimator().estimate(y_true, y_proxy, pi, random_seed=random_seed)
    simple_ptd_result = PTDMeanEstimator().estimate(y_true, y_proxy)

    assert ipw_ptd_result.mean == pytest.approx(simple_ptd_result.mean, abs=0.01)
    assert ipw_ptd_result.std == pytest.approx(simple_ptd_result.std, abs=0.01)


def test_equal_probabilities_match_simple_ptd():
    """When all π_i are equal, IPW weights cancel and the estimator reduces to PTDMeanEstimator.

    All sampling probabilities are set to the same constant value π. Because
    the IPW correction 1/π_i is identical for every observation, it factors
    out and cancels, leaving an estimate identical to the unweighted
    PTDMeanEstimator on the same data.
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
        random_seed=0,
    )
    pi = pi_value * np.ones(n_labeled + n_unlabeled)

    ipw_ptd_result = IPWPTDMeanEstimator().estimate(y_true, y_proxy, pi, random_seed=random_seed)
    simple_ptd_result = PTDMeanEstimator().estimate(y_true, y_proxy)

    assert ipw_ptd_result.mean == pytest.approx(simple_ptd_result.mean, abs=0.02)


def test_large_sample_matches_asi():
    """When n is large, IPWPTDMeanEstimator produces inference equivalent to ASIMeanEstimator.

    With a large sample and non-uniform sampling probabilities, the IPW-PTD
    point estimate converges to the ASI point estimate and their confidence
    intervals agree asymptotically. The mean, standard deviation and confidence
    interval bounds must therefore be close for a sufficiently large sample.
    """
    n_samples = 1000
    true_mean = 3
    true_std = 0.1
    proxy_mean = 0.6
    proxy_std = 1
    random_seed = 0
    rng = np.random.default_rng(seed=random_seed)

    y_true, y_proxy = generate_gaussian_dataset(
        n_labeled=n_samples,
        n_unlabeled=0,
        true_mean=true_mean,
        true_std=true_std,
        proxy_mean=proxy_mean,
        proxy_std=proxy_std,
        random_seed=random_seed,
    )
    pi = np.clip(rng.random(n_samples), 0.5, 1)
    y_true[rng.random(n_samples) > pi] = np.nan

    ipw_ptd_result = IPWPTDMeanEstimator().estimate(y_true, y_proxy, pi, random_seed=random_seed)
    asi_result = ASIMeanEstimator().estimate(y_true, y_proxy, pi)

    asi_lower_bound = asi_result.confidence_interval.lower_bound
    ipw_ptd_lower_bound = ipw_ptd_result.confidence_interval.lower_bound
    asi_upper_bound = asi_result.confidence_interval.upper_bound
    ipw_ptd_upper_bound = ipw_ptd_result.confidence_interval.upper_bound

    assert ipw_ptd_result.mean == pytest.approx(asi_result.mean, abs=0.02)
    assert ipw_ptd_result.std == pytest.approx(asi_result.std, abs=0.02)
    assert asi_lower_bound == pytest.approx(ipw_ptd_lower_bound, abs=0.02)
    assert asi_upper_bound == pytest.approx(ipw_ptd_upper_bound, abs=0.04)
