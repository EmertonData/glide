"""Functional tests for MultiPPIMeanEstimator.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

import numpy as np

from glide.estimators import MultiPPIMeanEstimator, PPIMeanEstimator
from glide.simulators import generate_binary_dataset, generate_multi_binary_dataset, simulate_annotation

# ── tests ──────────────────────────────────────────────────────────────────────


def test_single_proxy_equals_ppi():
    """MultiPPI with M=1 reproduces PPI++ exactly.

    When y_proxies has a single column, the optimal lambda vector reduces to
    the scalar PPI++ tuning parameter. The point estimate, standard error, and
    confidence interval bounds must match to floating-point precision.
    """
    y_true_oracle, y_proxy_1d = generate_binary_dataset(n_total=25, random_seed=0)
    xi = np.hstack([np.ones(5), np.zeros(20)])
    y_true = simulate_annotation(y_true_oracle, xi)
    y_proxies_2d = y_proxy_1d[:, np.newaxis]

    result_multi = MultiPPIMeanEstimator().estimate(y_true, y_proxies_2d)
    result_ppi = PPIMeanEstimator().estimate(y_true, y_proxy_1d)

    np.testing.assert_allclose(result_multi.confidence_interval.mean, result_ppi.confidence_interval.mean, atol=1e-12)
    np.testing.assert_allclose(result_multi.std, result_ppi.std, atol=1e-12)
    np.testing.assert_allclose(
        result_multi.confidence_interval.lower_bound, result_ppi.confidence_interval.lower_bound, atol=1e-12
    )
    np.testing.assert_allclose(
        result_multi.confidence_interval.upper_bound, result_ppi.confidence_interval.upper_bound, atol=1e-12
    )


def test_two_proxies_tighter_than_single_ppi():
    """MultiPPI with M=2 yields a tighter confidence interval than single PPI with either proxy alone.

    When two proxies carry independent information about the true label, the optimal
    lambda vector exploits both signals simultaneously, reducing variance below what
    any single proxy achieves.
    """
    y_true_oracle, y_proxies = generate_multi_binary_dataset(200, 0.6, [0.6, 0.65], [0.7, 0.7], random_seed=0)
    xi = np.hstack([np.ones(40), np.zeros(160)])
    y_true = simulate_annotation(y_true_oracle, xi)

    result_multi = MultiPPIMeanEstimator().estimate(y_true, y_proxies)
    result_ppi_0 = PPIMeanEstimator().estimate(y_true, y_proxies[:, 0])
    result_ppi_1 = PPIMeanEstimator().estimate(y_true, y_proxies[:, 1])

    assert result_multi.std < result_ppi_0.std
    assert result_multi.std < result_ppi_1.std
