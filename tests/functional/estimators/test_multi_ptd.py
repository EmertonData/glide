"""Functional tests for MultiPTDMeanEstimator.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

import numpy as np
import pytest

from glide.estimators import MultiPTDMeanEstimator, PTDMeanEstimator
from glide.simulators import generate_multi_binary_dataset, simulate_annotation

# ── tests ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("power_tuning", [False, True])
def test_single_proxy_equals_ptd(power_tuning):
    """MultiPTD with M=1 reproduces PTD exactly.

    When y_proxies has a single column, the optimal lambda vector reduces to
    the scalar PTD tuning parameter (or to 1.0 when power_tuning is disabled).
    The confidence interval bounds must match to floating-point precision.
    """
    y_true = np.array([1.0, 2.0, np.nan, np.nan, np.nan])
    y_proxy = np.array([1.1, 1.9, 2.5, 3.0, 1.5])
    y_proxies = y_proxy.reshape(-1, 1)

    result_multi = MultiPTDMeanEstimator().estimate(
        y_true, y_proxies, power_tuning=power_tuning, n_bootstrap=500, random_seed=42
    )
    result_ptd = PTDMeanEstimator().estimate(
        y_true, y_proxy, power_tuning=power_tuning, n_bootstrap=500, random_seed=42
    )

    assert result_multi.confidence_interval.lower_bound == pytest.approx(
        result_ptd.confidence_interval.lower_bound, abs=1e-10
    )
    assert result_multi.confidence_interval.upper_bound == pytest.approx(
        result_ptd.confidence_interval.upper_bound, abs=1e-10
    )


def test_two_proxies_tighter_than_single_ptd():
    """MultiPTD with M=2 yields a tighter confidence interval than single PTD with either proxy alone.

    When two proxies carry independent information about the true label, the optimal
    lambda vector exploits both signals simultaneously, reducing variance below what
    any single proxy achieves.
    """
    y_true_oracle, y_proxies = generate_multi_binary_dataset(200, 0.6, [0.6, 0.65], [0.7, 0.7], random_seed=0)
    xi = np.hstack([np.ones(40), np.zeros(160)])
    y_true = simulate_annotation(y_true_oracle, xi)

    result_multi = MultiPTDMeanEstimator().estimate(y_true, y_proxies, n_bootstrap=500, random_seed=0)
    result_ptd_0 = PTDMeanEstimator().estimate(y_true, y_proxies[:, 0], n_bootstrap=500, random_seed=0)
    result_ptd_1 = PTDMeanEstimator().estimate(y_true, y_proxies[:, 1], n_bootstrap=500, random_seed=0)

    assert result_multi.std < result_ptd_0.std
    assert result_multi.std < result_ptd_1.std
