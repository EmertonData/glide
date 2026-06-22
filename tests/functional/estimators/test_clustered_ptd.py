import numpy as np
import pytest

from glide.estimators import ClusteredPTDMeanEstimator, PTDMeanEstimator


def test_point_estimate_lambda_one_with_identical_clusters():
    # Labeled clusters have constant y_true=5.0 and y_proxy=4.8.
    # Unlabeled clusters have slightly different proxy values so var_proxy_unlabeled > 0.
    # mean_proxy_unlabeled = (2*5.2 + 2*5.4) / 4 = 5.3, so PTD mean = 5.0 + (5.3 - 4.8) = 5.5.
    # With n_bootstrap=2000 the bootstrap mean converges to 5.5 within tolerance.
    y_true = np.array([5.0, 5.0, 5.0, 5.0, np.nan, np.nan, np.nan, np.nan])
    y_proxy = np.array([4.8, 4.8, 4.8, 4.8, 5.2, 5.2, 5.4, 5.4])
    clusters = np.array(["A", "A", "B", "B", "C", "C", "D", "D"])

    result = ClusteredPTDMeanEstimator().estimate(
        y_true, y_proxy, clusters, power_tuning=False, n_bootstrap=2000, random_seed=0
    )

    assert result.confidence_interval.mean == pytest.approx(5.5, abs=0.02)


def test_single_observation_clusters_equals_ptd():
    y_true = np.array([5.0, 6.0, np.nan, np.nan])
    y_proxy = np.array([4.9, 6.1, 5.2, 6.1])
    clusters = np.array(["A", "B", "C", "D"])

    cluster_result = ClusteredPTDMeanEstimator().estimate(
        y_true, y_proxy, clusters, power_tuning=False, n_bootstrap=50, random_seed=0
    )
    ptd_result = PTDMeanEstimator().estimate(y_true, y_proxy, power_tuning=False, n_bootstrap=50, random_seed=0)

    assert cluster_result.confidence_interval.lower_bound == pytest.approx(
        ptd_result.confidence_interval.lower_bound, abs=1e-10
    )
    assert cluster_result.confidence_interval.upper_bound == pytest.approx(
        ptd_result.confidence_interval.upper_bound, abs=1e-10
    )
