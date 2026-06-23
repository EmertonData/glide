import numpy as np
import pytest

from glide.estimators import ClusteredPTDMeanEstimator, PTDMeanEstimator


@pytest.mark.parametrize("power_tuning", [False, True])
def test_single_observation_clusters_equals_ptd(power_tuning):
    y_true = np.array([5.0, 6.0, np.nan, np.nan])
    y_proxy = np.array([4.9, 6.1, 5.2, 6.1])
    clusters = np.array(["A", "B", "C", "D"])

    cluster_result = ClusteredPTDMeanEstimator().estimate(
        y_true, y_proxy, clusters, power_tuning=power_tuning, n_bootstrap=50, random_seed=0
    )
    ptd_result = PTDMeanEstimator().estimate(y_true, y_proxy, power_tuning=power_tuning, n_bootstrap=50, random_seed=0)

    assert cluster_result.confidence_interval.lower_bound == pytest.approx(
        ptd_result.confidence_interval.lower_bound, abs=1e-10
    )
    assert cluster_result.confidence_interval.upper_bound == pytest.approx(
        ptd_result.confidence_interval.upper_bound, abs=1e-10
    )
