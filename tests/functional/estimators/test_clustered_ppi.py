import numpy as np
import pytest

from glide.estimators import ClusteredPPIMeanEstimator, PPIMeanEstimator


@pytest.mark.parametrize("power_tuning", [False, True])
def test_single_observation_clusters_equals_ppi(power_tuning):
    y_true = np.array([4.0, np.nan, 6.0, np.nan])
    y_proxy = np.array([2.0, 4.0, 6.0, 6.0])
    clusters = np.array(["A", "C", "B", "D"])

    cluster_result = ClusteredPPIMeanEstimator().estimate(y_true, y_proxy, clusters, power_tuning=power_tuning)
    ppi_result = PPIMeanEstimator().estimate(y_true, y_proxy, power_tuning=power_tuning)

    assert cluster_result.confidence_interval.mean == pytest.approx(ppi_result.confidence_interval.mean, abs=1e-10)
    assert cluster_result.confidence_interval.std == pytest.approx(ppi_result.confidence_interval.std, abs=1e-10)
    assert cluster_result.confidence_interval.lower_bound == pytest.approx(
        ppi_result.confidence_interval.lower_bound, abs=1e-10
    )
    assert cluster_result.confidence_interval.upper_bound == pytest.approx(
        ppi_result.confidence_interval.upper_bound, abs=1e-10
    )
