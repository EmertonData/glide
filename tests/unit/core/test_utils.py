import numpy as np
import pytest
from glide.core.utils import compute_effective_sample_size
from glide.estimators.ppi import PPIMeanEstimator


@pytest.fixture
def estimator() -> PPIMeanEstimator:
    return PPIMeanEstimator()


# --- ess ---


def test_compute_effective_sample_size_manual(estimator):
    y_true = np.array([5.0, 6.0, 7.0])
    y_proxy_labeled = np.array([4.5, 5.5, 6.5])
    y_proxy_unlabeled = np.array([4.0, 5.0, 6.0, 7.0])
    std = estimator._ppi_std((y_true, y_proxy_labeled, y_proxy_unlabeled))
    ess = compute_effective_sample_size(y_true, std)
    assert ess == 2.0
