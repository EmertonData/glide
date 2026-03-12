import numpy as np

from glide.core.utils import compute_effective_sample_size


def test_compute_effective_sample_size_manual():
    y_true = np.array([5.0, 6.0, 7.0])
    std = 0.2
    effective_sample_size = compute_effective_sample_size(y_true, std)
    assert effective_sample_size == 24
