import numpy as np


def compute_effective_sample_size(y_true, estimated_std):
    n = len(y_true)
    var_y_true = np.var(y_true, ddof=1)
    std_labeled = np.sqrt(var_y_true / n)
    effective_sample_size = float(n * (std_labeled / estimated_std) ** 2)
    return effective_sample_size
