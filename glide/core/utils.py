import numpy as np


def compute_effective_sample_size(y_true: np.ndarray, effective_std: float) -> int:
    n = len(y_true)
    var_y_true = np.var(y_true, ddof=1)
    std_of_mean = np.sqrt(var_y_true / n)
    effective_sample_size = int(n * (std_of_mean / effective_std) ** 2)
    return effective_sample_size
