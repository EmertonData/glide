import numpy as np


def compute_effective_sample_size(y_true: np.ndarray, effective_var: float) -> int:
    var_y_true = np.nanvar(y_true, ddof=1)
    effective_sample_size = int(var_y_true / effective_var)
    return effective_sample_size
