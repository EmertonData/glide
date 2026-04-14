import numpy as np


def compute_effective_sample_size(y_true: np.ndarray, effective_var: float) -> int:
    """Compute effective sample size given the variance of y_true and effective variance.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values (may contain NaN).
    effective_var : float
        Effective variance of the point estimate.

    Returns
    -------
    int
        The effective sample size.
    """
    var_y_true = np.nanvar(y_true, ddof=1)
    effective_sample_size = int(var_y_true / effective_var)
    return effective_sample_size
