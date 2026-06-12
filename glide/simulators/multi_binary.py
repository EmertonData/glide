from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from glide.core.validation import _validate_bounds, _validate_equal_lengths


def generate_multi_binary_dataset(
    n_total: int,
    true_mean: float,
    proxy_means: List[float],
    correlations: List[float],
    random_seed: Optional[Union[int, np.random.SeedSequence]] = None,
) -> Tuple[NDArray, NDArray]:
    """Generate a synthetic binary oracle dataset with M proxy models.

    Parameters
    ----------
    n_total : int
        Total number of samples.
    true_mean : float
        Expected mean value of the true labels. Must be in (0, 1).
    proxy_means : List[float]
        Expected mean value of each proxy label. Length M determines the number of proxies.
        Each value must be in (0, 1).
    correlations : List[float]
        Pearson correlation between the true label and each proxy label. Length must equal
        ``len(proxy_means)``. Each value must yield a valid joint binary distribution
        given ``true_mean`` and the corresponding ``proxy_means`` entry.
    random_seed : int or np.random.SeedSequence, optional
        Seed for reproducibility.

    Returns
    -------
    Tuple[NDArray, NDArray]
        [0]: array of shape ``(n_total,)``, y_true containing ground-truth labels.
        [1]: array of shape ``(n_total, M)``, y_proxies where column m contains
             proxy labels with mean ``proxy_means[m]`` and correlation ``correlations[m]``
             with y_true.

    Raises
    ------
    ValueError
        If ``proxy_means`` and ``correlations`` have different lengths.
    ValueError
        If ``true_mean`` is not in (0, 1).
    ValueError
        If any ``proxy_means[m]`` is not in (0, 1).
    ValueError
        If the combination of ``true_mean``, ``proxy_means[m]``, and ``correlations[m]``
        is impossible (leads to negative joint probabilities).

    Notes
    -----
    **Step 1 — Generate y_true**

    Each sample's ground-truth label is drawn independently from a Bernoulli distribution:

    ```
    y_true_i ~ Bernoulli(true_mean)
    ```

    **Step 2 — Conditional probabilities for each proxy**

    For proxy m with marginal ``p_p = proxy_means[m]`` and correlation ``rho_m``, the
    bivariate binary joint distribution (see ``generate_binary_dataset``) gives:

    ```
    D_m  = sqrt(p_t * p_p * (1 - p_t) * (1 - p_p))
    p11_m = rho_m * D_m + p_t * p_p
    p01_m = p_p - p11_m
    ```

    where ``p_t = true_mean``. The conditional probabilities follow:

    ```
    P(proxy_m = 1 | y_true = 1) = p11_m / p_t
    P(proxy_m = 1 | y_true = 0) = p01_m / (1 - p_t)
    ```

    **Step 3 — Vectorized conditional generation**

    For each sample i and proxy m, draw independently:

    ```
    proxy_{m,i} | y_true_i ~ Bernoulli(P(proxy_m = 1 | y_true_i))
    ```

    All M proxies are generated in a single vectorised call over the ``(n_total, M)``
    matrix of conditional probabilities. The proxies are conditionally independent given
    y_true, so they have positive marginal correlation with each other only through the
    shared y_true.

    **Validation bounds**

    The same feasibility constraint as in ``generate_binary_dataset`` applies per proxy:

    ```
    max(-p_t * p_p_m, p_p_m + p_t - 1 - p_t * p_p_m) / D_m
        <= correlations[m]
        <= min(p_t * (1 - p_p_m), p_p_m * (1 - p_t)) / D_m
    ```

    Examples
    --------
    >>> import numpy as np
    >>> from glide.simulators import generate_multi_binary_dataset
    >>> y_true, y_proxies = generate_multi_binary_dataset(8, 0.7, [0.6, 0.5], [0.7, 0.6], random_seed=42)
    >>> len(y_true)
    8
    >>> y_proxies.shape
    (8, 2)
    >>> bool(np.all(np.isin(y_true, [0.0, 1.0])))
    True
    >>> bool(np.all(np.isin(y_proxies, [0.0, 1.0])))
    True
    """
    proxy_means_arr = np.array(proxy_means, dtype=float)
    correlations_arr = np.array(correlations, dtype=float)

    _validate_equal_lengths(proxy_means_arr, correlations_arr, names=["proxy_means", "correlations"])
    _validate_bounds(true_mean, "true_mean", lower=0, upper=1, left_inclusive=False, right_inclusive=False)

    for m, p_p in enumerate(proxy_means_arr):
        _validate_bounds(p_p, f"proxy_means[{m}]", lower=0, upper=1, left_inclusive=False, right_inclusive=False)

    p_t = true_mean
    D = np.sqrt(p_t * proxy_means_arr * (1 - p_t) * (1 - proxy_means_arr))
    min_correlations = np.maximum(-p_t * proxy_means_arr, proxy_means_arr + p_t - 1 - p_t * proxy_means_arr) / D
    max_correlations = np.minimum(p_t * (1 - proxy_means_arr), proxy_means_arr * (1 - p_t)) / D

    for m, (rho, lo, hi, p_p) in enumerate(zip(correlations_arr, min_correlations, max_correlations, proxy_means_arr)):
        if rho < lo or rho > hi:
            raise ValueError(
                f"proxy {m}: impossible combination of 'true_mean'={true_mean!r}, "
                f"'proxy_means[{m}]'={p_p!r}, and 'correlations[{m}]'={rho!r}: "
                f"leads to negative joint probabilities; "
                f"possible 'correlations[{m}]' values are in the range ({lo:.3f}, {hi:.3f})."
            )

    rng = np.random.default_rng(seed=random_seed)
    y_true = rng.binomial(1, p_t, size=n_total).astype(float)

    p11 = correlations_arr * D + p_t * proxy_means_arr
    p01 = proxy_means_arr - p11
    cond_prob_given_1 = p11 / p_t
    cond_prob_given_0 = p01 / (1 - p_t)

    is_true_one = y_true[:, np.newaxis].astype(bool)
    cond_probs = np.where(is_true_one, cond_prob_given_1, cond_prob_given_0)
    y_proxies = rng.binomial(1, cond_probs).astype(float)

    return y_true, y_proxies
