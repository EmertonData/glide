from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from glide.core.validation import _validate_bounds


def generate_binary_dataset(
    n_total: int,
    true_mean: float = 0.7,
    proxy_mean: float = 0.6,
    correlation: float = 0.8,
    random_seed: Optional[Union[int, np.random.SeedSequence]] = None,
) -> Tuple[NDArray, NDArray]:
    """Generate a synthetic binary-label oracle dataset.

    Parameters
    ----------
    n_total : int
        Total number of samples. All samples have both true and proxy labels.
    true_mean : float
        Expected mean value of the true labels.
    proxy_mean : float
        Expected mean value of the proxy labels.
    correlation : float
        Pearson correlation between true and proxy labels.
    random_seed : int or np.random.SeedSequence, optional
        Seed for reproducibility.

    Returns
    -------
    Tuple[NDArray, NDArray]
        [0]: array of shape ``(n_total,)``, y_true containing ground-truth labels.
        [1]: array of shape ``(n_total,)``, y_proxy containing proxy labels.

    Raises
    ------
    ValueError
        If ``true_mean`` is not in (0, 1).
    ValueError
        If ``proxy_mean`` is not in (0, 1).
    ValueError
        If the combination of ``true_mean``, ``proxy_mean``, and ``correlation`` is
        impossible (leads to negative joint probabilities).

    Notes
    -----
    **Step 1 — Joint distribution**

    For two binary variables with marginals ``p_t = P(y_true=1)`` and
    ``p_p = P(y_proxy=1)``, the Pearson correlation uniquely determines the
    joint distribution.  Let ``D = sqrt(p_t * p_p * (1-p_t) * (1-p_p))``
    (product of standard deviations).  Then:

    ```
    p11 = P(y_true=1, y_proxy=1) = correlation * D + p_t * p_p
    p00 = P(y_true=0, y_proxy=0) = 1 - p_t - p_p + p11
    p01 = P(y_true=0, y_proxy=1) = p_p - p11
    p10 = P(y_true=1, y_proxy=0) = p_t - p11
    ```

    These four probabilities must all be strictly positive — otherwise the
    parameter combination is impossible and a ``ValueError`` is raised. The
    previous probabilities become negative for the following respective
    values :

    ```
    p11 < 0 for correlation < -(p_t * p_p) / D
    p00 < 0 for correlation < (p_t + p_p - p_t * p_p - 1) / D
    p01 < 0 for correlation > p_p * (1 - p_t) / D
    p10 < 0 for correlation > p_t * (1 - p_p) / D
    ```

    Therefore, the correlation needs to satisfy :

    ```
    max(-p_t * p_p, p_t + p_p - p_t * p_p - 1) <= correlation * D <= min(p_t * (1 - p_p), p_p * (1 - p_t))
    ```

    **Step 2 — Conditional probabilities**

    The joint probability ``p11`` determines the conditional probability of
    ``y_proxy`` given each value of ``y_true``:

    ```
    p11 = correlation * D + p_t * p_p
    p01 = p_p - p11

    P(y_proxy = 1 | y_true = 1) = p11 / p_t
    P(y_proxy = 1 | y_true = 0) = p01 / (1 - p_t)
    ```

    **Step 3 — Two-stage generation**

    ``y_true`` is sampled first for all ``n_total`` observations from a
    ``Bernoulli(p_t)`` distribution.  For each observation, the corresponding
    conditional probability from Step 2 is selected, and ``y_proxy`` is then
    drawn from that conditional Bernoulli distribution:

    ```
    y_true_i ~ Bernoulli(p_t)
    y_proxy_i | y_true_i ~ Bernoulli(P(y_proxy = 1 | y_true_i))
    ```

    References
    ----------
    .. [SO] `Correlation between Bernoulli Variables <https://math.stackexchange.com/questions/610443/finding-a-correlation-between-bernoulli-variables>`_

    Examples
    --------
    >>> import numpy as np
    >>> from glide.simulators import generate_binary_dataset
    >>> y_true, y_proxy = generate_binary_dataset(n_total=8, random_seed=42)
    >>> len(y_true)
    8
    >>> len(y_proxy)
    8
    >>> bool(np.all(np.isin(y_true, [0.0, 1.0])))
    True
    >>> bool(np.all(np.isin(y_proxy, [0.0, 1.0])))
    True
    """
    _validate_bounds(true_mean, "true_mean", lower=0, upper=1, left_inclusive=False, right_inclusive=False)
    _validate_bounds(proxy_mean, "proxy_mean", lower=0, upper=1, left_inclusive=False, right_inclusive=False)

    p_t = true_mean
    p_p = proxy_mean

    D = np.sqrt(p_t * p_p * (1 - p_t) * (1 - p_p))

    min_possible_correlation = max(-p_t * p_p, p_p + p_t - 1 - p_t * p_p) / D
    max_possible_correlation = min(p_t * (1 - p_p), p_p * (1 - p_t)) / D
    if correlation < min_possible_correlation or correlation > max_possible_correlation:
        raise ValueError(
            f"Impossible combination of 'true_mean'={true_mean!r}, 'proxy_mean'={proxy_mean!r}, "
            f"and 'correlation'={correlation!r}: leads to negative joint probabilities; "
            f"possible 'correlation' values are in the range ({min_possible_correlation:.3f}"
            f", {max_possible_correlation:.3f})."
        )

    rng = np.random.default_rng(seed=random_seed)
    y_true = rng.binomial(1, p_t, size=n_total).astype(float)

    p11 = correlation * D + p_t * p_p
    p01 = p_p - p11
    cond_prob_given_1 = p11 / p_t
    cond_prob_given_0 = p01 / (1 - p_t)

    cond_probs = np.where(y_true.astype(bool), cond_prob_given_1, cond_prob_given_0)
    y_proxy = rng.binomial(1, cond_probs).astype(float)

    return y_true, y_proxy
