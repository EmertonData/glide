from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray


def generate_binary_dataset(
    n_labeled: int,
    n_unlabeled: int,
    true_mean: float = 0.7,
    proxy_mean: float = 0.6,
    correlation: float = 0.8,
    random_seed: Optional[Union[int, np.random.SeedSequence]] = None,
) -> Tuple[NDArray, NDArray]:
    """Generate a synthetic binary-label dataset for evaluation.

    Parameters
    ----------
    n_labeled : int
        Number of samples with both true and proxy labels (the labeled subset).
    n_unlabeled : int
        Number of samples with proxy labels only (the unlabeled subset).
    true_mean : float
        Expected mean value of the true labels.
    proxy_mean : float
        Expected mean value of the proxy labels.
    correlation : float
        Pearson correlation between true and proxy on the labeled subset.
    random_seed : int or np.random.SeedSequence, optional
        Seed for reproducibility.

    Returns
    -------
    Tuple[NDArray, NDArray]
        [0]: array of shape ``(n_labeled+n_unlabeled,)``, y_true with labeled values and NaN for unlabeled rows
        [1]: array of shape ``(n_labeled+n_unlabeled,)``, y_proxy with all values present

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

    **Step 2 — Sampling outcome pairs**

    The four outcomes ``(y_true=0, y_proxy=0)``, ``(y_true=0, y_proxy=1)``,
    ``(y_true=1, y_proxy=0)``, ``(y_true=1, y_proxy=1)`` are encoded as
    integers 0–3 with probabilities ``[p00, p01, p10, p11]``.  The ``n_labeled``
    labeled pairs are drawn in one call via ``numpy.random.Generator.choice``.

    **Step 3 — Decoding labels from integers**

    The integer encoding satisfies ``y_true = outcome // 2`` and
    ``y_proxy = outcome % 2``, so both labels are recovered with cheap
    integer arithmetic.

    **Step 4 — Unlabeled proxy sampling**

    The ``n_unlabeled`` unlabeled samples have only ``y_proxy`` values, sampled
    independently from ``Bernoulli(p_p)`` (marginal proxy distribution), with
    no dependence on ``y_true``.

    References
    ----------
    .. [SO] `Correlation between Bernoulli Variables <https://math.stackexchange.com/questions/610443/finding-a-correlation-between-bernoulli-variables>`_

    Examples
    --------
    >>> import numpy as np
    >>> from glide.simulators import generate_binary_dataset
    >>> y_true, y_proxy = generate_binary_dataset(n_labeled=100, n_unlabeled=500, random_seed=42)
    >>> len(y_true)
    600
    >>> len(y_proxy)
    600
    >>> int(np.sum(~np.isnan(y_true)))
    100
    >>> int(np.sum(~np.isnan(y_proxy)))
    600
    """
    if not (0 < true_mean < 1):
        raise ValueError(f"true_mean must be in (0, 1), got {true_mean}")
    if not (0 < proxy_mean < 1):
        raise ValueError(f"proxy_mean must be in (0, 1), got {proxy_mean}")

    rng = np.random.default_rng(seed=random_seed)

    p_t = true_mean
    p_p = proxy_mean

    # std product of the variable pair will be used multiple times
    D = np.sqrt(p_t * p_p * (1 - p_t) * (1 - p_p))

    # some combinations of true_mean, proxy_mean and correlation are impossible
    # and lead to negative probabilities, raise an error if this is the case
    min_possible_correlation = max(-p_t * p_p, p_p + p_t - 1 - p_t * p_p) / D
    max_possible_correlation = min(p_t * (1 - p_p), p_p * (1 - p_t)) / D
    if correlation < min_possible_correlation or correlation > max_possible_correlation:
        raise ValueError(
            f"Impossible combination of true_mean={true_mean}, proxy_mean={proxy_mean}, "
            f"and correlation={correlation}: leads to negative joint probabilities\n"
            f"possible correlation values are in the range ({min_possible_correlation:.3f}"
            f", {max_possible_correlation:.3f})"
        )

    # we will generate pairs values (true, proxy) with true and proxy equal to 0 or 1
    # probability of outcome (1, 1)
    p11 = correlation * D + p_t * p_p
    p00 = 1 - p_t - p_p + p11
    p01 = p_p - p11
    p10 = p_t - p11
    # probabilities of outcomes (0, 0), (0, 1), (1, 0), (1, 1)
    probs = [p00, p01, p10, p11]

    # generate the outcome pairs as integers between 0 and 3 inclusive
    samples = rng.choice(4, p=probs, size=n_labeled)
    # extract the true and proxy values via integer division and modulo 2
    # we have 0 = (0, 0), 1 = (0, 1), 2 = (1, 0), 3 = (1, 1)
    y_true_labeled = samples // 2
    y_proxy_labeled = samples % 2

    # generate proxy values for unlabeled samples
    y_proxy_unlabeled = rng.choice(2, p=[1 - p_p, p_p], size=n_unlabeled)

    # Combine labeled and unlabeled: NaN for unlabeled y_true, all y_proxy values
    y_true = np.hstack([y_true_labeled.astype(float), np.full(n_unlabeled, np.nan)])
    y_proxy = np.hstack([y_proxy_labeled.astype(float), y_proxy_unlabeled.astype(float)])

    return y_true, y_proxy
