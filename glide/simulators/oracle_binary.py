from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def generate_binary_dataset_with_oracle_sampling(
    N: int,
    true_mean: float = 0.7,
    proxy_mean: float = 0.6,
    correlation: float = 0.8,
    random_seed: Optional[int] = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Generate a synthetic binary dataset with oracle sampling probabilities.

    All N samples have ground-truth labels (y_true), proxy predictions (y_proxy),
    and an oracle root mean square error (RMSE) derived from the analytical
    proxy error. The RMSE values are non-uniform: samples where the proxy is less
    reliable receive higher RMSE following the optimal sampling rule.

    The sampling is based on a latent variable which determines the correlation
    between y_true and y_proxy in each sample. This variable is sampled uniformly
    around the given correlation value with limited spread within the interval of
    possible correlation levels given true_mean and proxy_mean. This way, the
    correlation between y_true and y_proxy matches the target value on average.

    Parameters
    ----------
    N : int
        Total number of samples.
    true_mean : float
        Expected mean of y_true. Must be in (0, 1).
    proxy_mean : float
        Expected mean of y_proxy. Must be in (0, 1).
    correlation : float
        Pearson correlation between y_true and y_proxy (marginal, across all samples).
    random_seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray]
        [0]: array of shape ``(n_samples,)``, y_true with the full ground-truth labels for all N
        samples (no NaN); the caller is responsible for masking unlabeled rows
        [1]: array of shape ``(n_samples,)``, y_proxy with proxy predictions
        [2]: array of shape ``(n_samples,)``, uncertainty with oracle RMSE per sample

    Raises
    ------
    ValueError
        If true_mean is not in (0, 1).
    ValueError
        If proxy_mean is not in (0, 1).
    ValueError
        If the combination of true_mean, proxy_mean, and correlation leads to
        negative joint probabilities.

    Notes
    -----
    **Step 1 — Global joint distribution**

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

    These four probabilities are fully determined by ``(p_t, p_p, correlation)``
    and must all be strictly positive — otherwise the parameter combination is
    impossible and a ``ValueError`` is raised. The previous probabilities become
    negative for the following respective values :

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

    **Step 2 — Latent variable x and per-sample correlation**

    Each sample receives a latent value ``x_i ~ Uniform(-1, 1)`` representing
    "annotation difficulty".  The per-sample Pearson correlation is defined as:

    ```
    corr(x_i) = correlation + correlation_spread * x_i
    ```

    Because ``E[x] = 0`` for ``x ~ Uniform(-1, 1)``, the marginal
    correlation ``E[corr(X)] = correlation`` exactly, preserving the target
    value on average.  Samples with low ``x`` have lower conditional
    correlation (proxy less reliable → higher oracle RMSE); samples with high
    ``x`` have higher conditional correlation (proxy more reliable → lower RMSE).

    ``correlation_spread`` is chosen as 90 % of the largest value that keeps
    all four per-sample probabilities strictly positive for every
    ``x in [-1, 1]``:

    ```
    max_safe_correlation_spread = min(p00, p01, p10, p11) / D
    ```

    **Step 3 — Per-sample probabilities and error probability**

    We adapt ``p11`` with ``x`` and this propagates to other values:

    ```
    p11(x) = corr(x) * D + p_t * p_p          # varies with x
    error_prob(x) = p01(x) + p10(x)
                    = p_t + p_p - 2 * p11(x)    # proxy ≠ y_true
    ```

    ``error_prob(x)`` is the per-sample proxy error probability, which
    decreases linearly as ``x`` increases (higher x → better proxy).

    **Step 4 — Vectorized CDF inversion**

    Since each sample has its own probability vector, ``numpy.random.choice``
    (which takes a single fixed probability vector) cannot be used.  Instead,
    the four outcomes ``(0,0), (0,1), (1,0), (1,1)`` are encoded as integers
    0–3 and sampled via cumulative-threshold comparison on a single
    ``u ~ Uniform(0,1)`` draw:

    ```
    u < p00(x)                 → outcome 0 : (y_true=0, y_proxy=0)
    u < p00(x)+p01(x)          → outcome 1 : (y_true=0, y_proxy=1)
    u < p00(x)+p01(x)+p10(x)   → outcome 2 : (y_true=1, y_proxy=0)
    else                       → outcome 3 : (y_true=1, y_proxy=1)
    ```

    The crucial simplification is that the second threshold collapses to the
    constant ``1 - p_t`` (independent of ``x``), because:

    ```
    p00(x) + p01(x) = (1-p_t-p_p+p11) + (p_p-p11) = 1 - p_t
    ```

    We also have :
    ```
    p00(x) + p01(x) + p10(x) = 1 - p11(x)
    ```

    This means only two of the three thresholds require per-sample arrays.
    The outcome integer encodes both labels: ``y_true = outcome // 2``,
    ``y_proxy = outcome % 2``.

    **Step 5 — Oracle RMSE**

    The optimal sampling probability satisfies
    ``RMSE = sqrt(E[(y_proxy - y_true)²]) = sqrt(error_prob(x))``.
    These values are stored directly as ``uncertainty``.

    Examples
    --------
    >>> from glide.simulators import generate_binary_dataset_with_oracle_sampling
    >>> y_true, y_proxy, uncertainty = generate_binary_dataset_with_oracle_sampling(N=4, random_seed=0)
    >>> len(y_true)
    4
    >>> len(y_proxy)
    4
    >>> len(uncertainty)
    4
    >>> bool(np.all(np.isin(y_true, [0.0, 1.0])))
    True
    >>> bool(np.all(np.isin(y_proxy, [0.0, 1.0])))
    True
    >>> bool(np.all((uncertainty >= 0) & (uncertainty <= 1)))
    True
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

    # Global (marginal) joint distribution — same as generate_binary_dataset
    p11 = correlation * D + p_t * p_p
    p00 = 1 - p_t - p_p + p11
    p01 = p_p - p11
    p10 = p_t - p11
    probs = [p00, p01, p10, p11]

    # Spread parameter: modulates the conditional correlation across samples
    max_safe_correlation_spread = min(probs) / D
    correlation_spread = 0.9 * max_safe_correlation_spread

    # Latent variable: controls per-sample proxy correlation
    x = rng.uniform(-1.0, 1.0, size=N)

    # Per-sample conditional joint distribution
    correlation_x = correlation + correlation_spread * x
    p11_x = correlation_x * D + p_t * p_p
    error_prob_x = p_t + p_p - 2.0 * p11_x

    # Vectorized CDF inversion to sample (y_true, y_proxy) per sample
    p00_x = 1.0 - p_t - p_p + p11_x
    u = rng.uniform(0.0, 1.0, size=N)
    samples = np.where(
        u < p00_x,
        0,
        np.where(
            u < 1.0 - p_t,
            1,
            np.where(u < 1.0 - p11_x, 2, 3),
        ),
    )
    y_true_arr = samples // 2
    y_proxy_arr = samples % 2

    # Oracle RMSE: sqrt(P(error | x_i))
    uncertainty = np.sqrt(error_prob_x)

    return y_true_arr.astype(float), y_proxy_arr.astype(float), uncertainty
