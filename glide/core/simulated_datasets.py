from typing import List, Optional, Tuple, Union

import numpy as np

from glide.core.dataset import Dataset


def generate_binary_dataset(
    n: int,
    N: int,
    true_mean: float = 0.7,
    proxy_mean: float = 0.6,
    correlation: float = 0.8,
    random_seed: Optional[Union[int, np.random.SeedSequence]] = None,
) -> Tuple[Dataset, Dataset]:
    """Generate a synthetic binary-label dataset for evaluation.

    Parameters
    ----------
    n : int
        Number of records with both true and proxy labels (the labeled subset).
    N : int
        Number of records with proxy labels only (the unlabeled subset).
    true_mean : float
        Expected mean value of the true labels.
    proxy_mean : float
        Expected mean value of the proxy labels.
    correlation : float
        Pearson correlation between true and proxy on the labeled subset.
    random_seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    Tuple[Dataset, Dataset]
        [0]: labeled dataset with ``n`` records, containing ``"y_true"`` et ``"y_proxy"`` fields
        [1]: unlabeld dataset with ``N`` records, containing only `"y_proxy"`` field

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
    integers 0–3 with probabilities ``[p00, p01, p10, p11]``.  The ``n``
    labeled pairs are drawn in one call via ``numpy.random.Generator.choice``.

    **Step 3 — Decoding labels from integers**

    The integer encoding satisfies ``y_true = outcome // 2`` and
    ``y_proxy = outcome % 2``, so both labels are recovered with cheap
    integer arithmetic.

    **Step 4 — Unlabeled proxy sampling**

    The ``N`` unlabeled records contain only the ``"y_proxy"`` field, sampled
    independently from ``Bernoulli(p_p)`` (marginal proxy distribution), with
    no dependence on ``y_true``.

    References
    ----------
    .. [SO] `Correlation between Bernoulli Variables <https://math.stackexchange.com/questions/610443/finding-a-correlation-between-bernoulli-variables>`_

    Examples
    --------
    >>> from glide.core.simulated_datasets import generate_binary_dataset
    >>> labeled, unlabeled = generate_binary_dataset(n=100, N=500, random_seed=42)
    >>> len(labeled)
    100
    >>> len(unlabeled)
    500
    >>> list(labeled.records[0].keys())
    ['y_true', 'y_proxy']
    >>> list(unlabeled.records[0].keys())
    ['y_proxy']

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
    samples = rng.choice(4, p=probs, size=n)
    # extract the true and proxy values via integer division and modulo 2
    # we have 0 = (0, 0), 1 = (0, 1), 2 = (1, 0), 3 = (1, 1)
    y_true = samples // 2
    y_proxy_labeled = samples % 2

    # generate proxy values for un labeled samples
    y_proxy_unlabeled = rng.choice(2, p=[1 - p_p, p_p], size=N)

    labeled_records = []
    unlabeled_records = []
    for y_true_, y_proxy_ in zip(y_true, y_proxy_labeled):
        labeled_records.append({"y_true": int(y_true_), "y_proxy": int(y_proxy_)})
    for y_proxy_ in y_proxy_unlabeled:
        unlabeled_records.append({"y_proxy": int(y_proxy_)})

    return Dataset(labeled_records), Dataset(unlabeled_records)


def generate_stratified_binary_dataset(
    n: List[int],
    N: List[int],
    true_mean: List[float],
    proxy_mean: List[float],
    correlation: List[float],
    random_seed: Optional[int] = None,
) -> Tuple[Dataset, Dataset]:
    """Generate a synthetic stratified binary-label dataset for evaluation.

    Generate multiple strata with potentially different parameters (true_mean, proxy_mean,
    correlation, n, N per stratum). This enables simulation of heterogeneous data where
    different groups have different proxy-truth relationships.

    Parameters
    ----------
    n : List[int]
        Number of records with both true and proxy labels per stratum.
        Length must equal number of strata.
    N : List[int]
        Number of records with proxy labels only per stratum.
        Length must equal number of strata.
    true_mean : List[float]
        Expected mean value of the true labels per stratum.
        Length must equal number of strata.
    proxy_mean : List[float]
        Expected mean value of the proxy labels per stratum.
        Length must equal number of strata.
    correlation : List[float]
        Pearson correlation between true and proxy per stratum.
        Length must equal number of strata.
    random_seed : int, optional
        Seed for reproducibility. If provided, seeds are derived deterministically.

    Returns
    -------
    Tuple[Dataset, Dataset]
        [0]: labeled dataset with all strata combined, containing ``"y_true"``, ``"y_proxy"``,
             and ``"stratum_id"`` fields
        [1]: unlabeled dataset with all strata combined, containing ``"y_proxy"`` and
             ``"stratum_id"`` fields

    Raises
    ------
    ValueError
        If input lists have different lengths.
    ValueError
        If fewer than 1 stratum is specified.
    ValueError
        If any stratum has invalid parameters (see generate_binary_dataset).

    Examples
    --------
    >>> from glide.core.simulated_datasets import generate_stratified_binary_dataset
    >>> labeled, unlabeled = generate_stratified_binary_dataset(
    ...     n=[50, 100],
    ...     N=[200, 300],
    ...     true_mean=[0.6, 0.8],
    ...     proxy_mean=[0.5, 0.7],
    ...     correlation=[0.7, 0.75],
    ...     random_seed=42
    ... )
    >>> len(labeled)
    150
    >>> len(unlabeled)
    500
    >>> list(labeled.records[0].keys())
    ['y_true', 'y_proxy', 'stratum_id']
    >>> labeled.records[0]["stratum_id"]
    0
    """
    num_strata = len(n)
    if num_strata < 1:
        raise ValueError(f"Number of strata must be at least 1, got {num_strata}")

    # Validate all lists have the same length
    param_lengths = {
        "n": len(n),
        "N": len(N),
        "true_mean": len(true_mean),
        "proxy_mean": len(proxy_mean),
        "correlation": len(correlation),
    }
    if not all(length == num_strata for length in param_lengths.values()):
        lengths_str = ", ".join(f"{name}={length}" for name, length in param_lengths.items())
        raise ValueError(f"All input lists must have the same length. Got: {lengths_str}")

    # Generate data for each stratum
    all_labeled_records = []
    all_unlabeled_records = []

    seed_sequence = np.random.SeedSequence(random_seed)
    child_sequences = seed_sequence.spawn(num_strata)

    for stratum_id in range(num_strata):
        # Generate data for this stratum
        labeled, unlabeled = generate_binary_dataset(
            n=n[stratum_id],
            N=N[stratum_id],
            true_mean=true_mean[stratum_id],
            proxy_mean=proxy_mean[stratum_id],
            correlation=correlation[stratum_id],
            random_seed=child_sequences[stratum_id],
        )

        # Add stratum_id to all records
        for record in labeled:
            record["stratum_id"] = stratum_id
            all_labeled_records.append(record)

        for record in unlabeled.records:
            record["stratum_id"] = stratum_id
            all_unlabeled_records.append(record)

    return Dataset(all_labeled_records), Dataset(all_unlabeled_records)


def generate_binary_dataset_with_oracle_sampling(
    N: int,
    true_mean: float = 0.7,
    proxy_mean: float = 0.6,
    correlation: float = 0.8,
    random_seed: Optional[int] = None,
) -> Dataset:
    """Generate a synthetic binary dataset with oracle sampling probabilities.

    All N records have ground-truth labels (y_true), proxy predictions (y_proxy),
    and an oracle root mean square error (RMSE) derived from the analytical
    proxy error. The RMSE values are non-uniform: records where the proxy is less
    reliable receive higher RMSE following the optimal sampling rule.

    The sampling is based on a latent variable which determines the correlation
    between y_true and y_proxy in each record. This variable is sampled uniformly
    around the given correlation value with limited spread within the interval of
    possible correlation levels given true_mean and proxy_mean. This way, the
    correlation between y_true and y_proxy matches the target value on average.

    Parameters
    ----------
    N : int
        Total number of records.
    true_mean : float
        Expected mean of y_true. Must be in (0, 1).
    proxy_mean : float
        Expected mean of y_proxy. Must be in (0, 1).
    correlation : float
        Pearson correlation between y_true and y_proxy (marginal, across all records).
    random_seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    Dataset
        Dataset with N records, each containing ``"y_true"`` (int), ``"y_proxy"`` (int),
        and ``"RMSE"`` (float > 0). All y_true values are present (no missing values).

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

    Each record receives a latent value ``x_i ~ Uniform(-1, 1)`` representing
    "annotation difficulty".  The per-sample Pearson correlation is defined as:

    ```
    corr(x_i) = correlation + correlation_spread * x_i
    ```

    Because ``E[x] = 0`` for ``x ~ Uniform(-1, 1)``, the marginal
    correlation ``E[corr(X)] = correlation`` exactly, preserving the target
    value on average.  Records with low ``x`` have lower conditional
    correlation (proxy less reliable → higher oracle RMSE); records with high
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

    Since each record has its own probability vector, ``numpy.random.choice``
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
    These values are stored directly as ``RMSE``.
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

    # Vectorized CDF inversion to sample (y_true, y_proxy) per record
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
    RMSE = np.sqrt(error_prob_x)

    records = [
        {"y_true": int(yt), "y_proxy": int(yp), "RMSE": float(p)} for yt, yp, p in zip(y_true_arr, y_proxy_arr, RMSE)
    ]
    return Dataset(records)


def generate_gaussian_dataset(
    n: int,
    N: int,
    true_mean: float = 0.7,
    true_std: float = 1,
    proxy_mean: float = 0.6,
    proxy_std: float = 1,
    correlation: float = 0.8,
    random_seed: Optional[int] = None,
) -> Tuple[Dataset, Dataset]:
    """Generate a synthetic Gaussian dataset for evaluation.

    Parameters
    ----------
    n : int
        Number of records with both true and proxy labels (the labeled subset).
    N : int
        Number of records with proxy labels only (the unlabeled subset).
    true_mean : float
        Mean of the true label distribution.
    true_std : float
        Standard deviation of the true label distribution.
    proxy_mean : float
        Mean of the proxy label distribution.
    proxy_std : float
        Standard deviation of the proxy label distribution.
    correlation : float
        Pearson correlation between true and proxy labels.
    random_seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    Tuple[Dataset, Dataset]
        A tuple ``(labeled, unlabeled)``. The labeled Dataset contains ``n`` records,
        each with ``"y_true"`` and ``"y_proxy"``. The unlabeled Dataset contains ``N``
        records with ``"y_proxy"`` only.

    Notes
    -----
    **Target distribution**

    The goal is to jointly sample ``(y_true, y_proxy)`` from a bivariate Gaussian:

    ```
    (y_true, y_proxy) ~ N(μ, Σ)
    ```

    where:

    ```
    μ = (true_mean, proxy_mean)

    Σ = [[true_std²,                          ρ · true_std · proxy_std],
         [ρ · true_std · proxy_std,           proxy_std²              ]]
    ```

    and ``ρ`` is the target Pearson correlation.

    **Step 1 — Cholesky decomposition of Σ**

    To sample from ``N(0, Σ)``, we find a lower-triangular matrix ``L`` such that
    ``Σ = L @ Lᵀ`` (Cholesky factor). The construction uses the angle
    ``θ = arccos(ρ)``, so that ``cos(θ) = ρ`` and ``sin(θ) = √(1 - ρ²)``:

    ```
    L = [[true_std,                  0                  ],
         [proxy_std · cos(θ),        proxy_std · sin(θ) ]]
    ```

    One can verify ``L @ Lᵀ = Σ`` directly:

    ```
    L @ Lᵀ = [[true_std²,                    true_std · proxy_std · cos(θ)],
              [true_std · proxy_std · cos(θ), proxy_std² · (cos²(θ)+sin²(θ))]]

           = [[true_std²,                    true_std · proxy_std · ρ],
              [true_std · proxy_std · ρ,     proxy_std²              ]]  = Σ
    ```

    **Step 2 — Sampling via the linear transform**

    Let ``Z`` be a ``2 × (n+N)`` matrix whose entries are i.i.d. standard normals
    ``Z_i ~ N(0, 1)``. Then:

    ```
    Y = L @ Z
    ```

    gives a ``2 × (n+N)`` matrix where each column is a zero-mean sample from
    ``N(0, Σ)``. In component form, each column ``(Z₁, Z₂)`` maps to:

    ```
    Y₁ = true_std · Z₁
    Y₂ = proxy_std · cos(θ) · Z₁ + proxy_std · sin(θ) · Z₂
    ```

    The resulting properties are:
    - ``Var(Y₁) = true_std²`` and ``Var(Y₂) = proxy_std²`` (correct marginal variances)
    - ``Cov(Y₁, Y₂) = true_std · proxy_std · cos(θ) = true_std · proxy_std · ρ``
    - ``Corr(Y₁, Y₂) = ρ`` (correct Pearson correlation)

    **Step 3 — Shifting by the means**

    Adding the desired means shifts the distribution to ``N(μ, Σ)``:

    ```
    y_true  = true_mean  + Y[0, :]
    y_proxy = proxy_mean + Y[1, :]
    ```

    The first ``n`` columns form the labeled set (both ``y_true`` and ``y_proxy``
    are observed); columns ``n`` through ``n+N-1`` form the unlabeled set
    (only ``y_proxy`` is observed).
    """
    if abs(correlation) > 1:
        raise ValueError("Correlation should be between -1 and 1")
    rng = np.random.default_rng(seed=random_seed)
    angle = np.arccos(correlation)
    lin_transform = np.array([[true_std, 0], [proxy_std * np.cos(angle), proxy_std * np.sin(angle)]])

    Y = lin_transform @ rng.standard_normal(size=(2, n + N))

    y_true = true_mean + Y[0, :]
    y_proxy = proxy_mean + Y[1, :]
    labeled = [{"y_true": y_true[i], "y_proxy": y_proxy[i]} for i in range(n)]
    unlabeled = [{"y_proxy": y_proxy[i]} for i in range(n, n + N)]
    return Dataset(labeled), Dataset(unlabeled)
