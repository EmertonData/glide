from typing import Optional, Tuple

import numpy as np

from glide.core.dataset import Dataset


def generate_binary_dataset(
    n: int,
    N: int,
    true_mean: float = 0.7,
    proxy_mean: float = 0.6,
    correlation: float = 0.8,
    random_seed: Optional[int] = None,
) -> Dataset:
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
    Dataset
        A Dataset of n + N records. Each record contains ``"proxy"`` (always present)
        and ``"true"`` (only for the first ``n`` records).

    Raises
    ------
    ValueError
        If ``true_mean`` is not in (0, 1).
    ValueError
        If ``proxy_mean`` is not in (0, 1).
    ValueError
        If the combination of ``true_mean``, ``proxy_mean``, and ``correlation`` is
        impossible (leads to negative joint probabilities).

    References
    ----------
    .. [SO] `Correlation between Bernoulli Variables <https://math.stackexchange.com/questions/610443/finding-a-correlation-between-bernoulli-variables>`_

    """
    if not (0 < true_mean < 1):
        raise ValueError(f"true_mean must be in (0, 1), got {true_mean}")
    if not (0 < proxy_mean < 1):
        raise ValueError(f"proxy_mean must be in (0, 1), got {proxy_mean}")

    rng = np.random.default_rng(seed=random_seed)

    t_mean = true_mean
    p_mean = proxy_mean

    # we will generate pairs values (true, proxy) with true and proxy equal to 0 or 1
    # probability of outcome (1, 1)
    both_1_prob = correlation * np.sqrt(t_mean * p_mean * (1 - t_mean) * (1 - p_mean)) + t_mean * p_mean
    # probabilities of outcomes (0, 0), (0, 1), (1, 0), (1, 1)
    probs = [1 - t_mean - p_mean + both_1_prob, p_mean - both_1_prob, t_mean - both_1_prob, both_1_prob]
    # some combinations of true_mean, proxy_mean and correlation are impossible
    # and lead to negative probabilities
    if min(probs) <= 0:
        raise ValueError(
            f"Impossible combination of true_mean={true_mean}, proxy_mean={proxy_mean}, "
            f"and correlation={correlation}: leads to negative joint probabilities"
        )

    # generate the outcome pairs as integers between 0 and 3 inclusive
    samples = rng.choice(4, p=probs, size=n)
    # extract the true and proxy values via integer division and modulo 2
    # we have 0 = (0, 0), 1 = (0, 1), 2 = (1, 0), 3 = (1, 1)
    y_true = samples // 2
    y_proxy_labeled = samples % 2

    # generate proxy values for un labeled samples
    y_proxy_unlabeled = rng.choice(2, p=[1 - p_mean, p_mean], size=N)

    records = []
    for y_true_, y_proxy_ in zip(y_true, y_proxy_labeled):
        records.append({"y_true": int(y_true_), "y_proxy": int(y_proxy_)})
    for y_proxy_ in y_proxy_unlabeled:
        records.append({"y_proxy": int(y_proxy_)})

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
        raise ValueError("Correlation should be strictly between -1 and 1")
    rng = np.random.default_rng(seed=random_seed)
    angle = np.arccos(correlation)
    lin_transform = np.array([[true_std, 0], [proxy_std * np.cos(angle), proxy_std * np.sin(angle)]])

    Y = lin_transform @ rng.standard_normal(size=(2, n + N))

    y_true = true_mean + Y[0, :]
    y_proxy = proxy_mean + Y[1, :]
    labeled = [{"y_true": y_true[i], "y_proxy": y_proxy[i]} for i in range(n)]
    unlabeled = [{"y_proxy": y_proxy[i]} for i in range(n, n + N)]
    return Dataset(labeled), Dataset(unlabeled)
