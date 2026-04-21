from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def generate_gaussian_dataset(
    n_labeled: int,
    n_unlabeled: int,
    true_mean: float = 0.7,
    true_std: float = 1,
    proxy_mean: float = 0.6,
    proxy_std: float = 1,
    correlation: float = 0.8,
    random_seed: Optional[int] = None,
) -> Tuple[NDArray, NDArray]:
    """Generate a synthetic Gaussian dataset for evaluation.

    Parameters
    ----------
    n_labeled : int
        Number of samples with both true and proxy labels (the labeled subset).
    n_unlabeled : int
        Number of samples with proxy labels only (the unlabeled subset).
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
    Tuple[NDArray, NDArray]
        [0]: array of shape ``(n_labeled+n_unlabeled,)``, y_true with labeled values and NaN for unlabeled rows
        [1]: array of shape ``(n_labeled+n_unlabeled,)``, y_proxy with all values present

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

    Let ``Z`` be a ``2 × (n_labeled+n_unlabeled)`` matrix whose entries are i.i.d. standard normals
    ``Z_i ~ N(0, 1)``. Then:

    ```
    Y = L @ Z
    ```

    gives a ``2 × (n_labeled+n_unlabeled)`` matrix where each column is a zero-mean sample from
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

    The first ``n_labeled`` columns form the labeled set (both ``y_true`` and ``y_proxy``
    are observed); columns ``n_labeled`` through ``n_labeled+n_unlabeled-1`` form the unlabeled set
    (only ``y_proxy`` is observed).
    """
    if abs(correlation) > 1:
        raise ValueError("Correlation should be between -1 and 1")
    rng = np.random.default_rng(seed=random_seed)
    angle = np.arccos(correlation)
    lin_transform = np.array([[true_std, 0], [proxy_std * np.cos(angle), proxy_std * np.sin(angle)]])

    Y = lin_transform @ rng.standard_normal(size=(2, n_labeled + n_unlabeled))

    y_true = true_mean + Y[0, :].copy()
    y_true[n_labeled:] = np.nan
    y_proxy = proxy_mean + Y[1, :]

    return y_true, y_proxy
