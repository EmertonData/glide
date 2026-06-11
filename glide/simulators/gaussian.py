from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from glide.core.validation import _validate_bounds


def generate_gaussian_dataset(
    n_total: int,
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
    n_total : int
        Total number of samples to generate.
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
        [0]: array of shape ``(n_total,)``, oracle true labels with no NaN
        [1]: array of shape ``(n_total,)``, proxy labels with no NaN

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

    Let ``Z`` be a ``2 × n_total`` matrix whose entries are i.i.d. standard normals
    ``Z_i ~ N(0, 1)``. Then:

    ```
    Y = L @ Z
    ```

    gives a ``2 × n_total`` matrix where each column is a zero-mean sample from
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

    Examples
    --------
    >>> import numpy as np
    >>> from glide.simulators import generate_gaussian_dataset
    >>> y_true, y_proxy = generate_gaussian_dataset(n_total=8, random_seed=42)
    >>> len(y_true)
    8
    >>> int(np.sum(~np.isnan(y_true)))
    8
    """
    _validate_bounds(correlation, "correlation", lower=-1, upper=1)
    rng = np.random.default_rng(seed=random_seed)
    angle = np.arccos(correlation)
    lin_transform = np.array([[true_std, 0], [proxy_std * np.cos(angle), proxy_std * np.sin(angle)]])

    Y = lin_transform @ rng.standard_normal(size=(2, n_total))

    y_true = true_mean + Y[0, :].copy()
    y_proxy = proxy_mean + Y[1, :]

    return y_true, y_proxy
