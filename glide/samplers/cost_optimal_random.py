from typing import Optional, Tuple, Union

import numpy as np
from numpy.random.bit_generator import SeedSequence
from numpy.typing import NDArray


class CostOptimalRandomSampler:
    """Sampler implementing cost-optimal random annotation.

    Implements the optimal random sampling strategy for two-rater annotation,
    where one rater is expensive (ground truth) and one is cheap (proxy).
    Determines the optimal probability of requesting the expensive rater
    based on relative costs and annotation quality differences.

    References
    ----------
    Angelopoulos, A. N., Eisenstein, J., Berant, J., Agarwal, A., and Fisch, A. (2025).
    Cost-Optimal Active AI Model Evaluation. arXiv:2506.07949, §2 — Optimal Random Annotation.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.samplers.cost_optimal_random import CostOptimalRandomSampler
    >>> y_true = np.array([1.0, 2.0])
    >>> y_proxy = np.array([1.1, 1.9])
    >>> sampler = CostOptimalRandomSampler()
    >>> sampler.fit(y_true, y_proxy)
    <glide.samplers.cost_optimal_random.CostOptimalRandomSampler object at ...>
    >>> indices, xi, pi = sampler.sample(
    ...     y_proxy=y_proxy,
    ...     y_true_cost=10.0,
    ...     y_proxy_cost=1.0,
    ...     budget=2,
    ...     random_seed=42
    ... )
    >>> len(indices)
    1

    When budget is large enough to cover all records, all indices are selected:

    >>> indices, xi, pi = sampler.sample(
    ...     y_proxy=y_proxy,
    ...     y_true_cost=1.0,
    ...     y_proxy_cost=1.0,
    ...     budget=1000,
    ...     random_seed=42
    ... )
    >>> np.array_equal(indices, np.arange(len(y_proxy)))
    True

    Random seed defaults to None for non-deterministic mode:

    >>> indices, xi, pi = sampler.sample(
    ...     y_proxy=y_proxy,
    ...     y_true_cost=10.0,
    ...     y_proxy_cost=1.0,
    ...     budget=2
    ... )
    >>> len(indices) >= 0
    True
    """

    def fit(
        self,
        y_true: NDArray,
        y_proxy: NDArray,
    ) -> "CostOptimalRandomSampler":
        """Estimate MSE(H, G) and Var(H) from a fully-labeled burn-in dataset.

        Parameters
        ----------
        y_true : NDArray
            Ground truth labels, shape (n,), dtype float. No NaN values allowed.
        y_proxy : NDArray
            Proxy labels, shape (n,), dtype float. No NaN values allowed.

        Returns
        -------
        CostOptimalRandomSampler
            Self, to allow method chaining.

        Raises
        ------
        ValueError
            If either array contains NaN, is empty, or arrays have different lengths.
            If Var(H) is zero (degenerate labels).
            If MSE(H, G) is zero (proxy is perfect).
        """
        if len(y_true) == 0:
            raise ValueError(
                "'y_true' must not be empty; fit() requires a non-empty burn-in dataset "
                "to estimate Var(H) and MSE(H, G)."
            )
        if len(y_proxy) == 0:
            raise ValueError(
                "'y_proxy' must not be empty; fit() requires a non-empty burn-in dataset "
                "to estimate Var(H) and MSE(H, G)."
            )
        if len(y_true) != len(y_proxy):
            raise ValueError(f"'y_true' and 'y_proxy' must have the same length; got {len(y_true)} and {len(y_proxy)}.")
        if np.any(np.isnan(y_true)):
            raise ValueError(
                "'y_true' must not contain NaN values; a fully-labeled burn-in dataset is required "
                "to compute valid variance and MSE estimates."
            )
        if np.any(np.isnan(y_proxy)):
            raise ValueError(
                "'y_proxy' must not contain NaN values; a fully-labeled burn-in dataset is required "
                "to compute valid variance and MSE estimates."
            )

        y_true_variance = np.var(y_true, ddof=1)
        if y_true_variance == 0.0:
            raise ValueError(
                "Var(H) is zero — all labels are identical. Estimation is impossible with degenerate labels."
            )

        mean_squared_error = np.mean((y_true - y_proxy) ** 2)
        if mean_squared_error == 0.0:
            raise ValueError(
                "MSE(H, G) is zero — the proxy perfectly predicts the expensive rater. "
                "There is no benefit from annotating with H."
            )

        self._y_true_variance = y_true_variance
        self._mean_squared_error = mean_squared_error
        return self

    def _compute_optimal_probability(
        self,
        y_true_cost: float,
        y_proxy_cost: float,
    ) -> float:
        threshold = y_true_cost / (y_true_cost + y_proxy_cost) * self._y_true_variance
        if self._mean_squared_error >= threshold:
            pi = 1.0
        else:
            ratio = (
                (y_proxy_cost / y_true_cost)
                * self._mean_squared_error
                / (self._y_true_variance - self._mean_squared_error)
            )
            pi = np.sqrt(ratio)
        return pi

    def sample(
        self,
        y_proxy: NDArray,
        y_true_cost: float,
        y_proxy_cost: float,
        budget: int,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ) -> Tuple[NDArray, NDArray, float]:
        """Apply cost-optimal random sampling to a dataset.

        Parameters
        ----------
        y_proxy : NDArray
            Proxy labels, shape ``(n_samples,)``.
        y_true_cost : float
            Per-record cost of the expensive rater (H). Must be strictly positive.
        y_proxy_cost : float
            Per-record cost of the cheap rater (G). Must be strictly positive.
        budget : int
            Total annotation budget in cost units. Must be strictly positive.
        random_seed : int or SeedSequence or None, optional
            Random seed passed to ``numpy.random.default_rng`` for reproducibility.
            Pass ``None`` (the default) to use a non-deterministic seed.

        Returns
        -------
        Tuple[NDArray, NDArray, float]
            [0]: indices, shape (T,), dtype int — sorted indices of selected records.
            [1]: xi, shape (T,), dtype float — Bernoulli indicators: 1 if expensive rater
                 was selected, 0 if cheap rater only.
            [2]: pi, dtype float — optimal annotation probability used.

        Raises
        ------
        ValueError
            If fit() has not been called yet.
            If y_true_cost or y_proxy_cost is not strictly positive.
            If budget is not a strictly positive integer.
            If budget is too small to afford a single record.
        """
        if not hasattr(self, "_y_true_variance") or not hasattr(self, "_mean_squared_error"):
            raise ValueError(
                "Call fit() before sample(); state variables (_y_true_variance, _mean_squared_error) "
                "must be initialized from the burn-in dataset."
            )
        if y_true_cost <= 0.0:
            raise ValueError(f"'y_true_cost' must be strictly positive; got {y_true_cost}.")
        if y_proxy_cost <= 0.0:
            raise ValueError(f"'y_proxy_cost' must be strictly positive; got {y_proxy_cost}.")
        if (not isinstance(budget, (int, np.integer))) or isinstance(budget, bool) or budget <= 0:
            raise ValueError(f"'budget' must be a strictly positive integer; got {budget!r}.")

        pi = self._compute_optimal_probability(y_true_cost, y_proxy_cost)
        cost_per_record = y_true_cost * pi + y_proxy_cost
        T = int(np.floor(budget / cost_per_record))
        if T < 1:
            raise ValueError(
                f"Budget {budget} is too small to afford a single record at cost_per_record={cost_per_record}."
            )

        rng = np.random.default_rng(random_seed)
        N = len(y_proxy)
        if T < N:
            indices = np.sort(rng.choice(N, size=T, replace=False))
        else:
            indices = np.arange(N)

        xi = rng.binomial(n=1, p=pi, size=len(indices)).astype(float)
        return indices, xi, pi
