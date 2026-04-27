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
    Angelopoulos, Anastasios N., Jacob Eisenstein, Jonathan Berant, Alekh
    Agarwal, and Adam Fisch. "Cost-optimal active ai model evaluation." arXiv
    preprint arXiv:2506.07949 (2025).

    Examples
    --------
    >>> import numpy as np
    >>> from glide.samplers import CostOptimalRandomSampler
    >>> y_true = np.array([1.0, 2.0])
    >>> y_proxy = np.array([1.1, 1.9])
    >>> sampler = CostOptimalRandomSampler()
    >>> sampler = sampler.fit(y_true, y_proxy)
    >>> pi, xi = sampler.sample(
    ...     n_samples=2,
    ...     y_true_cost=10.0,
    ...     y_proxy_cost=1.0,
    ...     budget=2,
    ...     random_seed=42
    ... )
    >>> pi # doctest: +ELLIPSIS
    0.045...
    >>> xi[0]  # doctest: +ELLIPSIS
    np.float64(0.0)
    >>> np.isnan(xi[1])  # doctest: +ELLIPSIS
    np.True_
    """

    def fit(
        self,
        y_true: NDArray,
        y_proxy: NDArray,
    ) -> "CostOptimalRandomSampler":
        """Calibrate the sampler by estimating proxy quality and label variance.

        Fits the sampler to a fully-labeled burn-in dataset by computing the mean
        squared error between proxy labels and ground truth labels, as well as the
        variance of ground truth labels. These statistics are used to determine the
        optimal probability of requesting expensive ground truth annotations during
        the sampling phase.

        Parameters
        ----------
        y_true : NDArray
            Ground truth labels, shape (n_samples,). Must not contain
            NaN values.
        y_proxy : NDArray
            Proxy labels, shape (n_samples,). Must not contain NaN values.

        Returns
        -------
        CostOptimalRandomSampler
            Self, to allow method chaining.

        Raises
        ------
        ValueError
            - If either array contains NaN, is empty, or arrays have different lengths.
            - If the variance of ``y_true`` is zero (all labels are identical).
            - If the mean squared error between ``y_true`` and ``y_proxy`` is zero
              (proxy labels match ground truth perfectly). This would lead to zero
              annotation probability making sampling impossible.
        """
        if len(y_true) == 0:
            raise ValueError("y_true must not be empty")
        if len(y_true) != len(y_proxy):
            raise ValueError(f"y_true and y_proxy must have the same length; got {len(y_true)} and {len(y_proxy)}.")
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_proxy)):
            raise ValueError("Input contains NaN values")
        if np.all(y_true == y_proxy):
            raise ValueError("Proxy values have zero MSE with ground-truths")
        if len(np.unique(y_true)) < 2:
            raise ValueError("Input ground-truth values have zero variance")

        y_true_variance = np.var(y_true, ddof=1)
        mean_squared_error = np.mean((y_true - y_proxy) ** 2)

        self._y_true_variance = y_true_variance
        self._mean_squared_error = mean_squared_error
        return self

    def _compute_optimal_probability(
        self,
        y_true_cost: float,
        y_proxy_cost: float,
    ) -> float:
        threshold = self._y_true_variance * y_true_cost / (y_true_cost + y_proxy_cost)
        if self._mean_squared_error >= threshold:
            pi = 1.0
        else:
            ratio = (
                (y_proxy_cost / y_true_cost)
                * self._mean_squared_error
                / (self._y_true_variance - self._mean_squared_error)
            )
            pi = float(np.sqrt(ratio))
        return pi

    def sample(
        self,
        n_samples: int,
        y_true_cost: float,
        y_proxy_cost: float,
        budget: float,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ) -> Tuple[float, NDArray]:
        """Sample observations with cost-optimal allocation between raters.

        Derives the optimal probability of querying the expensive rater (ground truth)
        based on relative costs and proxy quality. Determines the maximum number of samples
        that can be afforded within the budget, then selects those samples for annotation.

        If the budget can afford fewer samples than n_samples, samples are drawn uniformly
        at random without replacement. Otherwise, all n_samples are selected.

        For each selected sample, an independent Bernoulli draw with the optimal probability
        determines whether the expensive rater is also queried (1) or only the proxy is used (0).

        Parameters
        ----------
        n_samples : int
            Total number of candidate samples to draw from. Must be a strictly positive integer.
        y_true_cost : float
            Per-sample cost of the expensive rater (H). Must be strictly positive.
        y_proxy_cost : float
            Per-sample cost of the cheap rater (G). Must be strictly positive.
        budget : float
            Total annotation budget in cost units. Must be strictly positive.
        random_seed : int or SeedSequence or None, optional
            Random seed passed to ``numpy.random.default_rng`` for reproducibility.
            Pass ``None`` (the default) to use a non-deterministic seed.

        Returns
        -------
        Tuple[float, NDArray]
            [0]: float, pi with the optimal probability of querying the expensive rater.
            [1]: array of shape ``(n_samples,)``, xi with indicators for each sample:
            1 if both raters are queried, 0 if only proxy is used, NaN if sample is not selected.

        Raises
        ------
        ValueError
            - If fit() has not been called yet.
            - If ``n_samples`` is not a strictly positive integer.
            - If ``y_true_cost`` or ``y_proxy_cost`` is not strictly positive.
            - If ``budget`` is not strictly positive.
            - If ``budget`` is too small to afford a single sample.
        """
        if not hasattr(self, "_y_true_variance") or not hasattr(self, "_mean_squared_error"):
            raise RuntimeError("fit() must be called before sample()")
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError(f"n_samples must be a strictly positive integer; got {n_samples!r}.")
        if y_true_cost <= 0.0:
            raise ValueError(f"y_true_cost must be strictly positive; got {y_true_cost}.")
        if y_proxy_cost <= 0.0:
            raise ValueError(f"y_proxy_cost must be strictly positive; got {y_proxy_cost}.")
        if budget <= 0:
            raise ValueError(f"budget must be strictly positive; got {budget}.")

        pi = self._compute_optimal_probability(y_true_cost, y_proxy_cost)
        cost_per_sample = y_true_cost * pi + y_proxy_cost
        n_affordable = int(np.floor(budget / cost_per_sample))
        if n_affordable < 1:
            raise ValueError(
                f"Budget {budget} is too small to afford a single sample at cost_per_sample={cost_per_sample}."
            )

        rng = np.random.default_rng(random_seed)
        xi = np.full(n_samples, np.nan)
        if n_affordable < n_samples:
            indices = np.sort(rng.choice(n_samples, size=n_affordable, replace=False))
        else:
            indices = np.arange(n_samples)
        xi[indices] = rng.binomial(n=1, p=pi, size=len(indices)).astype(float)
        return pi, xi
