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
    >>> from glide.samplers.cost_optimal_random import CostOptimalRandomSampler
    >>> y_true = np.array([1.0, 2.0])
    >>> y_proxy = np.array([1.1, 1.9])
    >>> sampler = CostOptimalRandomSampler()
    >>> sampler = sampler.fit(y_true, y_proxy)
    >>> indices, xi, pi = sampler.sample(
    ...     y_proxy=y_proxy,
    ...     y_true_cost=10.0,
    ...     y_proxy_cost=1.0,
    ...     budget=2,
    ...     random_seed=42
    ... )
    >>> len(indices)
    1
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
            Ground truth labels, shape (n_samples,), dtype float. Must not contain
            NaN values.
        y_proxy : NDArray
            Proxy labels, shape (n_samples,), dtype float. Must not contain NaN values.

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
              (proxy labels match ground truth perfectly).
        """
        if len(y_true) == 0:
            raise ValueError("y_true must not be empty")
        if len(y_true) != len(y_proxy):
            raise ValueError(f"y_true and y_proxy must have the same length; got {len(y_true)} and {len(y_proxy)}.")
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_proxy)):
            raise ValueError("Input contains NaN values")

        y_true_variance = np.var(y_true, ddof=1)
        if y_true_variance == 0.0:
            raise ValueError("Input ground-truth values have zero variance")

        mean_squared_error = np.mean((y_true - y_proxy) ** 2)
        if mean_squared_error == 0.0:
            raise ValueError("Proxy values have zero MSE with ground-truths")

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
            pi = np.sqrt(ratio)
        return pi

    def sample(
        self,
        y_proxy: NDArray,
        y_true_cost: float,
        y_proxy_cost: float,
        budget: float,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ) -> Tuple[NDArray, NDArray, float]:
        """Sample observations with cost-optimal allocation between raters.

        Determines the optimal probability of requesting the expensive rater
        (ground truth) in addition to the cheap rater (proxy) based on relative costs
        and annotation quality. Each observation receives a drawing probability
        that is either the optimal value or 1.0 (if the budget constraint binds).
        Probabilities are capped at 1 before sampling, so the actual number of
        selected items is a random variable.

        Parameters
        ----------
        y_proxy : NDArray
            Proxy labels, shape ``(n_samples,)``.
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
        Tuple[NDArray, NDArray, float]
            Let T <= n_samples the maximum number of samples that can be annotated
            within the budget:

            [0]: indices, shape (T,), dtype int — sorted indices of the T samples selected
                 uniformly at random from the input for annotation.
            [1]: xi, shape (T,), dtype float — Bernoulli indicators for each selected
                 sample: 1 if the expensive rater (ground truth) was selected, 0 if only
                 the cheap rater (proxy) is used.
            [2]: pi, dtype float — optimal annotation probability used (probability of
                 selecting the expensive rater for each sample).

        Raises
        ------
        ValueError
            - If fit() has not been called yet.
            - If ``y_true_cost`` or ``y_proxy_cost`` is not strictly positive.
            - If ``budget`` is not strictly positive.
            - If ``budget`` is too small to afford a single sample.
        """
        if not hasattr(self, "_y_true_variance") or not hasattr(self, "_mean_squared_error"):
            raise RuntimeError("fit() must be called before sample()")
        if y_true_cost <= 0.0:
            raise ValueError(f"y_true_cost must be strictly positive; got {y_true_cost}.")
        if y_proxy_cost <= 0.0:
            raise ValueError(f"y_proxy_cost must be strictly positive; got {y_proxy_cost}.")
        if budget <= 0:
            raise ValueError(f"budget must be strictly positive; got {budget}.")

        pi = self._compute_optimal_probability(y_true_cost, y_proxy_cost)
        cost_per_sample = y_true_cost * pi + y_proxy_cost
        T = int(np.floor(budget / cost_per_sample))
        if T < 1:
            raise ValueError(
                f"Budget {budget} is too small to afford a single sample at cost_per_sample={cost_per_sample}."
            )

        rng = np.random.default_rng(random_seed)
        N = len(y_proxy)
        if T < N:
            indices = np.sort(rng.choice(N, size=T, replace=False))
        else:
            indices = np.arange(N)

        xi = rng.binomial(n=1, p=pi, size=len(indices)).astype(float)
        return indices, xi, pi
