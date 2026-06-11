from typing import Optional, Tuple, Union

import numpy as np
from numpy.random.bit_generator import SeedSequence
from numpy.typing import NDArray

from glide.core.validation import (
    _validate_bounds,
    _validate_equal_lengths,
    _validate_has_no_nan,
    _validate_is_integer,
    _validate_strictly_positive,
    _validate_y_true_burn_in,
)
from glide.samplers.core import _build_output, _compute_cutoff_indices, _shuffle


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
    ...     cost_limit=15,
    ...     random_seed=42
    ... )
    >>> pi
    array([0.0451754, 0.0451754])
    >>> xi
    array([0., 0.])
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
        _validate_y_true_burn_in(y_true)
        _validate_equal_lengths(y_true, y_proxy, names=["y_true", "y_proxy"])
        _validate_has_no_nan(y_proxy, "y_proxy")
        _validate_has_no_nan(y_true, "y_true")
        if np.max(np.abs(y_true - y_proxy)) == 0:
            raise ValueError("'y_proxy' predicts 'y_true' perfectly (zero MSE). Annotation probability would be zero")

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
        cost_limit: float,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ) -> Tuple[NDArray, NDArray]:
        """Sample observations with cost-optimal allocation between raters.

        Derives the optimal probability of querying the expensive rater (ground truth)
        based on relative costs and proxy quality.

        Samples are randomly permuted before drawing and the inverse permutation is applied
        to the output, so the returned arrays are always in the original input order. A
        post-draw cutoff is then applied to strictly respect the cost limit: samples beyond the
        cutoff are discarded by setting their entries in ``pi`` and ``xi`` to ``0.0`` and
        ``NaN`` respectively.

        The two returned arrays are intended for use with IPW-based downstream estimators. ``pi``
        holds the per-sample probability of querying the expensive rater. ``xi`` holds the
        annotation indicators for selected samples, with NaN marking samples excluded by the
        cost cutoff.

        Parameters
        ----------
        n_samples : int
            Total number of candidate samples to draw from. Must be a strictly positive integer.
        y_true_cost : float
            Per-sample cost of the expensive rater (H). Must be strictly positive.
        y_proxy_cost : float
            Per-sample cost of the cheap rater (G). Must be strictly positive.
        cost_limit : float
            Total annotation budget in cost units. Must be at least ``y_true_cost + y_proxy_cost``.
        random_seed : int or SeedSequence or None, optional
            Random seed passed to ``numpy.random.default_rng`` for reproducibility.
            Pass ``None`` (the default) to use a non-deterministic seed.

        Returns
        -------
        Tuple[NDArray, NDArray]
            [0]: array of shape ``(n_samples,)``, ``pi`` with per-sample annotation probabilities
            for selected samples and ``0.0`` for unselected samples.
            [1]: array of shape ``(n_samples,)``, ``xi`` with Bernoulli indicators:
            ``1.0`` if selected for annotation, ``0.0`` if not selected,
            ``NaN`` if excluded by the cost cutoff.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called before ``sample()``.
        ValueError
            - If ``n_samples`` is not a strictly positive integer.
            - If ``y_true_cost`` or ``y_proxy_cost`` is not strictly positive.
            - If ``cost_limit < y_true_cost + y_proxy_cost``.
        """
        if not hasattr(self, "_y_true_variance") or not hasattr(self, "_mean_squared_error"):
            raise RuntimeError("Call fit() before sample().")
        _validate_is_integer(n_samples, "n_samples")
        _validate_strictly_positive(n_samples, "n_samples")
        _validate_strictly_positive(y_true_cost, "y_true_cost")
        _validate_strictly_positive(y_proxy_cost, "y_proxy_cost")
        _validate_bounds(
            cost_limit,
            "cost_limit",
            lower=y_true_cost + y_proxy_cost,
            error_message=f"'cost_limit' should be at least {y_true_cost + y_proxy_cost}; got {cost_limit}.",
        )

        pi_opt = self._compute_optimal_probability(y_true_cost, y_proxy_cost)

        pi_all = np.full(n_samples, pi_opt)
        rng = np.random.default_rng(random_seed)
        pi_shuffled, order = _shuffle(pi_all, rng)
        xi_shuffled = rng.binomial(n=1, p=pi_shuffled).astype(float)
        cumulative_costs = np.cumsum(xi_shuffled * y_true_cost + y_proxy_cost)
        kept_indices = _compute_cutoff_indices(cumulative_costs, order, cost_limit)
        pi_out, xi_out = _build_output(kept_indices, pi_shuffled, xi_shuffled)
        return pi_out, xi_out
