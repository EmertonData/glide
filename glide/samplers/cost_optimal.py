from typing import Optional, Tuple, Union

import numpy as np
from numpy.random.bit_generator import SeedSequence
from numpy.typing import NDArray


class CostOptimalSampler:
    """Sampler that draws elements with optimal probabilities based on uncertainty
    scores and annotation costs on a limited budget.

    Implements a cost-optimal active annotation policy. Each sample is assigned
    an annotation probability proportional to how unreliable the proxy label is
    expected to be for that sample, as measured by the caller-supplied per-sample
    uncertainty scores. Samples with high expected proxy error are annotated more
    often; samples where the proxy is reliable are annotated less often. This
    concentrates the annotation budget where it matters most.

    The caller pre-computes per-sample uncertainty scores and passes them as a
    1D array to ``sample()``. This class does not learn those scores internally.

    References
    ----------
    Angelopoulos, A. N., Eisenstein, J., Berant, J., Agarwal, A., and Fisch, A.
    (2025). Cost-Optimal Active AI Model Evaluation. arXiv:2506.07949, §2.3.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.samplers import CostOptimalSampler
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
    >>> uncertainties = np.array([0.1, 0.4, 0.1, 0.4])
    >>> sampler = CostOptimalSampler().fit(y_true)
    >>> pi, xi = sampler.sample(
    ...     uncertainties,
    ...     y_true_cost=10.0,
    ...     y_proxy_cost=1.0,
    ...     budget=5,
    ...     random_seed=0
    ... )
    >>> float(pi[0])  # doctest: +ELLIPSIS
    0.084...
    >>> xi[0]
    np.float64(0.0)
    >>> np.isnan(xi[-1])
    np.True_
    """

    def _validate_y_true(self, y_true: NDArray) -> None:
        if len(y_true) == 0:
            raise ValueError("'y_true' must be non-empty.")
        if np.any(np.isnan(y_true)):
            raise ValueError("'y_true' contains NaN values. The burn-in dataset must be fully labeled.")
        if np.min(y_true) == np.max(y_true):
            raise ValueError("'y_true' label values have zero variance.")

    def _validate_uncertainties(self, uncertainties: NDArray) -> None:
        if np.any(np.isnan(uncertainties)):
            raise ValueError(
                "All uncertainty values must be finite; got a NaN value. "
                "A NaN conditional MSE estimate cannot be used to compute sampling probabilities."
            )
        if np.any(uncertainties <= 0.0):
            raise ValueError(
                "All uncertainty values must be strictly positive; got a non-positive value. "
                "A sample with zero conditional MSE would never be annotated by H."
            )

    def fit(self, y_true: NDArray) -> "CostOptimalSampler":
        """Estimate the true label variance from a burn-in dataset.

        The true label variance is computed ahead of active sampling so that
        ``sample()`` can derive the cost-optimal annotation probabilities.
        The caller must preserve these arrays for downstream estimation via
        inverse-variance weighting with the active-phase estimates.

        Parameters
        ----------
        y_true : NDArray
            1D float array of expensive-rater labels from the burn-in phase. No NaN.

        Returns
        -------
        CostOptimalSampler
            The fitted sampler (returns ``self`` for method chaining).

        Raises
        ------
        ValueError
            If ``y_true`` is empty, contains NaN, or all labels are identical (zero true label variance).

        """
        self._validate_y_true(y_true)
        self._y_true_variance = np.var(y_true, ddof=1)
        return self

    def _compute_gamma(
        self,
        tau: float,
        uncertainties: NDArray,
        y_true_cost: float,
        y_proxy_cost: float,
    ) -> float:
        cost_ratio = y_proxy_cost / y_true_cost
        above_mask = uncertainties > tau**2
        prob_above = np.mean(above_mask)
        e_u_below = np.mean(uncertainties * ~above_mask)
        denominator = max(self._y_true_variance - e_u_below, 0.0)
        if denominator > 0.0:
            gamma_uncapped = np.sqrt((cost_ratio + prob_above) / denominator)
        else:
            gamma_uncapped = float("inf")
        gamma = min(gamma_uncapped, 1.0 / tau)
        return gamma

    def _compute_per_sample_probabilities(
        self,
        tau: float,
        gamma: float,
        uncertainties: NDArray,
    ) -> NDArray:
        sqrt_u = np.sqrt(uncertainties)
        probabilities = np.where(sqrt_u > tau, 1.0, gamma * sqrt_u)
        return probabilities

    def _compute_objective(
        self,
        tau: float,
        uncertainties: NDArray,
        y_true_cost: float,
        y_proxy_cost: float,
    ) -> float:
        gamma = self._compute_gamma(tau, uncertainties, y_true_cost, y_proxy_cost)
        pi_values = self._compute_per_sample_probabilities(tau, gamma, uncertainties)
        mean_pi = np.mean(pi_values)
        cost_term = y_true_cost * mean_pi + y_proxy_cost
        error_term = self._y_true_variance + np.mean(uncertainties * (1.0 / pi_values - 1.0))
        objective = cost_term * error_term
        return objective

    def _find_optimal_threshold(
        self,
        uncertainties: NDArray,
        y_true_cost: float,
        y_proxy_cost: float,
    ) -> float:
        sqrt_u_values = np.sqrt(uncertainties)
        candidates = np.unique(sqrt_u_values)
        # the breakpoints where the policy changes character
        # are the distinct values of sqrt(u_i).
        objectives = [self._compute_objective(tau, uncertainties, y_true_cost, y_proxy_cost) for tau in candidates]
        optimal_tau = candidates[np.argmin(objectives)]
        return optimal_tau

    def sample(
        self,
        uncertainties: NDArray,
        y_true_cost: float,
        y_proxy_cost: float,
        budget: int,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ) -> Tuple[NDArray, NDArray]:
        """Draw samples and annotation indicators under the cost optimal policy.

        Per-sample annotation probabilities are derived from the supplied uncertainty
        scores and the true label variance estimated by ``fit()``. When the budget is
        tight, samples beyond a certain index in the input array are excluded from
        sampling entirely and receive a probability of zero.

        Parameters
        ----------
        uncertainties : NDArray
            1D float array of shape ``(n_samples,)`` containing the pre-computed per-sample
            expected squared error of the proxy label. All values must be strictly positive.
        y_true_cost : float
            Cost of one expensive-rater annotation. Must be strictly positive.
        y_proxy_cost : float
            Cost of one cheap-proxy annotation. Must be strictly positive.
        budget : int
            Total annotation budget. Must be strictly positive.
        random_seed : int or SeedSequence or None, optional
            Random seed passed to ``numpy.random.default_rng`` for reproducibility.
            Pass ``None`` (the default) to use a non-deterministic seed.

        Returns
        -------
        Tuple[NDArray, NDArray]
            [0]: array of shape ``(n_samples,)``, ``pi`` with per-sample annotation probabilities
            for selected samples and ``0.0`` for unselected samples.
            [1]: array of shape ``(n_samples,)``, ``xi`` with Bernoulli indicators:
            ``1.0`` if the expensive rater was queried, ``0.0`` if only the proxy
            was used, ``NaN`` if the sample was not selected.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called before ``sample()``.
        ValueError
            - If ``y_true_cost`` or ``y_proxy_cost`` is not strictly positive.
            - If ``budget`` is not strictly positive.
            - If any uncertainty value is NaN or non-positive.
            - If ``budget`` is too small to afford a single sample.

        """
        if not hasattr(self, "_y_true_variance"):
            raise RuntimeError("Call fit() before sample().")
        if y_true_cost <= 0.0:
            raise ValueError(f"'y_true_cost' must be strictly positive; got {y_true_cost}.")
        if y_proxy_cost <= 0.0:
            raise ValueError(f"'y_proxy_cost' must be strictly positive; got {y_proxy_cost}.")
        if budget <= 0:
            raise ValueError(f"'budget' must be strictly positive; got {budget}.")
        self._validate_uncertainties(uncertainties)

        tau_star = self._find_optimal_threshold(uncertainties, y_true_cost, y_proxy_cost)
        gamma_star = self._compute_gamma(tau_star, uncertainties, y_true_cost, y_proxy_cost)
        pi_all = self._compute_per_sample_probabilities(tau_star, gamma_star, uncertainties)

        cumulative_costs = np.cumsum(y_true_cost * pi_all + y_proxy_cost)
        T = np.searchsorted(cumulative_costs, budget, side="right")
        if T < 1:
            raise ValueError(f"Budget {budget} is too small to afford a single sample with the given inputs.")

        n_samples = len(uncertainties)
        T = min(T, n_samples)

        rng = np.random.default_rng(random_seed)

        pi = np.zeros(n_samples)
        xi = np.full(n_samples, np.nan)
        pi[:T] = pi_all[:T]
        xi[:T] = rng.binomial(n=1, p=pi_all[:T], size=T).astype(float)

        return pi, xi
