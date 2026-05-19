import math
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


class CostOptimalSampler:
    """Sampler that draws elements with probabilities proportional to uncertainty scores.

    Implements the cost-optimal active annotation policy (Proposition 2). Unlike the
    random policy that assigns the same annotation probability to every record,
    this sampler assigns a per-record annotation probability that scales with the
    conditional mean squared error ``u(x) = E[(H - G)² | X = x]``. Records where
    the cheap proxy G is likely to be wrong receive a higher probability of being
    annotated by the expensive rater H; records where G is reliable are sampled
    less often. This concentrates the annotation budget where it matters most.

    The caller pre-computes per-record conditional error estimates and passes them
    as a 1D array to ``sample()``. This class does not learn ``u(x)`` internally.

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
    >>> sampler = CostOptimalSampler()
    >>> _ = sampler.fit(y_true)
    >>> pi, xi = sampler.sample(uncertainties, y_true_cost=10.0, y_proxy_cost=1.0, budget=5, random_seed=0)
    >>> pi.shape == xi.shape == uncertainties.shape
    True
    """

    def _validate_fit_inputs(self, y_true: NDArray) -> None:
        if len(y_true) == 0:
            raise ValueError("'y_true' must be non-empty.")
        if np.any(np.isnan(y_true)):
            raise ValueError("'y_true' contains NaN values. The burn-in dataset must be fully labeled.")
        if np.min(y_true) == np.max(y_true):
            raise ValueError("'y_true' label values have zero variance.")

    def fit(self, y_true: NDArray) -> "CostOptimalSampler":
        """Estimate Var(H) from a fully-labeled burn-in dataset.

        Corresponds to Policy A2 from Angelopoulos et al. (2025): the burn-in
        records are annotated unconditionally (π = 1) to bootstrap the sampler.
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
            If ``y_true`` is empty, contains NaN, or ``Var(H) == 0`` (all labels are identical).

        Examples
        --------
        >>> import numpy as np
        >>> from glide.samplers import CostOptimalSampler
        >>> sampler = CostOptimalSampler().fit(np.array([1.0, 2.0, 3.0]))
        >>> float(sampler._y_true_variance)
        1.0
        """
        self._validate_fit_inputs(y_true)
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
            gamma_uncapped = float(np.sqrt((cost_ratio + prob_above) / denominator))
        else:
            gamma_uncapped = float("inf")
        gamma = min(gamma_uncapped, 1.0 / tau)
        return gamma

    def _compute_per_record_probabilities(
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
        pi_values = self._compute_per_record_probabilities(tau, gamma, uncertainties)
        mean_pi = np.mean(pi_values)
        cost_term = y_true_cost * mean_pi + y_proxy_cost
        # For records where pi = 0 (i.e. u = 0), the product u * (pi^{-1} - 1) = 0 regardless
        # of pi^{-1}; replacing pi with 1 in the denominator preserves this zero contribution.
        safe_pi = np.where(pi_values > 0.0, pi_values, 1.0)
        error_term = self._y_true_variance + np.mean(uncertainties * (1.0 / safe_pi - 1.0))
        objective = cost_term * error_term
        return float(objective)

    def _find_optimal_threshold(
        self,
        uncertainties: NDArray,
        y_true_cost: float,
        y_proxy_cost: float,
    ) -> float:
        sqrt_u_values = np.sqrt(uncertainties)
        candidates = np.unique(sqrt_u_values)
        # τ must be strictly positive; the breakpoints where the policy changes character
        # are the distinct positive values of sqrt(u_i).
        candidates = candidates[candidates > 0.0]
        objectives = [self._compute_objective(tau, uncertainties, y_true_cost, y_proxy_cost) for tau in candidates]
        optimal_tau = float(candidates[int(np.argmin(objectives))])
        return optimal_tau

    def sample(
        self,
        uncertainties: NDArray,
        y_true_cost: float,
        y_proxy_cost: float,
        budget: int,
        random_seed: int,
    ) -> Tuple[NDArray, NDArray]:
        """Sample records and draw annotation indicators under the active policy.

        Applies the cost-optimal active annotation policy (Proposition 2). The
        optimal clipping threshold τ* is found by a grid search over the empirical
        distribution of ``uncertainties`` values. Per-record probabilities
        ``π_i = π_clip(x_i; τ*)`` are then used to determine how many records can
        be afforded within ``budget`` and to draw Bernoulli annotation indicators.

        Parameters
        ----------
        uncertainties : NDArray
            1D float array of shape ``(N,)`` containing the pre-computed conditional
            MSE estimate ``u(x_i) = E[(H - G)² | X = x_i]`` for each record.
            All values must be non-negative and not all zero.
        y_true_cost : float
            Cost of one expensive-rater annotation. Must be strictly positive.
        y_proxy_cost : float
            Cost of one cheap-proxy annotation. Must be strictly positive.
        budget : int
            Total annotation budget. Must be strictly positive.
        random_seed : int
            Seed for ``numpy.random.default_rng``. Mandatory for reproducibility.

        Returns
        -------
        Tuple[NDArray, NDArray]
            [0]: array of shape ``(N,)``, ``pi`` with per-record annotation probabilities
            for selected records and ``0.0`` for unselected records.
            [1]: array of shape ``(N,)``, ``xi`` with Bernoulli indicators:
            ``1.0`` if the expensive rater was queried, ``0.0`` if only the proxy
            was used, ``NaN`` if the record was not selected.

        Raises
        ------
        ValueError
            If ``fit()`` has not been called, if any cost or budget argument is
            non-positive, if ``uncertainties`` contains NaN or negative values,
            if all uncertainty values are zero, or if the budget is too small to
            afford a single record.

        Examples
        --------
        >>> import numpy as np
        >>> from glide.samplers import CostOptimalSampler
        >>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
        >>> uncertainties = np.array([0.1, 0.4, 0.1, 0.4])
        >>> sampler = CostOptimalSampler().fit(y_true)
        >>> pi, xi = sampler.sample(uncertainties, y_true_cost=10.0, y_proxy_cost=1.0, budget=5, random_seed=0)
        >>> (pi == 0.0).sum() == np.isnan(xi).sum()
        True
        >>> np.all(~np.isnan(xi[pi > 0]))
        True
        """
        if not hasattr(self, "_y_true_variance"):
            raise ValueError("Call fit() before sample().")
        if y_true_cost <= 0.0:
            raise ValueError(f"'y_true_cost' must be strictly positive; got {y_true_cost}.")
        if y_proxy_cost <= 0.0:
            raise ValueError(f"'y_proxy_cost' must be strictly positive; got {y_proxy_cost}.")
        if budget <= 0:
            raise ValueError(f"'budget' must be strictly positive; got {budget}.")
        if np.any(np.isnan(uncertainties)):
            raise ValueError("'uncertainties' contains NaN values.")
        if np.any(uncertainties < 0.0):
            raise ValueError("'uncertainties' contains negative values.")
        if np.all(uncertainties == 0.0):
            raise ValueError("All values in 'uncertainties' are zero.")

        tau_star = self._find_optimal_threshold(uncertainties, y_true_cost, y_proxy_cost)
        gamma_star = self._compute_gamma(tau_star, uncertainties, y_true_cost, y_proxy_cost)
        pi_all = self._compute_per_record_probabilities(tau_star, gamma_star, uncertainties)

        cost_per_record = y_true_cost * np.mean(pi_all) + y_proxy_cost
        T = math.floor(budget / cost_per_record)
        if T < 1:
            raise ValueError(
                f"Budget {budget} is too small to afford a single record at cost_per_record={cost_per_record:.4f}."
            )

        N = len(uncertainties)
        rng = np.random.default_rng(random_seed)
        if T < N:
            indices = np.sort(rng.choice(N, size=T, replace=False))
        else:
            indices = np.arange(N)

        pi = np.zeros(N)
        xi = np.full(N, np.nan)
        pi[indices] = pi_all[indices]
        xi[indices] = rng.binomial(n=1, p=pi_all[indices], size=len(indices)).astype(float)

        return pi, xi
