from typing import Optional, Tuple, Union

import numpy as np
from numpy.random.bit_generator import SeedSequence
from numpy.typing import NDArray

from glide.core.validation import (
    _validate_non_constant,
    _validate_strictly_positive,
    _validate_uncertainties,
    _validate_y_true_burn_in,
)


class CostOptimalSampler:
    """Sampler that draws elements with optimal probabilities based on uncertainty
    scores and annotation costs on a limited budget.

    Implements a cost-optimal active annotation policy. Each sample is assigned
    an annotation probability proportional to how unreliable the proxy label is
    expected to be for that sample, as measured by the caller-supplied per-sample
    uncertainty scores. Samples with high expected proxy error are more likely to
    be annotated whereas those with low expected proxy error are less likely to be
    annotated. This concentrates the annotation budget where it matters most.

    The caller provides per-sample uncertainty scores and passes them as a
    1D array to ``sample()``. These are treated as oracle root mean square error
    estimates. This class does not learn those scores internally.

    References
    ----------
    Angelopoulos, Anastasios N., Jacob Eisenstein, Jonathan Berant, Alekh
    Agarwal, and Adam Fisch. "Cost-optimal active ai model evaluation." arXiv
    preprint arXiv:2506.07949 (2025).

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
    0.025...
    >>> xi[0]
    np.float64(0.0)
    >>> np.isnan(xi[-1])
    np.True_
    """

    def fit(self, y_true: NDArray) -> "CostOptimalSampler":
        """Estimate the true label variance from a burn-in dataset.

        The true label variance is computed ahead of active sampling so that
        ``sample()`` can derive the cost-optimal annotation probabilities.

        Parameters
        ----------
        y_true : NDArray
            1D float array of true labels from the burn-in phase. Must not
            contain NaN values.

        Returns
        -------
        CostOptimalSampler
            The fitted sampler (returns ``self`` for method chaining).

        Raises
        ------
        ValueError
            If ``y_true`` is empty, contains NaN, or all labels are identical (zero true label variance).

        """
        _validate_y_true_burn_in(y_true)
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
        above_mask = uncertainties > tau
        prob_above = np.mean(above_mask)
        e_u_below = np.mean(uncertainties**2 * ~above_mask)
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
        probabilities = np.where(uncertainties > tau, 1.0, gamma * uncertainties)
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
        error_term = self._y_true_variance + np.mean(uncertainties**2 * (1.0 / pi_values - 1.0))
        objective = cost_term * error_term
        return objective

    def _find_optimal_threshold(
        self,
        uncertainties: NDArray,
        y_true_cost: float,
        y_proxy_cost: float,
    ) -> float:
        candidates = np.unique(uncertainties)
        if y_proxy_cost == 0:
            # When y_proxy_cost=0, cost_ratio=0 in _compute_gamma, so gamma = sqrt(prob_above / ...).
            # If tau equals the largest uncertainty, no sample exceeds it, so prob_above=0 and gamma=0.
            # gamma=0 makes every pi value 0, which causes division by zero in _compute_objective.
            # Dropping the largest candidate ensures at least one sample always exceeds tau (prob_above > 0).
            candidates = candidates[:-1]
        objectives = [self._compute_objective(tau, uncertainties, y_true_cost, y_proxy_cost) for tau in candidates]
        optimal_tau = candidates[np.argmin(objectives)]
        return optimal_tau

    def sample(
        self,
        uncertainties: NDArray,
        y_true_cost: float,
        y_proxy_cost: float,
        budget: float,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ) -> Tuple[NDArray, NDArray]:
        """Compute sampling probabilities and draw annotation indicators under the cost
        optimal policy.

        Per-sample annotation probabilities are derived from the supplied uncertainty
        scores (root mean squared errors) and the true label variance estimated by ``fit()``.
        When the budget is tight, samples beyond a certain index in the input array are
        excluded from sampling and receive a probability of zero.

        The two returned arrays are intended for use with IPW-based downstream estimators. ``pi``
        holds the per-sample probability of querying the expensive rater. ``xi`` holds the
        annotation indicators for selected samples, with NaN marking unselected samples that
        should be discarded before running an estimator.

        Parameters
        ----------
        uncertainties : NDArray
            1D float array of shape ``(n_samples,)`` containing the pre-computed per-sample
            root mean squared error of the proxy label. All values must be strictly positive.
        y_true_cost : float
            Cost of one true label. Must be strictly positive.
        y_proxy_cost : float
            Cost of one proxy label. Must be non-negative.
        budget : float
            Total annotation budget in cost units. Must be strictly positive.
        random_seed : int or SeedSequence or None, optional
            Random seed passed to ``numpy.random.default_rng`` for reproducibility.
            Pass ``None`` (the default) to use a non-deterministic seed.

        Returns
        -------
        Tuple[NDArray, NDArray]
            [0]: array of shape ``(n_samples,)``, ``pi`` with per-sample annotation probabilities
            for selected samples and ``0.0`` for unselected samples.
            [1]: array of shape ``(n_samples,)``, ``xi`` with Bernoulli indicators:
            ``1.0`` if the true label was requested, ``0.0`` if only the proxy
            was used, ``NaN`` if the sample was not selected.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called before ``sample()``.
        ValueError
            - If ``y_true_cost`` is not strictly positive or ``y_proxy_cost`` is negative.
            - If ``budget`` is not strictly positive.
            - If any uncertainty value is NaN or non-positive.
            - If all uncertainty values are equal and ``y_proxy_cost`` is zero.
            - If ``budget`` is too small to afford a single sample.

        """
        if not hasattr(self, "_y_true_variance"):
            raise RuntimeError("Call fit() before sample().")
        _validate_strictly_positive(y_true_cost, "y_true_cost")
        if y_proxy_cost < 0.0:
            raise ValueError(f"'y_proxy_cost' must be non-negative; got {y_proxy_cost}.")
        if y_proxy_cost == 0.0:
            _validate_non_constant(
                uncertainties,
                "All uncertainty values are equal and 'y_proxy_cost' is zero."
                " Provide non-constant uncertainties or set 'y_proxy_cost' to a positive value.",
            )
        _validate_strictly_positive(budget, "budget")
        _validate_uncertainties(uncertainties)

        tau_star = self._find_optimal_threshold(uncertainties, y_true_cost, y_proxy_cost)
        gamma_star = self._compute_gamma(tau_star, uncertainties, y_true_cost, y_proxy_cost)
        pi_all = self._compute_per_sample_probabilities(tau_star, gamma_star, uncertainties)

        cumulative_costs = np.cumsum(y_true_cost * pi_all + y_proxy_cost)
        n_affordable = np.searchsorted(cumulative_costs, budget, side="right")
        if n_affordable < 1:
            raise ValueError(
                f"'budget' is too small to afford a single sample; got {budget}."
                " Increase 'budget' or reduce 'y_true_cost'."
            )

        n_samples = len(uncertainties)
        cutoff = min(n_affordable, n_samples)

        rng = np.random.default_rng(random_seed)

        pi = np.zeros(n_samples)
        xi = np.full(n_samples, np.nan)
        pi[:cutoff] = pi_all[:cutoff]
        xi[:cutoff] = rng.binomial(n=1, p=pi_all[:cutoff], size=cutoff).astype(float)

        return pi, xi
