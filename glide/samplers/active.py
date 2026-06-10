import warnings
from typing import Optional, Tuple, Union

import numpy as np
from numpy.random.bit_generator import SeedSequence
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, minimize

from glide.core.validation import (
    _validate_is_integer,
    _validate_n_samples_bound,
    _validate_strictly_positive,
    _validate_uncertainties,
)
from glide.samplers.core import _build_output, _compute_cutoff_indices, _shuffle


class ActiveSampler:
    """Sampler that draws elements with probabilities based on uncertainty scores.

    Implements active sampling for inference pipelines which support inverse
    probability weighting (IPW).
    Each observation is assigned a drawing probability π_i proportional to its
    uncertainty score, then independently selected via a Bernoulli trial. This
    concentrates the annotation budget on the most uncertain observations.

    References
    ----------
    Zrnic, Tijana, and Emmanuel J. Candès. "Active statistical inference." In Proceedings
    of the 41st International Conference on Machine Learning, pp. 62993-63010. 2024.

    Gligorić, Kristina, Tijana Zrnic, Cinoo Lee, Emmanuel Candes, and Dan Jurafsky.
    "Can unconfident llm annotations be used for confident conclusions?." In Proceedings
    of the 2025 Conference of the Nations of the Americas Chapter of the Association for
    Computational Linguistics: Human Language Technologies (Volume 1: Long Papers),
    pp. 3514-3533. 2025.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.samplers import ActiveSampler
    >>> uncertainties = np.array([0.1, 0.4])
    >>> sampler = ActiveSampler()
    >>> pi, xi = sampler.sample(uncertainties, n_samples=1, random_seed=0)
    >>> pi
    array([0.2, 0.8])
    >>> xi
    array([0., 1.])
    """

    def _compute_probabilities(self, uncertainties: NDArray, n_samples: int) -> NDArray:
        uncertainty_ratio = np.max(uncertainties) / np.min(uncertainties)
        if uncertainty_ratio > 1e3:
            warnings.warn(
                f"Extreme uncertainty ratio detected among samples (max/min={uncertainty_ratio:.2e} > 1e3); "
                "this may cause numerical instability.",
                UserWarning,
            )
        naive_pi = n_samples * uncertainties / uncertainties.sum()
        if np.max(naive_pi) <= 1.0:
            return naive_pi

        n = len(uncertainties)
        squared_uncertainties = np.power(uncertainties, 2)

        def objective(pi: NDArray) -> float:
            result = np.sum(squared_uncertainties / pi)
            return result

        def jacobian(pi: NDArray) -> NDArray:
            gradient = -squared_uncertainties / np.power(pi, 2)
            return gradient

        bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))
        n_samples_constraint = LinearConstraint(np.ones((1, n)), lb=n_samples, ub=n_samples)
        optimization_result = minimize(
            objective,
            naive_pi,
            method="trust-constr",
            jac=jacobian,
            constraints=[n_samples_constraint],
            bounds=bounds,
            options={"maxiter": 100},
        )
        result = np.minimum(optimization_result.x, 1.0)
        return result

    def sample(
        self,
        uncertainties: NDArray,
        n_samples: int,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ) -> Tuple[NDArray, NDArray]:
        """Sample observations with probability proportional to uncertainty.

        Each observation receives a drawing probability π_i that minimizes the variance
        of downstream IPW-based estimators. This is equivalently done by minimizing the sum of
        ``uncertainty_i^2 / π_i`` over all observations. Probabilities are constrained to
        ``(0, 1]`` and sum to ``n_samples``. The actual number of selected items is random
        but limited to ``n_samples``.

        Samples are randomly permuted before drawing and the inverse permutation
        is applied to the output, so the returned arrays are always in the
        original input order. A post-draw cutoff is then applied to strictly
        respect the budget: samples beyond the cutoff are discarded by setting their entries
        in ``pi`` and ``xi`` to ``0.0`` and ``NaN`` respectively.

        The two returned arrays are intended for use with IPW-based downstream estimators.
        ``pi`` holds the per-sample probability of being selected. ``xi`` holds the
        selection indicators for each sample so that a value of 1 means the sample
        should be sent for annotation, a value of 0 means it was not selected, and
        ``NaN`` means it was discarded by the budget cutoff.

        Parameters
        ----------
        uncertainties : NDArray
            Array of shape ``(n_total,)`` with strictly positive uncertainty scores.
        n_samples : int
            Expected total number of annotations to collect. Must be a strictly
            positive integer and must not exceed ``len(uncertainties)``.
        random_seed : int or SeedSequence or None, optional
            Random seed passed to ``numpy.random.default_rng`` for
            reproducibility. Pass ``None`` (the default) to use a
            non-deterministic seed.

        Returns
        -------
        Tuple[NDArray, NDArray]
            [0]: array of shape ``(n_total,)``, ``pi`` with per-sample annotation probabilities
            for selected samples and ``0.0`` for unselected samples.
            [1]: array of shape ``(n_total,)``, ``xi`` with Bernoulli indicators:
            ``1.0`` if selected for annotation, ``0.0`` if not selected,
            ``NaN`` if excluded by the budget cutoff.

        Raises
        ------
        ValueError
            If ``n_samples`` is not a strictly positive integer, if ``n_samples``
            exceeds ``len(uncertainties)``, or if any uncertainty value is NaN,
            zero, or negative.

        Warns
        -----
        UserWarning
            If the ratio of the largest to the smallest uncertainty is extreme,
            indicating potential numerical instability.

        References
        ----------
        Zrnic, Tijana, and Emmanuel J. Candès. "Active statistical inference." In Proceedings
        of the 41st International Conference on Machine Learning, pp. 62993-63010. 2024.
        """
        _validate_is_integer(n_samples, "n_samples")
        _validate_strictly_positive(n_samples, "n_samples")
        _validate_n_samples_bound(n_samples, len(uncertainties))
        _validate_uncertainties(uncertainties)
        pi = self._compute_probabilities(uncertainties, n_samples)

        rng = np.random.default_rng(random_seed)
        pi_shuffled, order = _shuffle(pi, rng)
        xi_shuffled = rng.binomial(n=1, p=pi_shuffled).astype(float)
        cumulative_costs = np.cumsum(xi_shuffled)
        kept_indices = _compute_cutoff_indices(cumulative_costs, order, n_samples)
        pi_out, xi_out = _build_output(kept_indices, pi_shuffled, xi_shuffled)
        return pi_out, xi_out
