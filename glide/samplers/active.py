from typing import Optional, Tuple, Union

import numpy as np
from numpy.random.bit_generator import SeedSequence
from numpy.typing import NDArray

from glide.core.validation import (
    _validate_budget_bound,
    _validate_is_integer,
    _validate_strictly_positive,
    _validate_uncertainties,
)


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
    >>> pi, xi = sampler.sample(uncertainties, budget=1, random_seed=0)
    >>> pi
    array([0.2, 0.8])
    >>> xi
    array([0., 1.])
    """

    def sample(
        self,
        uncertainties: NDArray,
        budget: int,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ) -> Tuple[NDArray, NDArray]:
        """Sample observations with probability proportional to uncertainty.

        Each observation receives a drawing probability π_i proportional to
        ``uncertainty_i``, normalised so that the raw probabilities sum to
        ``budget`` (the expected number of selected observations). Because each
        π_i must be a valid Bernoulli probability, values are capped at 1 before
        the coin flip; the actual number of selected items is therefore a random
        variable whose expectation equals at most ``budget``.

        The two returned arrays are intended for use with IPW-based downstream estimators.
        ``pi`` holds the per-sample probability of being selected. ``xi`` holds the
        selection indicators for each sample so that a value of 1 means the sample
        should be sent for annotation and a value of 0 means it should not.

        Parameters
        ----------
        uncertainties : NDArray
            Array of shape ``(n_samples,)`` with strictly positive uncertainty scores.
        budget : int
            Expected total number of annotations to collect. Must be a strictly
            positive integer and must not exceed ``len(uncertainties)``.
        random_seed : int or SeedSequence or None, optional
            Random seed passed to ``numpy.random.default_rng`` for
            reproducibility. Pass ``None`` (the default) to use a
            non-deterministic seed.

        Returns
        -------
        Tuple[NDArray, NDArray]
            [0]: array of shape ``(n_samples,)``, pi with drawing probabilities in ``(0, 1]``
            [1]: array of shape ``(n_samples,)``, xi with Bernoulli selection indicators
            (1 if selected for annotation, 0 otherwise)

        Raises
        ------
        ValueError
            If ``budget`` is not a strictly positive integer, if ``budget``
            exceeds ``len(uncertainties)``, or if any uncertainty value is NaN,
            zero, or negative.
        """
        _validate_is_integer(budget, "budget")
        _validate_strictly_positive(budget, "budget")
        _validate_budget_bound(budget, len(uncertainties))
        _validate_uncertainties(uncertainties)
        rng = np.random.default_rng(random_seed)

        pi = budget * uncertainties / uncertainties.sum()
        # Cap at 1: a Bernoulli probability cannot exceed 1.
        pi = np.minimum(pi, 1.0)
        xi = rng.binomial(n=1, p=pi).astype(float)

        return pi, xi
