from typing import Optional, Tuple, Union

import numpy as np
from numpy.random.bit_generator import SeedSequence
from numpy.typing import NDArray


class ActiveSampler:
    """Sampler that draws elements with probabilities based on uncertainty scores.

    Implements active sampling for inference pipelines which support inverse
    probability weighting (IPW).
    Each observation is assigned a drawing probability π_i proportional to its
    uncertainty score, then independently selected via a Bernoulli trial. This
    concentrates the annotation budget on the most uncertain observations.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.samplers.active import ActiveSampler
    >>> uncertainties = np.array([0.1, 0.4])
    >>> sampler = ActiveSampler()
    >>> pi, xi = sampler.sample(uncertainties, budget=1, random_seed=0)
    >>> pi
    array([0.2, 0.8])
    >>> xi
    array([0., 1.])
    """

    def _validate(self, uncertainties: NDArray) -> None:
        if np.any(np.isnan(uncertainties)):
            raise ValueError(
                "All uncertainty values must be finite; got a NaN value. "
                "A NaN uncertainty score cannot be used to compute sampling probabilities."
            )
        if np.any(uncertainties <= 0.0):
            raise ValueError(
                "All uncertainty values must be strictly positive; got a non-positive value. "
                "An observation with zero or negative uncertainty would never be selected."
            )

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
        if (not isinstance(budget, (int, np.integer))) or isinstance(budget, bool) or budget <= 0:
            raise ValueError(f"'budget' must be a strictly positive integer; got {budget!r}.")
        if budget > len(uncertainties):
            raise ValueError(
                f"'budget' must not exceed the number of samples; "
                f"got budget={budget} but uncertainties has {len(uncertainties)} elements."
            )

        self._validate(uncertainties)
        rng = np.random.default_rng(random_seed)

        drawing_probabilities = budget * uncertainties / uncertainties.sum()
        # Cap at 1: a Bernoulli probability cannot exceed 1.
        clipped_probabilities = np.minimum(drawing_probabilities, 1.0)
        indicators = rng.binomial(n=1, p=clipped_probabilities).astype(float)

        return clipped_probabilities, indicators
