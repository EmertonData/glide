from typing import Optional, Union

import numpy as np
from numpy.random.bit_generator import SeedSequence
from numpy.typing import NDArray


class UniformSampler:
    """Sampler that draws observations uniformly without replacement from the pool.

    It is the standard approach when no auxiliary signal is available.

    Examples
    --------
    >>> from glide.samplers import UniformSampler
    >>> sampler = UniformSampler()
    >>> xi = sampler.sample(n_samples=2, budget=1, random_seed=0)
    >>> xi
    array([0., 1.])
    """

    def sample(
        self,
        n_samples: int,
        budget: int,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ) -> NDArray:
        """Sample observations uniformly at random without replacement.

        Selects exactly ``budget`` observations from a pool of ``n_samples``
        via ``numpy.random.Generator.choice`` without replacement.

        Parameters
        ----------
        n_samples : int
            Total number of observations in the pool. Must be a strictly
            positive integer.
        budget : int
            Exact number of observations to select. Must be a strictly
            positive integer and must not exceed ``n_samples``.
        random_seed : int or SeedSequence or None, optional
            Random seed passed to ``numpy.random.default_rng`` for
            reproducibility. Pass ``None`` (the default) to use a
            non-deterministic seed.

        Returns
        -------
        NDArray
            Array of shape ``(n_samples,)`` with selection indicators
            (1 if selected for annotation, 0 otherwise).

        Raises
        ------
        ValueError
            If ``n_samples`` or ``budget`` is not a strictly positive integer,
            or if ``budget`` exceeds ``n_samples``.
        """
        if (not isinstance(n_samples, (int, np.integer))) or isinstance(n_samples, bool) or n_samples <= 0:
            raise ValueError(f"'n_samples' must be a strictly positive integer; got {n_samples!r}.")
        if (not isinstance(budget, (int, np.integer))) or isinstance(budget, bool) or budget <= 0:
            raise ValueError(f"'budget' must be a strictly positive integer; got {budget!r}.")
        if budget > n_samples:
            raise ValueError(f"'budget' must not exceed 'n_samples'; got budget={budget} but n_samples={n_samples}.")

        rng = np.random.default_rng(random_seed)

        selected_indices = rng.choice(n_samples, size=budget, replace=False)
        xi = np.zeros(n_samples)
        xi[selected_indices] = 1.0

        return xi
