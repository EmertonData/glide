from typing import Optional, Union

import numpy as np
from numpy.random.bit_generator import SeedSequence
from numpy.typing import NDArray

from glide.core.validation import _validate_is_integer, _validate_strictly_positive


class UniformSampler:
    """Sampler that draws observations uniformly without replacement from the pool.

    It is the standard approach when no auxiliary signal is available.

    Examples
    --------
    >>> from glide.samplers import UniformSampler
    >>> sampler = UniformSampler()
    >>> xi = sampler.sample(n_total=2, n_samples=1, random_seed=0)
    >>> xi
    array([0., 1.])
    """

    def sample(
        self,
        n_total: int,
        n_samples: int,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ) -> NDArray:
        """Sample observations uniformly at random without replacement.

        Selects exactly ``n_samples`` observations from a pool of ``n_total``
        without replacement.

        Parameters
        ----------
        n_total : int
            Total number of observations in the pool. Must be a strictly
            positive integer.
        n_samples : int
            Exact number of observations to select. Must be a strictly
            positive integer and must not exceed ``n_total``.
        random_seed : int or SeedSequence or None, optional
            Random seed passed to ``numpy.random.default_rng`` for
            reproducibility. Pass ``None`` (the default) to use a
            non-deterministic seed.

        Returns
        -------
        NDArray
            Array of shape ``(n_total,)`` with selection indicators
            (1 if selected for annotation, 0 otherwise).

        Raises
        ------
        ValueError
            If ``n_total`` or ``n_samples`` is not a strictly positive integer,
            or if ``n_samples`` exceeds ``n_total``.
        """
        _validate_is_integer(n_total, "n_total")
        _validate_strictly_positive(n_total, "n_total")
        _validate_is_integer(n_samples, "n_samples")
        _validate_strictly_positive(n_samples, "n_samples")
        if n_samples > n_total:
            raise ValueError(f"'n_samples' must not exceed 'n_total'; got n_samples={n_samples} but n_total={n_total}.")

        rng = np.random.default_rng(random_seed)

        selected_indices = rng.choice(n_total, size=n_samples, replace=False)
        xi = np.zeros(n_total)
        xi[selected_indices] = 1.0

        return xi
