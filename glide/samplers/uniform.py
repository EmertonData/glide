from typing import Optional, Union

import numpy as np
from numpy.random.bit_generator import SeedSequence
from numpy.typing import NDArray

from glide.core.validation import _validate_is_integer, _validate_n_samples_bound, _validate_strictly_positive


class UniformSampler:
    """Sampler that draws observations uniformly without replacement from the pool.

    It is the standard approach when no auxiliary signal is available.

    Examples
    --------
    >>> from glide.samplers import UniformSampler
    >>> sampler = UniformSampler()
    >>> import numpy as np
    >>> y_proxy = np.array([0.0, 1.0])
    >>> xi = sampler.sample(y_proxy, n_samples=1, random_seed=0)
    >>> xi
    array([0., 1.])
    """

    def sample(
        self,
        y_proxy: NDArray,
        n_samples: int,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ) -> NDArray:
        """Sample observations uniformly at random without replacement.

        Selects exactly ``n_samples`` observations from the pool described by
        ``y_proxy`` without replacement.

        Parameters
        ----------
        y_proxy : NDArray
            Proxy labels for all observations in the pool, of shape ``(n_total,)``
            or ``(n_total, n_proxies)``. Only the number of observations is used.
        n_samples : int
            Exact number of observations to select. Must be a strictly
            positive integer and must not exceed the number of observations in
            ``y_proxy``.
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
            If ``n_samples`` is not a strictly positive integer, or if
            ``n_samples`` exceeds the number of observations in ``y_proxy``.
        """
        n_total = len(y_proxy)
        _validate_is_integer(n_samples, "n_samples")
        _validate_strictly_positive(n_samples, "n_samples")
        _validate_n_samples_bound(n_samples, n_total)

        rng = np.random.default_rng(random_seed)

        selected_indices = rng.choice(n_total, size=n_samples, replace=False)
        xi = np.zeros(n_total)
        xi[selected_indices] = 1.0

        return xi
