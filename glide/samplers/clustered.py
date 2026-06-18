from typing import Optional, Union

import numpy as np
from numpy.random.bit_generator import SeedSequence
from numpy.typing import NDArray

from glide.core.validation import _validate_bounds, _validate_is_integer, _validate_strictly_positive


class UniformClusteredSampler:
    """Sampler that selects entire clusters without replacement using uniform sampling.

    Each call to ``sample`` draws a fixed number of clusters from the pool of unique
    cluster labels in ``clusters``, then marks every observation in a selected cluster
    for annotation. Every cluster has equal probability of being selected, so every
    individual observation has the same marginal probability of being annotated.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.samplers import UniformClusteredSampler
    >>> clusters = np.array(["A", "A", "B", "B"], dtype=object)
    >>> sampler = UniformClusteredSampler()
    >>> xi = sampler.sample(clusters, n_clusters=1, random_seed=0)
    >>> xi
    array([0, 0, 1, 1])
    """

    def sample(
        self,
        clusters: NDArray,
        n_clusters: int,
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ) -> NDArray:
        """Select entire clusters without replacement.

        Draws ``n_clusters`` clusters from the unique values of ``clusters`` with equal
        probability and returns selection indicators: every observation whose cluster was
        drawn receives a 1, all others receive a 0.

        Parameters
        ----------
        clusters : NDArray
            Array of shape ``(n_samples,)`` with cluster identifiers for all observations.
        n_clusters : int
            Number of clusters to select. Must be a strictly positive integer and must
            not exceed the number of unique clusters in ``clusters``.
        random_seed : int or SeedSequence or None, optional
            Random seed passed to ``numpy.random.default_rng`` for reproducibility.
            Defaults to ``None`` (non-deterministic).

        Returns
        -------
        NDArray
            Selection indicators of shape ``(n_samples,)``: 1 if the observation belongs
            to a selected cluster, 0 otherwise.

        Raises
        ------
        ValueError
            - If ``n_clusters`` is not a strictly positive integer.
            - If ``n_clusters`` exceeds the number of unique clusters in ``clusters``.
        """
        _validate_is_integer(n_clusters, "n_clusters")
        _validate_strictly_positive(n_clusters, "n_clusters")
        unique_clusters = np.unique(clusters)
        n_total_clusters = len(unique_clusters)
        _validate_bounds(
            n_clusters,
            "n_clusters",
            upper=n_total_clusters,
            error_message=f"'n_clusters' must not exceed the number of unique clusters; "
            f"got n_clusters={n_clusters} but there are only {n_total_clusters} unique clusters.",
        )

        rng = np.random.default_rng(random_seed)
        selected_clusters = rng.choice(unique_clusters, size=n_clusters, replace=False)

        xi = np.isin(clusters, selected_clusters).astype(int)
        return xi
