from typing import Literal, Optional, Union

import numpy as np
from numpy.random.bit_generator import SeedSequence
from numpy.typing import NDArray

from glide.core.validation import (
    _validate_is_integer,
    _validate_literal,
    _validate_strictly_positive,
)


class ClusterSampler:
    """Sampler that selects entire clusters without replacement.

    Each call to ``sample`` draws a fixed number of clusters from the pool of unique
    cluster labels in ``clusters``, then marks every observation in a selected cluster
    for annotation. Two strategies control how clusters are drawn:

    - **Uniform** (default): every cluster has equal probability of being selected,
      so every individual observation has the same marginal probability of being annotated.
    - **Proportional**: each cluster is drawn with probability proportional to its size,
      so larger clusters are more likely to be selected and their observations more likely
      to be annotated.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.samplers import ClusterSampler
    >>> clusters = np.array(["A", "A", "B", "B"], dtype=object)
    >>> sampler = ClusterSampler()
    >>> xi = sampler.sample(clusters, n_clusters=1, random_seed=0)
    >>> xi
    array([0, 0, 1, 1])
    """

    def _validate(self, clusters: NDArray, n_clusters: int, strategy: str) -> None:
        _validate_is_integer(n_clusters, "n_clusters")
        _validate_strictly_positive(n_clusters, "n_clusters")
        _validate_literal(strategy, "strategy", ["uniform", "proportional"])
        n_total_clusters = len(np.unique(clusters))
        if n_clusters > n_total_clusters:
            raise ValueError(
                f"'n_clusters' must not exceed the number of unique clusters; "
                f"got n_clusters={n_clusters} but there are only {n_total_clusters} unique clusters."
            )

    def sample(
        self,
        clusters: NDArray,
        n_clusters: int,
        strategy: Literal["uniform", "proportional"] = "uniform",
        random_seed: Optional[Union[int, SeedSequence]] = None,
    ) -> NDArray:
        """Select entire clusters without replacement.

        Draws ``n_clusters`` clusters from the unique values of ``clusters`` and returns
        selection indicators: every observation whose cluster was drawn receives a 1,
        all others receive a 0.

        Parameters
        ----------
        clusters : NDArray
            Cluster identifiers for all observations, shape ``(n_samples,)``.
        n_clusters : int
            Number of clusters to select. Must be a strictly positive integer and must
            not exceed the number of unique clusters in ``clusters``.
        strategy : str, optional
            Cluster-selection strategy. ``"uniform"`` draws clusters with equal
            probability, giving every observation the same marginal probability of being
            annotated (default). ``"proportional"`` draws with probability proportional
            to cluster size, so larger clusters are more likely to be selected.
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
            - If ``strategy`` is not recognised.
        """
        self._validate(clusters, n_clusters, strategy)

        unique_clusters, cluster_sizes = np.unique(clusters, return_counts=True)
        rng = np.random.default_rng(random_seed)

        if strategy == "uniform":
            selected_clusters = rng.choice(unique_clusters, size=n_clusters, replace=False)
        else:
            probabilities = cluster_sizes / len(clusters)
            selected_clusters = rng.choice(unique_clusters, size=n_clusters, replace=False, p=probabilities)

        xi = np.isin(clusters, selected_clusters).astype(int)
        return xi
