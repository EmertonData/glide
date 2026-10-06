from typing import Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from glide.core.validation import _validate_equal_lengths, _validate_non_empty
from glide.simulators.clustered_binary import generate_clustered_binary_dataset


def generate_batched_clustered_binary_dataset(
    n_samples: Sequence[int],
    n_clusters: Sequence[int],
    true_mean: Sequence[float],
    proxy_mean: Sequence[float],
    correlation: Sequence[float],
    within_cluster_diversity: Union[float, Sequence[float]] = 0.9,
    random_seed: Optional[int] = None,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Generate a synthetic batched, clustered binary-label oracle dataset.

    Generalizes ``generate_clustered_binary_dataset`` with an outer batch axis: batch
    ``t`` gets its own call to ``generate_clustered_binary_dataset``, using
    ``n_samples[t]``, ``n_clusters[t]``, ``true_mean[t]``, ``proxy_mean[t]``,
    ``correlation[t]``, and ``within_cluster_diversity[t]`` as that batch's own
    parameters. Cluster identifiers are offset per batch so every cluster in the
    returned array is unique across the whole stream. Batches are concatenated in
    order, oldest first.

    Parameters
    ----------
    n_samples : list of int
        Length ``T`` (number of batches). Entry ``t`` is the ``n_samples`` argument
        passed to ``generate_clustered_binary_dataset`` for batch ``t``.
    n_clusters : list of int
        Length ``T``. Entry ``t`` is that batch's ``n_clusters`` argument.
    true_mean : list of float
        Length ``T``. Entry ``t`` is that batch's ``true_mean`` argument.
    proxy_mean : list of float
        Length ``T``. Entry ``t`` is that batch's ``proxy_mean`` argument.
    correlation : list of float
        Length ``T``. Entry ``t`` is that batch's ``correlation`` argument.
    within_cluster_diversity : float or list of float
        Either a single value shared by every batch, or a list of length ``T`` whose
        entry ``t`` is that batch's ``within_cluster_diversity`` argument.
    random_seed : int, optional
        Seed for reproducibility. If provided, one independent child seed is derived
        deterministically per batch.

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray, NDArray]
        Let ``N`` be the total number of samples across all batches.

        [0]: array of shape ``(N,)``, y_true containing ground-truth labels.
        [1]: array of shape ``(N,)``, y_proxy containing proxy labels.
        [2]: array of shape ``(N,)``, batch identifiers, grouped into contiguous
             blocks ordered oldest first.
        [3]: array of shape ``(N,)``, cluster identifiers, unique across the whole
             stream: no two batches share a cluster identifier.

    Raises
    ------
    ValueError
        - If ``n_samples``, ``n_clusters``, ``true_mean``, ``proxy_mean``,
          ``correlation``, and ``within_cluster_diversity`` have different lengths.
        - If fewer than 1 batch is specified.
        - If any batch has an infeasible combination of parameters (see
          ``generate_clustered_binary_dataset``).

    Examples
    --------
    >>> import numpy as np
    >>> from glide.simulators import generate_batched_clustered_binary_dataset
    >>> y_true, y_proxy, batches, clusters = generate_batched_clustered_binary_dataset(
    ...     n_samples=[10, 10],
    ...     n_clusters=[3, 3],
    ...     true_mean=[0.6, 0.6],
    ...     proxy_mean=[0.5, 0.7],
    ...     correlation=[0.7, 0.7],
    ...     random_seed=42,
    ... )
    >>> len(y_true)
    20
    >>> len(np.unique(batches))
    2
    >>> len(np.unique(clusters))
    6
    """
    _validate_non_empty(n_samples, "n_samples")
    n_batches = len(n_samples)
    if isinstance(within_cluster_diversity, (Sequence, np.ndarray)):
        within_cluster_diversities = within_cluster_diversity
    else:
        within_cluster_diversities = [within_cluster_diversity] * n_batches
    _validate_equal_lengths(
        n_samples,
        n_clusters,
        true_mean,
        proxy_mean,
        correlation,
        within_cluster_diversities,
        names=["n_samples", "n_clusters", "true_mean", "proxy_mean", "correlation", "within_cluster_diversity"],
    )

    y_true_per_batch = []
    y_proxy_per_batch = []
    batches_per_batch = []
    clusters_per_batch = []

    seed_sequence = np.random.SeedSequence(random_seed)
    seeds = seed_sequence.spawn(n_batches)

    cumulative_n_clusters = np.cumsum(n_clusters)
    for batch_id in range(n_batches):
        y_true_t, y_proxy_t, clusters_t = generate_clustered_binary_dataset(
            n_samples=n_samples[batch_id],
            n_clusters=n_clusters[batch_id],
            true_mean=true_mean[batch_id],
            proxy_mean=proxy_mean[batch_id],
            correlation=correlation[batch_id],
            within_cluster_diversity=within_cluster_diversities[batch_id],
            random_seed=seeds[batch_id],
        )
        y_true_per_batch.append(y_true_t)
        y_proxy_per_batch.append(y_proxy_t)
        batches_per_batch.append(np.full_like(y_true_t, batch_id, dtype=np.int64))
        clusters_per_batch.append(clusters_t + cumulative_n_clusters[batch_id] - n_clusters[batch_id])

    y_true = np.hstack(y_true_per_batch)
    y_proxy = np.hstack(y_proxy_per_batch)
    batches = np.hstack(batches_per_batch)
    clusters = np.hstack(clusters_per_batch)
    return y_true, y_proxy, batches, clusters
