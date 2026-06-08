from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from glide.core.validation import _validate_bounds
from glide.simulators.binary import generate_binary_dataset


def generate_clustered_binary_dataset(
    n_total: int,
    n_clusters: int,
    true_mean: float = 0.7,
    proxy_mean: float = 0.6,
    correlation: float = 0.8,
    cluster_source_fraction: float = 0,
    random_seed: Optional[Union[int, np.random.SeedSequence]] = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Generate a synthetic clustered binary-label dataset for evaluation.

    Draws ``n_total`` i.i.d. ``(y_true, y_proxy)`` pairs from the
    joint binary distribution defined by ``true_mean``, ``proxy_mean``, and
    ``correlation``, then randomly partitions the observations into
    ``n_clusters`` non-empty groups.

    Parameters
    ----------
    n_total : int
        Exact total number of observations across all clusters.
    n_clusters : int
        Exact number of clusters. Must be at least 2.
    true_mean : float
        Expected mean value of the true labels. Must be in ``(0, 1)``.
    proxy_mean : float
        Expected mean value of the proxy labels. Must be in ``(0, 1)``.
    correlation : float
        Pearson correlation between true and proxy labels.
    cluster_source_fraction : float
        Fraction of observations per cluster used as source draws. The
        remaining observations have their labels resampled from those sources,
        inducing within-cluster label repetition. Must be in ``[0, 1]``. A
        value of ``0`` enforces maximum homogeneity (1 source per cluster); a
        value of ``1`` leaves all labels unchanged.
    random_seed : int or np.random.SeedSequence, optional
        Seed for reproducibility.

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray]
        [0]: ``y_true`` — shape ``(n_total,)``, values in ``{0.0, 1.0}``.
        [1]: ``y_proxy`` — shape ``(n_total,)``, values in ``{0.0, 1.0}``.
        [2]: ``clusters`` — shape ``(n_total,)``, integer cluster
             identifiers in ``{0, 1, ..., n_clusters - 1}``.

    Raises
    ------
    ValueError
        If ``true_mean`` is not in ``(0, 1)``.
    ValueError
        If ``proxy_mean`` is not in ``(0, 1)``.
    ValueError
        If the combination of ``true_mean``, ``proxy_mean``, and
        ``correlation`` is impossible (leads to negative joint probabilities).
    ValueError
        If ``n_clusters < 2``.
    ValueError
        If ``n_total < n_clusters``.

    Notes
    -----
    **Step 1 — Draw observations**

    Call ``generate_binary_dataset(n_total, ...)`` to obtain ``n_total``
    i.i.d. ``(y_true, y_proxy)`` pairs from the joint binary
    distribution defined by ``true_mean``, ``proxy_mean``, and
    ``correlation``.

    **Step 2 — Random cluster partition**

    Draw ``n_clusters - 1`` cut positions uniformly without replacement from
    ``{1, 2, ..., n_total - 1}`` and sort them. Combined with ``0`` and
    ``n_total``, these define ``n_clusters`` contiguous intervals of random
    lengths that sum to ``n_total``. Assign cluster identifier ``k`` to all
    observations whose position falls in the ``k``-th interval. Every cluster
    contains at least 1 observation by construction.

    **Step 3 — Shuffle**

    Randomly permute the cluster identifier array so that cluster membership is
    not determined by position in the output.

    **Step 4 — Induce within-cluster homogeneity**

    For each cluster, select ``max(1, floor(cluster_source_fraction *
    cluster_size))`` observations as sources. Resample the labels of the
    remaining observations uniformly from those sources, making clusters more
    internally homogeneous.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.simulators import generate_clustered_binary_dataset
    >>> y_true, y_proxy, clusters = generate_clustered_binary_dataset(
    ...     n_total=10, n_clusters=4, random_seed=0
    ... )
    >>> y_true
    array([1., 1., 1., 0., 1., 1., 1., 1., 1., 1.])
    >>> y_proxy
    array([1., 1., 1., 0., 1., 1., 1., 1., 1., 1.])
    >>> clusters
    array([3, 0, 3, 1, 0, 3, 3, 2, 0, 0])
    """
    _validate_bounds(n_clusters, "n_clusters", lower=2, error_message=f"'n_clusters' must be >= 2; got {n_clusters}.")
    _validate_bounds(
        n_total,
        "n_total",
        lower=n_clusters,
        error_message=f"'n_total' must be >= 'n_clusters'; got n_total={n_total} and n_clusters={n_clusters}.",
    )

    if isinstance(random_seed, np.random.SeedSequence):
        seed_sequence = random_seed
    else:
        seed_sequence = np.random.SeedSequence(random_seed)
    data_seed, partition_seed = seed_sequence.spawn(2)

    y_true, y_proxy = generate_binary_dataset(
        n_total=n_total,
        true_mean=true_mean,
        proxy_mean=proxy_mean,
        correlation=correlation,
        random_seed=data_seed,
    )

    rng = np.random.default_rng(partition_seed)

    # cut_positions = np.sort(rng.choice(n_total - 1, size=n_clusters - 1, replace=False) + 1)
    cut_positions = np.array([int(i * n_total / n_clusters) for i in range(1, n_clusters)])
    interval_lengths = np.diff(np.hstack([[0], cut_positions, [n_total]]))
    clusters = np.repeat(np.arange(n_clusters, dtype=np.int64), interval_lengths)
    rng.shuffle(clusters)

    for cluster_id in range(n_clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_size = len(cluster_indices)
        n_kept_samples = max(1, int(cluster_source_fraction * cluster_size))
        source_mask = np.zeros(cluster_size, dtype=bool)
        kept_relative_indices = rng.choice(cluster_size, size=n_kept_samples, replace=False)
        source_mask[kept_relative_indices] = True
        source_indices = cluster_indices[source_mask]
        copy_indices = cluster_indices[~source_mask]
        y_true[copy_indices] = rng.choice(y_true[source_indices], size=len(copy_indices))
        y_proxy[copy_indices] = rng.choice(y_proxy[source_indices], size=len(copy_indices))

    return y_true, y_proxy, clusters


# def generate_clustered_binary_dataset2(
#     n_total: int,
#     n_clusters: int,
#     true_mean: float = 0.7,
#     proxy_mean: float = 0.6,
#     correlation: float = 0.8,
#     cluster_source_fraction: float = 0,
#     variability: Optional[float] = 0.05,
#     random_seed: Optional[Union[int, np.random.SeedSequence]] = None,
# ) -> Tuple[NDArray, NDArray, NDArray]:
#     _validate_bounds(n_clusters, "n_clusters", lower=2, error_message=f"'n_clusters' must be >= 2; got {n_clusters}.")
#     _validate_bounds(
#         n_total,
#         "n_total",
#         lower=n_clusters,
#         error_message=f"'n_total' must be >= 'n_clusters'; got n_total={n_total} and n_clusters={n_clusters}.",
#     )

#     if isinstance(random_seed, np.random.SeedSequence):
#         seed_sequence = random_seed
#     else:
#         seed_sequence = np.random.SeedSequence(random_seed)
#     data_seed, partition_seed = seed_sequence.spawn(2)

#     rng = np.random.default_rng(partition_seed)

#     # cut_positions = np.sort(rng.choice(n_total - 1, size=n_clusters - 1, replace=False) + 1)
#     cut_positions = np.array([int(i * n_total / n_clusters) for i in range(1, n_clusters)])
#     cluster_sizes = np.diff(np.hstack([[0], cut_positions, [n_total]]))

#     variability = min(true_mean, proxy_mean, 1 - true_mean, 1 - proxy_mean, variability or 1)

#     offsets = variability * (2 * rng.uniform(0, 1, size=n_clusters) - 1)
#     true_means = true_mean + offsets
#     proxy_means = proxy_mean + offsets

#     y_true, y_proxy, clusters = generate_stratified_binary_dataset(
#         n_total=cluster_sizes,
#         true_mean=true_means,
#         proxy_mean=proxy_means,
#         correlation=[correlation] * n_clusters,
#         random_seed=data_seed,
#     )

#     for cluster_id in range(n_clusters):
#         cluster_indices = np.where(clusters == cluster_id)[0]
#         cluster_size = len(cluster_indices)
#         n_kept_samples = max(1, int(cluster_source_fraction * cluster_size))
#         source_mask = np.zeros(cluster_size, dtype=bool)
#         kept_relative_indices = rng.choice(cluster_size, size=n_kept_samples, replace=False)
#         source_mask[kept_relative_indices] = True
#         source_indices = cluster_indices[source_mask]
#         copy_indices = cluster_indices[~source_mask]
#         y_true[copy_indices] = rng.choice(y_true[source_indices], size=len(copy_indices))
#         y_proxy[copy_indices] = rng.choice(y_proxy[source_indices], size=len(copy_indices))

#     permutation = rng.permutation(n_total)
#     y_true = y_true[permutation]
#     y_proxy = y_proxy[permutation]
#     clusters = clusters[permutation]

#     return y_true, y_proxy, clusters
