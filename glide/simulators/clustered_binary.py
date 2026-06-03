from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from glide.simulators.binary import generate_binary_dataset


def generate_clustered_binary_dataset(
    n_total: int,
    n_clusters: int,
    true_mean: float = 0.7,
    proxy_mean: float = 0.6,
    correlation: float = 0.8,
    random_seed: Optional[Union[int, np.random.SeedSequence]] = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Generate a synthetic clustered binary-label dataset for evaluation.

    Draws ``n_total`` i.i.d. ``(y_true_oracle, y_proxy)`` pairs from the
    joint binary distribution defined by ``true_mean``, ``proxy_mean``, and
    ``correlation``, then randomly partitions the observations into
    ``n_clusters`` non-empty groups. Returns oracle arrays with no NaN
    masking.

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
    random_seed : int or np.random.SeedSequence, optional
        Seed for reproducibility.

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray]
        [0]: ``y_true_oracle`` — shape ``(n_total,)``, all values in
             ``{0.0, 1.0}``, no NaN.
        [1]: ``y_proxy`` — shape ``(n_total,)``, all values in
             ``{0.0, 1.0}``, no NaN.
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
        ``correlation`` leads to negative joint probabilities.
    ValueError
        If ``n_clusters < 2``.
    ValueError
        If ``n_total < n_clusters``.

    Notes
    -----
    **Step 1 — Draw observations**

    Call ``generate_binary_dataset(n_total, ...)`` to obtain ``n_total``
    i.i.d. ``(y_true_oracle, y_proxy)`` pairs from the joint binary
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

    Examples
    --------
    >>> import numpy as np
    >>> from glide.simulators import generate_clustered_binary_dataset
    >>> y_true_oracle, y_proxy, clusters = generate_clustered_binary_dataset(
    ...     n_total=20, n_clusters=4, random_seed=0
    ... )
    >>> len(np.unique(clusters))
    4
    >>> bool(np.all(~np.isnan(y_true_oracle)))
    True
    >>> bool(np.all(~np.isnan(y_proxy)))
    True
    """
    if n_clusters < 2:
        raise ValueError(f"'n_clusters' must be >= 2; got {n_clusters}.")
    if n_total < n_clusters:
        raise ValueError(f"'n_total' must be >= 'n_clusters'; got n_total={n_total} and n_clusters={n_clusters}.")

    if isinstance(random_seed, np.random.SeedSequence):
        seed_sequence = random_seed
    else:
        seed_sequence = np.random.SeedSequence(random_seed)
    data_seed, partition_seed = seed_sequence.spawn(2)

    y_true_oracle, y_proxy = generate_binary_dataset(
        n_total=n_total,
        true_mean=true_mean,
        proxy_mean=proxy_mean,
        correlation=correlation,
        random_seed=data_seed,
    )

    rng = np.random.default_rng(partition_seed)
    cut_positions = np.sort(rng.choice(np.arange(1, n_total), size=n_clusters - 1, replace=False))
    interval_lengths = np.diff(np.hstack([[0], cut_positions, [n_total]]))
    clusters = np.repeat(np.arange(n_clusters, dtype=np.int64), interval_lengths)
    rng.shuffle(clusters)

    return y_true_oracle, y_proxy, clusters
