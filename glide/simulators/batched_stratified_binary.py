from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from glide.core.validation import _validate_equal_lengths, _validate_non_empty
from glide.simulators.stratified_binary import generate_stratified_binary_dataset


def generate_batched_stratified_binary_dataset(
    n_samples: Sequence[ArrayLike],
    true_mean: Sequence[ArrayLike],
    proxy_mean: Sequence[ArrayLike],
    correlation: Sequence[ArrayLike],
    random_seed: Optional[int] = None,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Generate a synthetic batched, stratified binary-label oracle dataset.

    Generalizes ``generate_stratified_binary_dataset`` with an outer batch axis: batch
    ``t`` gets its own call to ``generate_stratified_binary_dataset``, using
    ``n_samples[t]``, ``true_mean[t]``, ``proxy_mean[t]``, and ``correlation[t]`` as
    that batch's per-stratum parameters, so the number of strata may vary from batch
    to batch. Batches are concatenated in order, oldest first.

    Parameters
    ----------
    n_samples : list of array_like
        Length ``T`` (number of batches). Entry ``t`` is the ``n_samples`` argument
        passed to ``generate_stratified_binary_dataset`` for batch ``t``, of length
        ``K_t`` (that batch's number of strata).
    true_mean : list of array_like
        Length ``T``. Entry ``t`` is that batch's ``true_mean`` argument, of length ``K_t``.
    proxy_mean : list of array_like
        Length ``T``. Entry ``t`` is that batch's ``proxy_mean`` argument, of length ``K_t``.
    correlation : list of array_like
        Length ``T``. Entry ``t`` is that batch's ``correlation`` argument, of length ``K_t``.
    random_seed : int, optional
        Seed for reproducibility. If provided, one independent child seed is derived
        deterministically per batch.

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray, NDArray]
        Let ``N`` be the total number of samples across all batches and strata.

        [0]: array of shape ``(N,)``, y_true containing ground-truth labels.
        [1]: array of shape ``(N,)``, y_proxy containing proxy labels.
        [2]: array of shape ``(N,)``, batch identifiers, grouped into contiguous
             blocks ordered oldest first.
        [3]: array of shape ``(N,)``, stratum identifiers, local to each batch (i.e.
             stratum ``0`` in batch ``0`` and stratum ``0`` in batch ``1`` are
             unrelated unless the caller's parameters treat them consistently).

    Raises
    ------
    ValueError
        - If ``n_samples``, ``true_mean``, ``proxy_mean``, and ``correlation`` have
          different lengths (different number of batches).
        - If fewer than 1 batch is specified.
        - If any batch has an infeasible combination of parameters, or inconsistent
          per-stratum lengths within that batch (see ``generate_stratified_binary_dataset``).

    Examples
    --------
    >>> import numpy as np
    >>> from glide.simulators import generate_batched_stratified_binary_dataset
    >>> y_true, y_proxy, batches, groups = generate_batched_stratified_binary_dataset(
    ...     n_samples=[[6, 8], [6, 8]],
    ...     true_mean=[[0.6, 0.8], [0.6, 0.8]],
    ...     proxy_mean=[[0.5, 0.7], [0.7, 0.9]],
    ...     correlation=[[0.7, 0.7], [0.7, 0.6]],
    ...     random_seed=42,
    ... )
    >>> len(y_true)
    28
    >>> len(np.unique(batches))
    2
    >>> len(np.unique(groups))
    2
    """
    _validate_non_empty(n_samples, "n_samples")
    n_batches = len(n_samples)
    _validate_equal_lengths(
        n_samples, true_mean, proxy_mean, correlation, names=["n_samples", "true_mean", "proxy_mean", "correlation"]
    )

    y_true_per_batch = []
    y_proxy_per_batch = []
    batches_per_batch = []
    groups_per_batch = []

    seed_sequence = np.random.SeedSequence(random_seed)
    seeds = seed_sequence.spawn(n_batches)

    for batch_id in range(n_batches):
        y_true_t, y_proxy_t, groups_t = generate_stratified_binary_dataset(
            n_samples=n_samples[batch_id],
            true_mean=true_mean[batch_id],
            proxy_mean=proxy_mean[batch_id],
            correlation=correlation[batch_id],
            random_seed=seeds[batch_id],
        )
        y_true_per_batch.append(y_true_t)
        y_proxy_per_batch.append(y_proxy_t)
        batches_per_batch.append(np.full_like(y_true_t, batch_id))
        groups_per_batch.append(groups_t)

    y_true = np.hstack(y_true_per_batch)
    y_proxy = np.hstack(y_proxy_per_batch)
    batches = np.hstack(batches_per_batch)
    groups = np.hstack(groups_per_batch)
    return y_true, y_proxy, batches, groups
