from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from glide.core.validation import _validate_equal_lengths, _validate_non_empty
from glide.simulators.oracle_binary import generate_binary_dataset_with_oracle_sampling


def generate_batched_binary_dataset_with_oracle_sampling(
    n_samples: List[int],
    true_mean: List[float],
    proxy_mean: List[float],
    correlation: List[float],
    random_seed: Optional[int] = None,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Generate a synthetic batched binary dataset with oracle sampling probabilities.

    Generalizes ``generate_binary_dataset_with_oracle_sampling`` with an outer batch
    axis: batch ``t`` gets its own call to that function, using ``n_samples[t]``,
    ``true_mean[t]``, ``proxy_mean[t]`` and ``correlation[t]`` as its parameters.
    Batches are concatenated in order, oldest first. All samples have a ground-truth
    label; use ``simulate_annotation`` to mask the unlabeled ones.

    Parameters
    ----------
    n_samples : list of int
        Length ``T`` (number of batches). Entry ``t`` is the number of samples of batch ``t``.
    true_mean : list of float
        Length ``T``. Entry ``t`` is that batch's expected mean of ``y_true_oracle``.
    proxy_mean : list of float
        Length ``T``. Entry ``t`` is that batch's expected mean of ``y_proxy``.
    correlation : list of float
        Length ``T``. Entry ``t`` is that batch's Pearson correlation between
        ``y_true_oracle`` and ``y_proxy``.
    random_seed : int, optional
        Seed for reproducibility. If provided, one independent child seed is derived
        deterministically per batch.

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray, NDArray]
        Let ``N`` be the total number of samples across all batches.

        [0]: array of shape ``(N,)``, y_true_oracle with the full ground-truth labels (no NaN).
        [1]: array of shape ``(N,)``, y_proxy with proxy predictions.
        [2]: array of shape ``(N,)``, uncertainty (oracle uncertainty score) per sample.
        [3]: array of shape ``(N,)``, batch identifiers, grouped into contiguous
             blocks ordered oldest first.

    Raises
    ------
    ValueError
        - If ``n_samples``, ``true_mean``, ``proxy_mean`` and ``correlation`` have different lengths.
        - If fewer than 1 batch is specified.
        - If any batch has an infeasible combination of parameters (see
          ``generate_binary_dataset_with_oracle_sampling``).

    Examples
    --------
    >>> import numpy as np
    >>> from glide.simulators import generate_batched_binary_dataset_with_oracle_sampling
    >>> y_true_oracle, y_proxy, uncertainty, batches = generate_batched_binary_dataset_with_oracle_sampling(
    ...     n_samples=[4, 6],
    ...     true_mean=[0.6, 0.8],
    ...     proxy_mean=[0.5, 0.7],
    ...     correlation=[0.7, 0.7],
    ...     random_seed=42,
    ... )
    >>> len(y_true_oracle)
    10
    >>> len(np.unique(batches))
    2
    """
    _validate_non_empty(n_samples, "n_samples")
    n_batches = len(n_samples)
    _validate_equal_lengths(
        n_samples, true_mean, proxy_mean, correlation, names=["n_samples", "true_mean", "proxy_mean", "correlation"]
    )

    y_true_oracle_per_batch = []
    y_proxy_per_batch = []
    uncertainty_per_batch = []
    batches_per_batch = []

    seed_sequence = np.random.SeedSequence(random_seed)
    seeds = seed_sequence.spawn(n_batches)

    for batch_id in range(n_batches):
        y_true_oracle_t, y_proxy_t, uncertainty_t = generate_binary_dataset_with_oracle_sampling(
            n_samples=n_samples[batch_id],
            true_mean=true_mean[batch_id],
            proxy_mean=proxy_mean[batch_id],
            correlation=correlation[batch_id],
            random_seed=seeds[batch_id],
        )
        y_true_oracle_per_batch.append(y_true_oracle_t)
        y_proxy_per_batch.append(y_proxy_t)
        uncertainty_per_batch.append(uncertainty_t)
        batches_per_batch.append(np.full_like(y_true_oracle_t, batch_id, dtype=np.int64))

    y_true_oracle = np.hstack(y_true_oracle_per_batch)
    y_proxy = np.hstack(y_proxy_per_batch)
    uncertainty = np.hstack(uncertainty_per_batch)
    batches = np.hstack(batches_per_batch)
    return y_true_oracle, y_proxy, uncertainty, batches
