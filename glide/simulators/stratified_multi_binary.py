from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from glide.core.validation import _validate_equal_lengths, _validate_non_empty
from glide.simulators.multi_binary import generate_multi_binary_dataset


def generate_stratified_multi_binary_dataset(
    n_samples: ArrayLike,
    true_mean: ArrayLike,
    proxy_means: ArrayLike,
    correlations: ArrayLike,
    random_seed: Optional[int] = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Generate a synthetic stratified binary-label oracle dataset with multiple proxy models.

    Combines ``M >= 1`` proxy models (see ``generate_multi_binary_dataset``) with ``K >= 1``
    strata (see ``generate_stratified_binary_dataset``). This enables simulation of
    heterogeneous, multi-proxy data, e.g. a batched production stream monitored against
    several models at once, where both the true/proxy relationship and the proxy panel
    itself can vary across strata.

    Parameters
    ----------
    n_samples : list of int or NDArray of shape (K,)
        Total number of samples per stratum.
        Length must equal the number of strata.
    true_mean : list of float or NDArray of shape (K,)
        Expected mean value of the true labels per stratum.
        Length must equal the number of strata.
    proxy_means : list of list of float, or NDArray of shape (K, M)
        Expected mean value of each proxy label, per stratum.
        Row ``k`` holds stratum ``k``'s mean for each of the ``M`` proxies;
        ``M`` (the number of proxies) is fixed across all strata.
    correlations : list of list of float, or NDArray of shape (K, M)
        Pearson correlation between the true label and each proxy label, per stratum.
        Row ``k`` holds stratum ``k``'s correlation for each of the ``M`` proxies.
    random_seed : int, optional
        Seed for reproducibility. If provided, one independent child seed is derived
        deterministically per stratum.

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray]
        Let ``N = sum(n_samples)`` be the total number of samples across all strata.

        [0]: array of shape ``(N,)``, y_true containing ground-truth labels.
        [1]: array of shape ``(N, M)``, y_proxies where column m contains proxy labels
             for proxy m across all strata.
        [2]: array of shape ``(N,)``, stratum identifiers.

    Raises
    ------
    ValueError
        - If ``n_samples``, ``true_mean``, ``proxy_means``, and ``correlations`` have
          inconsistent lengths (row counts).
        - If fewer than 1 stratum is specified.
        - If ``proxy_means`` or ``correlations`` is not a 2D array.
        - If any stratum has an infeasible combination of ``true_mean``,
          ``proxy_means``, and ``correlations`` (see ``generate_multi_binary_dataset``).

    Examples
    --------
    >>> import numpy as np
    >>> from glide.simulators import generate_stratified_multi_binary_dataset
    >>> y_true, y_proxies, groups = generate_stratified_multi_binary_dataset(
    ...     n_samples=[6, 8],
    ...     true_mean=[0.6, 0.8],
    ...     proxy_means=[[0.5, 0.55], [0.7, 0.75]],
    ...     correlations=[[0.7, 0.65], [0.75, 0.7]],
    ...     random_seed=42,
    ... )
    >>> len(y_true)
    14
    >>> y_proxies.shape
    (14, 2)
    >>> len(groups)
    14
    >>> bool(np.all(np.isin(y_true, [0.0, 1.0])))
    True
    >>> bool(np.all(np.isin(y_proxies, [0.0, 1.0])))
    True
    """
    n_samples_arr = np.asarray(n_samples, dtype=int)
    true_mean_arr = np.asarray(true_mean, dtype=float)
    proxy_means_arr = np.asarray(proxy_means, dtype=float)
    correlations_arr = np.asarray(correlations, dtype=float)

    for array, name in [(proxy_means_arr, "proxy_means"), (correlations_arr, "correlations")]:
        if array.ndim != 2:
            raise ValueError(f"'{name}' must be a 2D array; got shape {array.shape!r}.")

    _validate_non_empty(n_samples_arr, "n_samples")
    num_strata = len(n_samples_arr)

    _validate_equal_lengths(
        n_samples_arr,
        true_mean_arr,
        proxy_means_arr,
        correlations_arr,
        names=["n_samples", "true_mean", "proxy_means", "correlations"],
    )

    y_true_per_stratum = []
    y_proxies_per_stratum = []
    groups_per_stratum = []

    seed_sequence = np.random.SeedSequence(random_seed)
    seeds = seed_sequence.spawn(num_strata)

    for stratum_id in range(num_strata):
        y_true_k, y_proxies_k = generate_multi_binary_dataset(
            n_samples=n_samples_arr[stratum_id],
            true_mean=true_mean_arr[stratum_id],
            proxy_means=proxy_means_arr[stratum_id],
            correlations=correlations_arr[stratum_id],
            random_seed=seeds[stratum_id],
        )
        y_true_per_stratum.append(y_true_k)
        y_proxies_per_stratum.append(y_proxies_k)
        groups_per_stratum.append(np.full_like(y_true_k, stratum_id))

    y_true = np.hstack(y_true_per_stratum)
    y_proxies = np.vstack(y_proxies_per_stratum)
    groups = np.hstack(groups_per_stratum)

    return y_true, y_proxies, groups
