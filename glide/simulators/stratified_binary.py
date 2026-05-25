from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from glide.simulators.binary import generate_binary_dataset


def generate_stratified_binary_dataset(
    n_total: List[int],
    true_mean: List[float],
    proxy_mean: List[float],
    correlation: List[float],
    random_seed: Optional[int] = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Generate a synthetic stratified binary-label oracle dataset.

    Generate multiple strata with potentially different parameters (true_mean, proxy_mean,
    correlation, n_total per stratum). This enables simulation of heterogeneous data where
    different groups have different proxy-truth relationships.

    Parameters
    ----------
    n_total : List[int]
        Total number of samples per stratum. All samples have both true and proxy labels.
        Length must equal number of strata.
    true_mean : List[float]
        Expected mean value of the true labels per stratum.
        Length must equal number of strata.
    proxy_mean : List[float]
        Expected mean value of the proxy labels per stratum.
        Length must equal number of strata.
    correlation : List[float]
        Pearson correlation between true and proxy per stratum.
        Length must equal number of strata.
    random_seed : int, optional
        Seed for reproducibility. If provided, seeds are derived deterministically.

    Returns
    -------
    Tuple[NDArray, NDArray, NDArray]
        Let ``N = sum(n_total)`` be the total number of samples across all strata.

        [0]: array of shape ``(N,)``, y_true containing ground-truth labels.
        [1]: array of shape ``(N,)``, y_proxy containing proxy labels.
        [2]: array of shape ``(N,)``, integer stratum identifiers.

    Raises
    ------
    ValueError
        If input lists have different lengths.
    ValueError
        If fewer than 1 stratum is specified.
    ValueError
        If any stratum has invalid parameters (see generate_binary_dataset).

    Examples
    --------
    >>> import numpy as np
    >>> from glide.simulators import generate_stratified_binary_dataset
    >>> y_true, y_proxy, groups = generate_stratified_binary_dataset(
    ...     n_total=[6, 8],
    ...     true_mean=[0.6, 0.8],
    ...     proxy_mean=[0.5, 0.7],
    ...     correlation=[0.7, 0.75],
    ...     random_seed=42
    ... )
    >>> len(y_true)
    14
    >>> len(groups)
    14
    >>> len(y_proxy)
    14
    >>> bool(np.all(np.isin(y_true, [0.0, 1.0])))
    True
    >>> bool(np.all(np.isin(y_proxy, [0.0, 1.0])))
    True
    """
    num_strata = len(n_total)
    if num_strata < 1:
        raise ValueError(f"Number of strata must be at least 1, got {num_strata}")

    # Validate all lists have the same length
    param_lengths = {
        "n_total": len(n_total),
        "true_mean": len(true_mean),
        "proxy_mean": len(proxy_mean),
        "correlation": len(correlation),
    }
    if not all(length == num_strata for length in param_lengths.values()):
        lengths_str = ", ".join(f"{name}={length}" for name, length in param_lengths.items())
        raise ValueError(f"All input lists must have the same length. Got: {lengths_str}")

    # Generate data for each stratum
    y_true_per_stratum = []
    y_proxy_per_stratum = []
    groups_per_stratum = []

    seed_sequence = np.random.SeedSequence(random_seed)
    seeds = seed_sequence.spawn(num_strata)

    for stratum_id in range(num_strata):
        y_true_k, y_proxy_k = generate_binary_dataset(
            n_total=n_total[stratum_id],
            true_mean=true_mean[stratum_id],
            proxy_mean=proxy_mean[stratum_id],
            correlation=correlation[stratum_id],
            random_seed=seeds[stratum_id],
        )
        y_true_per_stratum.append(y_true_k)
        y_proxy_per_stratum.append(y_proxy_k)
        groups_per_stratum.append(np.full_like(y_true_k, stratum_id))

    y_true = np.hstack(y_true_per_stratum)
    y_proxy = np.hstack(y_proxy_per_stratum)
    groups = np.hstack(groups_per_stratum)

    return y_true, y_proxy, groups
