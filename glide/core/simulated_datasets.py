from typing import Optional

import numpy as np

from glide.core.dataset import Dataset


def generate_binary_dataset(
    n: int,
    N: int,
    true_mean: float = 0.7,
    proxy_mean: float = 0.6,
    correlation: float = 0.8,
    random_seed: Optional[int] = None,
) -> Dataset:
    """Generate a synthetic binary-label dataset for evaluation.

    Parameters
    ----------
    n : int
        Number of records with both true and proxy labels (the labeled subset).
    N : int
        Number of records with proxy labels only (the unlabeled subset).
    true_mean : float
        Expected mean value of the true labels.
    proxy_mean : float
        Expected mean value of the proxy labels.
    correlation : float
        Pearson correlation between true and proxy on the labeled subset.
    random_seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    Dataset
        A Dataset of n + N records. Each record contains ``"proxy"`` (always present)
        and ``"true"`` (only for the first ``n`` records).

    Raises
    ------
    ValueError
        If ``true_mean`` is not in (0, 1).
    ValueError
        If ``proxy_mean`` is not in (0, 1).
    ValueError
        If the combination of ``true_mean``, ``proxy_mean``, and ``correlation`` is
        impossible (leads to negative joint probabilities).

    References
    ----------
    .. [SO] `Correlation between Bernoulli Variables <https://math.stackexchange.com/questions/610443/finding-a-correlation-between-bernoulli-variables>`_

    """
    if not (0 < true_mean < 1):
        raise ValueError(f"true_mean must be in (0, 1), got {true_mean}")
    if not (0 < proxy_mean < 1):
        raise ValueError(f"proxy_mean must be in (0, 1), got {proxy_mean}")

    rng = np.random.default_rng(seed=random_seed)

    t_mean = true_mean
    p_mean = proxy_mean

    # we will generate pairs values (true, proxy) with true and proxy equal to 0 or 1
    # probability of outcome (1, 1)
    both_1_prob = correlation * np.sqrt(t_mean * p_mean * (1 - t_mean) * (1 - p_mean)) + t_mean * p_mean
    # probabilities of outcomes (0, 0), (0, 1), (1, 0), (1, 1)
    probs = [1 - t_mean - p_mean + both_1_prob, p_mean - both_1_prob, t_mean - both_1_prob, both_1_prob]
    # some combinations of true_mean, proxy_mean and correlation are impossible
    # and lead to negative probabilities
    if min(probs) <= 0:
        raise ValueError(
            f"Impossible combination of true_mean={true_mean}, proxy_mean={proxy_mean}, "
            f"and correlation={correlation}: leads to negative joint probabilities"
        )

    # generate the outcome pairs as integers between 0 and 3 inclusive
    samples = rng.choice(4, p=probs, size=n)
    # extract the true and proxy values via integer division and modulo 2
    # we have 0 = (0, 0), 1 = (0, 1), 2 = (1, 0), 3 = (1, 1)
    y_true = samples // 2
    y_proxy_labeled = samples % 2

    # generate proxy values for un labeled samples
    y_proxy_unlabeled = rng.choice(2, p=[1 - p_mean, p_mean], size=N)

    records = []
    for y_true_, y_proxy_ in zip(y_true, y_proxy_labeled):
        records.append({"y_true": int(y_true_), "y_proxy": int(y_proxy_)})
    for y_proxy_ in y_proxy_unlabeled:
        records.append({"y_proxy": int(y_proxy_)})

    return Dataset(records)
