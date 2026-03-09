from typing import Optional

import numpy as np

from glide.core.dataset import Dataset


def generate_dataset_binary(
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
    """
    assert 0 < true_mean < 1, "true_mean must be in (0, 1)"
    assert 0 < proxy_mean < 1, "proxy_mean must be in (0, 1)"

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
    assert min(probs) > 0, "Impossible combination of true_mean, bias, and correlation"

    # generate the outcome pairs as integers between 0 and 3 inclusive
    labeled = rng.choice(4, p=probs, size=n)
    # extract the true and proxy values via integer division and modulo 2
    # we have 0 = (0, 0), 1 = (0, 1), 2 = (1, 0), 3 = (1, 1)
    true_vals = labeled // 2
    proxy_labeled = labeled % 2

    # generate proxy values for un labeled samples
    proxy_unlabeled = rng.choice(2, p=[1 - p_mean, p_mean], size=N)

    records = []
    for y_true, y_proxy in zip(true_vals, proxy_labeled):
        records.append({"true": int(y_true), "proxy": int(y_proxy)})
    for y_proxy in proxy_unlabeled:
        records.append({"proxy": int(y_proxy)})

    return Dataset(records)
