from typing import Optional, Tuple

import numpy as np

from glide.core.dataset import Dataset


def generate_binary_dataset(
    n: int,
    N: int,
    true_mean: float = 0.7,
    proxy_mean: float = 0.6,
    correlation: float = 0.8,
    random_seed: Optional[int] = None,
) -> Tuple[Dataset, Dataset]:
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
    Tuple[Dataset, Dataset]
        [0]: labeled dataset with ``n`` records, containing ``"y_true"`` et ``"y_proxy"`` fields
        [1]: unlabeld dataset with ``N`` records, containing only `"y_proxy"`` field

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
    D = np.sqrt(t_mean * p_mean * (1 - t_mean) * (1 - p_mean))
    both_1_prob = correlation * D + t_mean * p_mean
    max_possible_correlation = min(t_mean * (1 - p_mean), p_mean * (1 - t_mean)) / D
    # probabilities of outcomes (0, 0), (0, 1), (1, 0), (1, 1)
    probs = [1 - t_mean - p_mean + both_1_prob, p_mean - both_1_prob, t_mean - both_1_prob, both_1_prob]
    # some combinations of true_mean, proxy_mean and correlation are impossible
    # and lead to negative probabilities
    if min(probs) <= 0:
        raise ValueError(
            f"Impossible combination of true_mean={true_mean}, proxy_mean={proxy_mean}, "
            f"and correlation={correlation}: leads to negative joint probabilities\n"
            f"maximum possible correlation is {max_possible_correlation}"
        )

    # generate the outcome pairs as integers between 0 and 3 inclusive
    samples = rng.choice(4, p=probs, size=n)
    # extract the true and proxy values via integer division and modulo 2
    # we have 0 = (0, 0), 1 = (0, 1), 2 = (1, 0), 3 = (1, 1)
    y_true = samples // 2
    y_proxy_labeled = samples % 2

    # generate proxy values for un labeled samples
    y_proxy_unlabeled = rng.choice(2, p=[1 - p_mean, p_mean], size=N)

    labeled_records = []
    unlabeled_records = []
    for y_true_, y_proxy_ in zip(y_true, y_proxy_labeled):
        labeled_records.append({"y_true": int(y_true_), "y_proxy": int(y_proxy_)})
    for y_proxy_ in y_proxy_unlabeled:
        unlabeled_records.append({"y_proxy": int(y_proxy_)})

    return Dataset(labeled_records), Dataset(unlabeled_records)


def generate_binary_dataset_with_oracle_sampling(
    N: int,
    true_mean: float = 0.7,
    proxy_mean: float = 0.6,
    correlation: float = 0.8,
    random_seed: Optional[int] = None,
) -> Dataset:
    """Generate a synthetic binary dataset with oracle sampling probabilities for ASI.

    All N records have ground-truth labels (y_true), proxy predictions (y_proxy),
    and an oracle sampling probability (pi) derived from the analytical proxy error.
    The pi values are non-uniform: records where the proxy is less reliable receive
    higher pi, following the optimal ASI sampling rule (Zrnic & Candès, 2024).

    Parameters
    ----------
    N : int
        Total number of records.
    true_mean : float
        Expected mean of y_true. Must be in (0, 1).
    proxy_mean : float
        Expected mean of y_proxy. Must be in (0, 1).
    correlation : float
        Pearson correlation between y_true and y_proxy (marginal, across all records).
    random_seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    Dataset
        Dataset with N records, each containing ``"y_true"`` (int), ``"y_proxy"`` (int),
        and ``"pi"`` (float > 0). All y_true values are present (no missing values).

    Raises
    ------
    ValueError
        If true_mean is not in (0, 1).
    ValueError
        If proxy_mean is not in (0, 1).
    ValueError
        If the combination of true_mean, proxy_mean, and correlation leads to
        negative joint probabilities.

    References
    ----------
    Zrnic, Tijana, and Emmanuel Candès. "Active statistical inference."
    arXiv:2403.03208 (2024). https://arxiv.org/abs/2403.03208
    """
    if not (0 < true_mean < 1):
        raise ValueError(f"true_mean must be in (0, 1), got {true_mean}")
    if not (0 < proxy_mean < 1):
        raise ValueError(f"proxy_mean must be in (0, 1), got {proxy_mean}")

    rng = np.random.default_rng(seed=random_seed)
    p_t = true_mean
    p_p = proxy_mean

    # Global (marginal) joint distribution — same as generate_binary_dataset
    D = np.sqrt(p_t * p_p * (1 - p_t) * (1 - p_p))
    max_possible_correlation = min(p_t * (1 - p_p), p_p * (1 - p_t)) / D
    both_1_prob = correlation * D + p_t * p_p
    probs = [1 - p_t - p_p + both_1_prob, p_p - both_1_prob, p_t - both_1_prob, both_1_prob]
    if min(probs) <= 0:
        raise ValueError(
            f"Impossible combination of true_mean={true_mean}, proxy_mean={proxy_mean}, "
            f"and correlation={correlation}: leads to negative joint probabilities\n"
            f"maximum possible correlation is {max_possible_correlation}"
        )

    # Spread parameter: modulates the conditional correlation across samples
    max_safe_c = 2.0 * min(probs) / D
    c = 0.9 * max_safe_c

    # Latent variable: controls per-sample proxy accuracy
    x = rng.uniform(0.0, 1.0, size=N)

    # Per-sample conditional joint distribution
    both_1_prob_x = (correlation + c * (x - 0.5)) * D + p_t * p_p
    error_prob_x = p_t + p_p - 2.0 * both_1_prob_x

    # Vectorized CDF inversion to sample (y_true, y_proxy) per record
    p00_x = 1.0 - p_t - p_p + both_1_prob_x
    u = rng.uniform(0.0, 1.0, size=N)
    samples = np.where(
        u < p00_x,
        0,
        np.where(
            u < 1.0 - p_t,
            1,
            np.where(u < 1.0 - both_1_prob_x, 2, 3),
        ),
    )
    y_true_arr = samples // 2
    y_proxy_arr = samples % 2

    # Oracle pi: proportional to sqrt(P(error | x_i))
    pi = np.sqrt(error_prob_x)

    records = [
        {"y_true": int(yt), "y_proxy": int(yp), "pi": float(p)} for yt, yp, p in zip(y_true_arr, y_proxy_arr, pi)
    ]
    return Dataset(records)
