from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray


def generate_dataset_binary(
    n: int,
    N: int,
    true_mean: float = 0.7,
    proxy_mean: float = 0.6,
    correlation: float = 0.8,
    random_seed: Optional[int] = None,
) -> "Dataset":
    """Generate a synthetic binary-label dataset for PPI evaluation.

    Parameters
    ----------
    n : int
        Number of records with both true and proxy labels (the labeled subset).
    N : int
        Number of records with proxy labels only (the unlabeled subset).
    true_mean : float
        Expected value (mean) of the true labels.
    proxy_mean : float
        Expected value (mean) of the proxy labels.
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
    s_mean = proxy_mean

    both_1_prob = correlation * np.sqrt(t_mean * s_mean * (1 - t_mean) * (1 - s_mean)) + t_mean * s_mean
    probs = [1 - t_mean - s_mean + both_1_prob, s_mean - both_1_prob, t_mean - both_1_prob, both_1_prob]
    assert min(probs) > 0, "Impossible combination of true_mean, bias, and correlation"

    labeled = rng.choice(4, p=probs, size=n)
    true_vals = labeled // 2
    proxy_labeled = labeled % 2

    proxy_unlabeled = rng.choice(2, p=[1 - s_mean, s_mean], size=N)

    records: List[Dict] = []
    for y, yhat in zip(true_vals, proxy_labeled):
        records.append({"true": int(y), "proxy": int(yhat)})
    for yhat in proxy_unlabeled:
        records.append({"proxy": int(yhat)})

    return Dataset(records)


class Dataset(list):
    @property
    def records(self) -> List[Dict]:
        return list(self)

    def to_numpy(self, fields: List[str]) -> NDArray:
        """Convert the dataset to a 2D numpy array of floats.

        Parameters
        ----------
        fields : List[str]
            Ordered list of record keys to use as columns. Missing values are filled with NaN.

        Returns
        -------
        NDArray
            2D float array of shape (n_records, n_fields).

        Raises
        ------
        ValueError
            If a field is not present in any record.
        """
        rows = []
        for record in self:
            row = [record.get(field, np.nan) for field in fields]
            rows.append(row)
        result = np.array(rows, dtype=float)

        nan_counts = np.isnan(result).sum(axis=0)
        all_nan_mask = nan_counts == len(result)
        unknown_fields = np.array(fields)[all_nan_mask].tolist()
        if unknown_fields:
            raise ValueError(f"Unknown fields: {', '.join(unknown_fields)}")

        return result
