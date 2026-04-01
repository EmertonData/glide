from typing import Dict, Hashable, Literal, Tuple

import numpy as np
from numpy.typing import NDArray

from glide.core.dataset import Dataset


class StratifiedSampler:
    """Sampler for optimal per-stratum annotation budget allocation.

    This class implements stratified sampling strategies that determine how many records
    to annotate from each stratum, given a fixed annotation budget and proxy labels for
    all records. It supports two allocation strategies:

    - **Neyman allocation** (default, optimal): Assigns more budget to strata with higher
      proxy variance, minimising the asymptotic variance of downstream estimators.
      Particularly effective when proxy variance varies substantially across strata.

    - **Proportional allocation** (baseline): Allocates budget proportionally to stratum
      sizes, resulting in uniform sampling probabilities across the dataset.

    Both allocators use largest-remainder rounding (Hamilton's method) to allocate budget
    across strata. Per-stratum sample sizes are capped at stratum size, so total allocated
    budget Σ n_h ≤ budget (may be less if strata are small). The sampler is typically used
    upstream of statistical estimators to plan annotation effort. Common downstream workflows
    include:

    - **ASIMeanEstimator**: Uses per-record sampling probabilities to correct for non-uniform
      annotation via inverse-probability weighting (IPW).
    - **StratifiedPPIMeanEstimator**: Leverages per-stratum sample counts to stratify
      labeled and unlabeled data, improving efficiency under heterogeneous variance.

    References
    ----------
    Fogliato, Riccardo, Pratik Patil, Mathew Monfort, and Pietro Perona.
    "A framework for efficient model evaluation through stratification, sampling,
    and estimation." In European Conference on Computer Vision, pp. 140-158.
    Cham: Springer Nature Switzerland, 2024. https://arxiv.org/abs/2406.07320

    Examples
    --------
    >>> from glide.core.dataset import Dataset
    >>> from glide.samplers.stratified import StratifiedSampler
    >>> dataset = Dataset([
    ...     {"group": "A", "y_proxy": 0.9},
    ...     {"group": "A", "y_proxy": 0.8},
    ...     {"group": "B", "y_proxy": 0.2},
    ...     {"group": "B", "y_proxy": 0.3},
    ... ])
    >>> sampler = StratifiedSampler()
    >>> result = sampler.sample(
    ...     dataset,
    ...     groups_field="group",
    ...     y_proxy_field="y_proxy",
    ...     budget=2
    ... )
    >>> type(result)
    <class 'glide.core.dataset.Dataset'>
    >>> result[0]["n_h"]  # doctest: +SKIP
    1
    """

    def __init__(self) -> None:
        """Initialize the StratifiedSampler."""
        pass

    def _preprocess(
        self,
        dataset: Dataset,
        groups_field: str,
        y_proxy_field: str,
    ) -> Tuple[NDArray, NDArray]:
        """Extract proxy labels and stratum identifiers from dataset.

        Parameters
        ----------
        dataset : Dataset
            Input dataset with records.
        groups_field : str
            Name of the field holding stratum identifiers.
        y_proxy_field : str
            Name of the field holding proxy labels.

        Returns
        -------
        y_proxy : NDArray
            1D array of proxy labels, shape (n,).
        groups : NDArray
            1D array of stratum identifiers, shape (n,).

        Raises
        ------
        KeyError
            If groups_field is missing from a record.
        ValueError
            If y_proxy_field is not present in any record.
        """
        data = dataset.to_numpy(fields=[y_proxy_field])
        y_proxy = data[:, 0]
        groups = np.array([record[groups_field] for record in dataset])
        return y_proxy, groups

    def _apply_largest_remainder_rounding(
        self,
        raw_allocation: Dict[Hashable, float],
        budget: int,
    ) -> Dict[Hashable, int]:
        """Apply largest-remainder rounding to distribute budget across strata.

        Assigns the floor value to each stratum, then distributes remaining slots
        one-by-one to strata with the largest fractional parts. Result sums to budget
        before clipping; after clipping to stratum sizes, sum may be less than budget.

        Parameters
        ----------
        raw_allocation : Dict[Hashable, float]
            Dictionary mapping stratum_id to raw (float) allocation.
        budget : int
            Target budget for distribution before clipping.

        Returns
        -------
        allocation : Dict[Hashable, int]
            Dictionary mapping stratum_id to integer allocation summing to budget.
        """
        allocation = {}
        remainders = {}

        for stratum_id, raw_value in raw_allocation.items():
            floor_value = int(np.floor(raw_value))
            allocation[stratum_id] = floor_value
            remainder = raw_value - floor_value
            remainders[stratum_id] = remainder

        current_sum = sum(allocation.values())
        remaining_slots = budget - current_sum

        sorted_strata = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
        for i in range(remaining_slots):
            stratum_id = sorted_strata[i][0]
            allocation[stratum_id] += 1

        return allocation

    def _neyman_allocation(
        self,
        y_proxy: NDArray,
        groups: NDArray,
        budget: int,
    ) -> Dict[Hashable, int]:
        """Allocate budget via Neyman allocation (optimal for heterogeneous variance).

        Allocates more budget to strata with higher proxy variance. If all strata have
        zero variance, falls back to proportional allocation based on stratum sizes.

        Parameters
        ----------
        y_proxy : NDArray
            Proxy labels, shape (n,).
        groups : NDArray
            Stratum identifiers, shape (n,).
        budget : int
            Total annotation budget.

        Returns
        -------
        allocation : Dict[Hashable, int]
            Mapping from stratum_id to per-stratum budget (count of annotations).
        """
        unique_strata = np.unique(groups)

        weights = {}
        for stratum_id in unique_strata:
            stratum_mask = groups == stratum_id
            stratum_size = stratum_mask.sum()
            stratum_y_proxy = y_proxy[stratum_mask]
            stratum_variance = np.std(stratum_y_proxy, ddof=1)
            weight = stratum_size * stratum_variance
            weights[stratum_id] = weight

        total_weight = sum(weights.values())

        if total_weight == 0:
            return self._proportional_allocation(y_proxy, groups, budget)

        raw_allocation = {}
        for stratum_id in unique_strata:
            raw_allocation[stratum_id] = budget * weights[stratum_id] / total_weight

        allocation = self._apply_largest_remainder_rounding(raw_allocation, budget)

        for stratum_id in allocation:
            stratum_size = (groups == stratum_id).sum()
            allocation[stratum_id] = min(allocation[stratum_id], stratum_size)

        return allocation

    def _proportional_allocation(
        self,
        y_proxy: NDArray,
        groups: NDArray,
        budget: int,
    ) -> Dict[Hashable, int]:
        """Allocate budget proportionally to stratum sizes.

        Parameters
        ----------
        y_proxy : NDArray
            Proxy labels, shape (n,).
        groups : NDArray
            Stratum identifiers, shape (n,).
        budget : int
            Total annotation budget.

        Returns
        -------
        allocation : Dict[Hashable, int]
            Mapping from stratum_id to per-stratum budget.
        """
        unique_strata = np.unique(groups)
        total_size = len(groups)

        raw_allocation = {}
        for stratum_id in unique_strata:
            stratum_size = (groups == stratum_id).sum()
            raw_allocation[stratum_id] = budget * stratum_size / total_size

        allocation = self._apply_largest_remainder_rounding(raw_allocation, budget)

        for stratum_id in allocation:
            stratum_size = (groups == stratum_id).sum()
            allocation[stratum_id] = min(allocation[stratum_id], stratum_size)

        return allocation

    def sample(
        self,
        dataset: Dataset,
        groups_field: str,
        y_proxy_field: str,
        budget: int,
        strategy: Literal["proportional", "neyman"] = "neyman",
    ) -> Dataset:
        """Allocate annotation budget across strata and augment dataset with allocations.

        Computes per-stratum sample sizes using the specified allocation strategy and adds
        an `n_h` column to the dataset indicating the allocated budget for each record's stratum.
        Neyman allocation (default) assigns more budget to strata with higher proxy variance,
        minimising asymptotic variance of downstream estimators. Proportional allocation
        allocates budget proportionally to stratum sizes and serves as a baseline.
        Per-stratum allocations are capped at stratum size, so total Σ n_h ≤ budget.

        Parameters
        ----------
        dataset : Dataset
            Dataset with all records and proxy labels.
        groups_field : str
            Field name holding stratum identifiers. Mandatory.
        y_proxy_field : str
            Field name holding proxy labels. Mandatory.
        budget : int
            Target annotation budget. Must be positive. Mandatory.
        strategy : str, optional
            Allocation strategy: "neyman" (default) or "proportional".
            "neyman": assigns more budget to higher-variance strata.
            "proportional": allocates proportionally to stratum sizes.

        Returns
        -------
        result : Dataset
            Input dataset augmented with `n_h` column. For each record, `n_h` contains
            the per-stratum allocation for that record's stratum. Total Σ n_h ≤ budget.

        Raises
        ------
        KeyError
            If groups_field is missing from a record.
        ValueError
            If y_proxy_field is not present in any record, or if strategy is unknown.
        """
        y_proxy, groups = self._preprocess(dataset, groups_field, y_proxy_field)

        if strategy == "proportional":
            allocation = self._proportional_allocation(y_proxy, groups, budget)
        elif strategy == "neyman":
            allocation = self._neyman_allocation(y_proxy, groups, budget)
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Expected 'proportional' or 'neyman'.")

        result_records = []
        for record in dataset:
            stratum_id = record[groups_field]
            new_record = dict(record)
            new_record["n_h"] = allocation[stratum_id]
            result_records.append(new_record)

        return Dataset(result_records)
