from typing import Dict, Hashable, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from glide.core.dataset import Dataset


class StratifiedSampler:
    """Sampler for per-stratum annotation budget allocation.

    This class implements stratified sampling strategies that determine how many records
    to annotate from each stratum, given a fixed annotation budget and proxy labels for
    all records. It supports two allocation strategies:

    - **Proportional allocation** (baseline): Allocates budget proportionally to stratum
      sizes, resulting in uniform sampling probabilities across the dataset.

    - **Neyman allocation** (default, optimal): Assigns more budget to strata with higher
      proxy variance, minimising the asymptotic variance of downstream estimators.
      Particularly effective when proxy variance varies substantially across strata.

    Both allocators use largest-remainder rounding (Hamilton's method) to allocate budget
    across strata. Per-stratum sample sizes are capped at stratum size, so total allocated
    budget Σ n_h ≤ budget (may be less if strata are small). The sampler is typically used
    upstream of statistical estimators to plan annotation effort.

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
    ...     {"group": "A", "y_proxy": 0.95},
    ...     {"group": "B", "y_proxy": 0.1},
    ...     {"group": "B", "y_proxy": 0.9},
    ... ])
    >>> sampler = StratifiedSampler()
    >>> result = sampler.sample(
    ...     dataset,
    ...     y_proxy_field="y_proxy",
    ...     groups_field="group",
    ...     budget=2,
    ...     random_seed=0
    ... )
    >>> result  # doctest: +NORMALIZE_WHITESPACE
    [{'group': 'A', 'y_proxy': 0.9, 'pi': np.float64(0.0), 'xi': 0},
     {'group': 'A', 'y_proxy': 0.95, 'pi': np.float64(0.0), 'xi': 0},
     {'group': 'B', 'y_proxy': 0.1, 'pi': np.float64(1.0), 'xi': 1},
     {'group': 'B', 'y_proxy': 0.9, 'pi': np.float64(1.0), 'xi': 1}]
    """

    def _preprocess(
        self,
        dataset: Dataset,
        y_proxy_field: str,
        groups_field: str,
    ) -> Tuple[NDArray, NDArray]:
        if len(dataset) == 0:
            raise ValueError("Dataset cannot be empty.")
        data = dataset.to_numpy(fields=[y_proxy_field])
        y_proxy = data[:, 0]
        if np.isnan(y_proxy).any():
            raise ValueError("Input proxy values contain NaN")
        if len(np.unique(y_proxy)) == 1:
            raise ValueError("Input proxy values have zero variance")
        groups = np.array([record[groups_field] for record in dataset])

        unique_strata = np.unique(groups)
        has_nonzero_variance = False
        for stratum_id in unique_strata:
            stratum_mask = groups == stratum_id
            stratum_y_proxy = y_proxy[stratum_mask]
            stratum_std = np.std(stratum_y_proxy, ddof=1)
            if not np.isnan(stratum_std) and stratum_std > 0:
                has_nonzero_variance = True
                break

        if not has_nonzero_variance:
            raise ValueError("All strata have zero variance in proxy values")

        return y_proxy, groups

    def _apply_largest_remainder_rounding(
        self,
        raw_allocation: Dict[Hashable, float],
        budget: int,
    ) -> Dict[Hashable, int]:
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
        for stratum_id, _ in sorted_strata[:remaining_slots]:
            allocation[stratum_id] += 1

        return allocation

    def _neyman_allocation(
        self,
        y_proxy: NDArray,
        groups: NDArray,
        budget: int,
    ) -> Dict[Hashable, int]:

        unique_strata = np.unique(groups)

        weights = {}
        stratum_sizes = {}
        for stratum_id in unique_strata:
            stratum_mask = groups == stratum_id
            stratum_size = stratum_mask.sum()
            stratum_y_proxy = y_proxy[stratum_mask]
            stratum_variance = np.std(stratum_y_proxy, ddof=1)
            stratum_variance = np.nan_to_num(stratum_variance, nan=0.0)
            weight = stratum_size * stratum_variance
            weights[stratum_id] = weight
            stratum_sizes[stratum_id] = stratum_size

        total_weight = sum(weights.values())

        raw_allocation = {}
        for stratum_id in unique_strata:
            raw_allocation[stratum_id] = budget * weights[stratum_id] / total_weight

        allocation = self._apply_largest_remainder_rounding(raw_allocation, budget)

        for stratum_id in allocation:
            allocation[stratum_id] = min(allocation[stratum_id], stratum_sizes[stratum_id])

        return allocation

    def _proportional_allocation(
        self,
        y_proxy: NDArray,
        groups: NDArray,
        budget: int,
    ) -> Dict[Hashable, int]:

        unique_strata = np.unique(groups)
        total_size = len(groups)

        raw_allocation = {}
        stratum_sizes = {}
        for stratum_id in unique_strata:
            stratum_mask = groups == stratum_id
            stratum_size = stratum_mask.sum()
            raw_allocation[stratum_id] = budget * stratum_size / total_size
            stratum_sizes[stratum_id] = stratum_size

        allocation = self._apply_largest_remainder_rounding(raw_allocation, budget)

        for stratum_id in allocation:
            allocation[stratum_id] = min(allocation[stratum_id], stratum_sizes[stratum_id])

        return allocation

    def sample(
        self,
        dataset: Dataset,
        y_proxy_field: str,
        groups_field: str,
        budget: int,
        strategy: Literal["proportional", "neyman"] = "neyman",
        random_seed: Optional[int] = None,
        pi_field: str = "pi",
        xi_field: str = "xi",
    ) -> Dataset:
        """Allocate annotation budget across strata and perform stratified sampling.

        Computes per-stratum sample sizes using the specified allocation strategy and performs
        Bernoulli sampling for each record based on its stratum's allocation. Neyman allocation
        (default) assigns more budget to strata with higher proxy variance, minimising asymptotic
        variance of downstream estimators. Proportional allocation allocates budget proportionally
        to stratum sizes and serves as a baseline.

        Each record receives a drawing probability π_i = n_h / stratum_size (capped at 1), and
        is independently selected via a Bernoulli trial. The actual number of selected items is
        a random variable with expectation ≤ budget.

        Parameters
        ----------
        dataset : Dataset
            Dataset with all records and proxy labels.
        y_proxy_field : str
            Field name holding proxy labels. Mandatory.
        groups_field : str
            Field name holding stratum identifiers. Mandatory.
        budget : int
            Target annotation budget. Must be positive. Mandatory.
        strategy : str, optional
            Allocation strategy: "neyman" (default) or "proportional".
            "neyman": assigns more budget to higher-variance strata.
            "proportional": allocates proportionally to stratum sizes.
        random_seed : int or None, optional
            Random seed for reproducible sampling. Defaults to None (non-deterministic).
        pi_field : str, optional
            Name of the output column for drawing probabilities. Defaults to "pi".
        xi_field : str, optional
            Name of the output column for selection indicators. Defaults to "xi".

        Returns
        -------
        result : Dataset
            Input dataset augmented with two columns:
            - `pi_field`: drawing probability π_i ∈ (0, 1] for each record.
            - `xi_field`: 1 if selected for annotation, 0 otherwise.

        Raises
        ------
        KeyError
            If groups_field is missing from a record.
        ValueError
            If y_proxy_field is not present in any record, if strategy is unknown,
            if budget is not a strictly positive integer, or if budget exceeds
            the number of records in the dataset.
        """
        if (not isinstance(budget, (int, np.integer))) or isinstance(budget, bool) or budget <= 0:
            raise ValueError(f"'budget' must be a strictly positive integer; got {budget!r}.")
        if budget > len(dataset):
            raise ValueError(
                f"'budget' must not exceed the number of records in the dataset; "
                f"got budget={budget} but dataset has {len(dataset)} records."
            )

        y_proxy, groups = self._preprocess(dataset, y_proxy_field, groups_field)

        if strategy == "proportional":
            allocation = self._proportional_allocation(y_proxy, groups, budget)
        elif strategy == "neyman":
            allocation = self._neyman_allocation(y_proxy, groups, budget)
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Expected 'proportional' or 'neyman'.")

        rng = np.random.default_rng(random_seed)

        result_records = []
        for record in dataset:
            stratum_id = record[groups_field]
            stratum_size = (groups == stratum_id).sum()
            n_h = allocation[stratum_id]
            pi = min(n_h / stratum_size, 1.0)
            xi = rng.binomial(n=1, p=pi)

            new_record = dict(record)
            new_record[pi_field] = pi
            new_record[xi_field] = xi
            result_records.append(new_record)

        return Dataset(result_records)
