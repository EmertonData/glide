from typing import Dict, Generic, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from glide.confidence_sequences.asymptotic import _compute_asymptotic_bounds
from glide.engines.base import DatasetT, MeanEstimationEngine, TuningParameterT
from glide.monitors.core import _postprocess, _preprocess


class AsymptoticRM(Generic[DatasetT, TuningParameterT]):
    _engine: MeanEstimationEngine[DatasetT, TuningParameterT]

    def _preprocess_subset(self, fields: Dict[str, NDArray], mask: NDArray) -> DatasetT:
        field_subsets = [field[mask] for field in fields.values()]
        dataset = self._engine.preprocess(*field_subsets)
        return dataset

    def _detect(
        self,
        label_fields: Dict[str, NDArray],
        batches: NDArray,
        *,
        higher_is_better: bool,
        confidence_level: float,
        tightest_at_batch: int,
        power_tuning: bool,
        identifier_fields: Optional[Dict[str, NDArray]] = None,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        if identifier_fields is None:
            identifier_fields = {}
        risk_fields, batch_identifiers, batch_codes = _preprocess(
            label_fields,
            identifier_fields,
            batches,
            higher_is_better=higher_is_better,
            confidence_level=confidence_level,
        )
        n_batches = len(batch_identifiers)
        batch_risk_mean_estimates = np.empty(n_batches)
        batch_risk_std_estimates = np.empty(n_batches)
        for position in range(n_batches):
            try:
                batch_risk_dataset = self._preprocess_subset(risk_fields, batch_codes == position)
            except ValueError as error:
                raise ValueError(f"{error} (batch '{batch_identifiers[position]}').") from error

            if position == 0 or not power_tuning:
                tuning_parameter = self._engine.fit_tuning_parameter(batch_risk_dataset, power_tuning=False)
            else:
                prefix_risk_dataset = self._preprocess_subset(risk_fields, batch_codes < position)
                tuning_parameter = self._engine.fit_tuning_parameter(prefix_risk_dataset, power_tuning=True)

            batch_risk_mean_estimates[position], batch_risk_std_estimates[position] = self._engine.compute_mean_and_std(
                batch_risk_dataset, tuning_parameter
            )

        miscoverage = 1.0 - confidence_level
        risk_running_means, risk_lower_bounds = _compute_asymptotic_bounds(
            batch_risk_mean_estimates, batch_risk_std_estimates, miscoverage, tightest_at_batch
        )
        running_means, confidence_bounds, batch_mean_estimates = _postprocess(
            risk_running_means, risk_lower_bounds, batch_risk_mean_estimates, higher_is_better=higher_is_better
        )
        return batch_codes, batch_mean_estimates, running_means, confidence_bounds
