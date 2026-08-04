from typing import Generic, List, Tuple

import numpy as np
from numpy.typing import NDArray

from glide.confidence_sequences import AsymptoticConfidenceSequence
from glide.confidence_sequences.asymptotic import _compute_asymptotic_bounds
from glide.core.validation import _validate_bounds, _validate_equal_lengths, _validate_has_no_nan, _validate_non_empty
from glide.engines.base import DatasetT, MeanEstimationEngine
from glide.monitors.core import _postprocess, _reorient, _unique_ordered_batches


class AsymptoticRM(Generic[DatasetT]):
    _engine: MeanEstimationEngine[DatasetT]

    def _preprocess_subset(self, fields: List[NDArray], mask: NDArray) -> DatasetT:
        sliced_fields = [field[mask] for field in fields]
        dataset = self._engine.preprocess(*sliced_fields)
        return dataset

    def _detect(
        self,
        fields: List[NDArray],
        field_names: List[str],
        batches: NDArray,
        higher_is_better: bool,
        confidence_level: float,
        tightest_at_batch: int,
        power_tuning: bool,
    ) -> Tuple[NDArray, NDArray, AsymptoticConfidenceSequence]:
        _validate_bounds(
            confidence_level,
            "confidence_level",
            lower=0.5,
            upper=1,
            left_inclusive=False,
            right_inclusive=False,
            error_message=(
                f"'confidence_level' must be in (0.5, 1) for the asymptotic monitor; got {confidence_level!r}."
            ),
        )
        _validate_non_empty(batches, "batches")
        _validate_equal_lengths(*fields, batches, names=[*field_names, "batches"])
        _validate_has_no_nan(batches, "batches")

        risk_fields = [_reorient(field, higher_is_better) for field in fields]
        batch_identifiers, batch_codes = _unique_ordered_batches(batches)
        n_batches = len(batch_identifiers)
        batch_mean_estimates = np.empty(n_batches)
        batch_std_estimates = np.empty(n_batches)
        for position in range(n_batches):
            try:
                batch_dataset = self._preprocess_subset(risk_fields, batch_codes == position)
            except ValueError as error:
                raise ValueError(f"{error} (batch '{batch_identifiers[position]}').") from error

            if position == 0 or not power_tuning:
                tuning_parameter = self._engine.fit_tuning_parameter(batch_dataset, power_tuning=False)
            else:
                prefix_dataset = self._preprocess_subset(risk_fields, batch_codes < position)
                tuning_parameter = self._engine.fit_tuning_parameter(prefix_dataset, power_tuning=True)

            batch_mean_estimates[position], batch_std_estimates[position] = self._engine.compute_mean_and_std(
                batch_dataset, tuning_parameter
            )

        miscoverage = 1.0 - confidence_level
        risk_running_means, risk_lower_bounds = _compute_asymptotic_bounds(
            batch_mean_estimates, batch_std_estimates, miscoverage, tightest_at_batch
        )
        running_means, confidence_bounds, metric_batch_mean_estimates = _postprocess(
            risk_running_means, risk_lower_bounds, batch_mean_estimates, higher_is_better
        )
        confidence_sequence = AsymptoticConfidenceSequence(
            running_mean_estimates=running_means, confidence_bounds=confidence_bounds
        )
        return batch_codes, metric_batch_mean_estimates, confidence_sequence
