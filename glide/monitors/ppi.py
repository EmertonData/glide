from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from glide.confidence_sequences import EmpiricalBernsteinConfidenceSequence
from glide.confidence_sequences.empirical_bernstein import _compute_empirical_bernstein_bounds
from glide.core.validation import (
    _validate_bounds,
    _validate_equal_lengths,
    _validate_has_no_nan,
    _validate_non_empty,
    _validate_y_proxy,
    _validate_y_true,
)
from glide.estimators.core import _split_labeled_unlabeled
from glide.estimators.ppi_core import _compute_mean_estimate, _compute_tuning_parameter
from glide.mean_monitoring_results import PredictionPoweredMeanMonitoringResult
from glide.monitors.core import _scale_from_unit_risk, _scale_to_unit_risk, _unique_ordered_batches


class PPIMeanMonitor:
    """Anytime-valid drift monitor over a batched dataset of true and proxy labels.

    It uses a per-batch Prediction-Powered Inference (PPI) estimate: a small set of
    true labels and a large set of proxy labels, combined with a power-tuning
    parameter fitted on the batches that strictly precede it (predictable, as
    required for validity). The monitor builds an anytime-valid confidence sequence
    on the running mean of the per-batch PPI estimates, and raises a drift alarm
    when it crosses a user-supplied ``threshold`` (the worst tolerable metric
    value). Because the bounds are valid at all times simultaneously, :meth:`detect`
    may be called after every new batch without inflating the false-alarm
    probability.

    References
    ----------
    Zhang, Guangyi, Yunlong Cai, Guanding Yu, and Osvaldo Simeone. "Prediction-powered
    risk monitoring of deployed models for detecting harmful distribution shifts."
    arXiv preprint arXiv:2602.02229 (2026).

    Howard, Steven R., Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon. "Time-uniform,
    nonparametric, nonasymptotic confidence sequences." The Annals of Statistics 49,
    no. 2 (2021): 1055-1080.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.monitors import PPIMeanMonitor
    >>> pre_drift_y_true = np.array([0.0, 0.2, np.nan, np.nan])
    >>> pre_drift_y_proxy = np.array([0.0, 0.2, 0.0, 0.2])
    >>> post_drift_y_true = np.array([0.8, 1.0, np.nan, np.nan])
    >>> post_drift_y_proxy = np.array([0.8, 1.0, 0.8, 1.0])
    >>> y_true = np.hstack([pre_drift_y_true, np.tile(post_drift_y_true, 50)])
    >>> y_proxy = np.hstack([pre_drift_y_proxy, np.tile(post_drift_y_proxy, 50)])
    >>> batches = np.repeat(np.arange(51), 4)
    >>> monitor = PPIMeanMonitor()
    >>> result = monitor.detect(y_true, y_proxy, batches, higher_is_better=False, threshold=0.5)
    >>> result.drift_detected
    True
    >>> result.first_alarm_index
    11
    """

    def _preprocess(
        self,
        y_true: NDArray,
        y_proxy: NDArray,
        batches: NDArray,
        higher_is_better: bool,
        threshold: float,
        confidence_level: float,
        metric_lower_bound: float,
        metric_upper_bound: float,
    ) -> Tuple[NDArray, NDArray, float, NDArray, NDArray, NDArray]:
        _validate_non_empty(y_true, "y_true")
        _validate_equal_lengths(y_true, y_proxy, batches, names=["y_true", "y_proxy", "batches"])
        _validate_has_no_nan(batches, "batches")
        _validate_bounds(
            confidence_level, "confidence_level", lower=0, upper=1, left_inclusive=False, right_inclusive=False
        )
        _validate_bounds(
            metric_lower_bound,
            "metric_lower_bound",
            upper=metric_upper_bound,
            right_inclusive=False,
            error_message=(
                f"'metric_lower_bound' must be strictly smaller than 'metric_upper_bound'; "
                f"got {metric_lower_bound!r} and {metric_upper_bound!r}."
            ),
        )
        _validate_bounds(
            threshold,
            "threshold",
            lower=metric_lower_bound,
            upper=metric_upper_bound,
            error_message=(
                f"'threshold' must lie between 'metric_lower_bound'={metric_lower_bound!r} and "
                f"'metric_upper_bound'={metric_upper_bound!r}; got {threshold!r}."
            ),
        )
        _validate_y_proxy(y_proxy)
        _validate_y_true(y_true)
        labeled_mask = ~np.isnan(y_true)
        labeled_values = y_true[labeled_mask]
        _validate_bounds(
            labeled_values,
            "y_true",
            lower=metric_lower_bound,
            upper=metric_upper_bound,
            error_message=(
                f"'y_true' values must lie between 'metric_lower_bound'={metric_lower_bound!r} and "
                f"'metric_upper_bound'={metric_upper_bound!r}; got values in "
                f"[{labeled_values.min()!r}, {labeled_values.max()!r}]."
            ),
        )
        _validate_bounds(
            y_proxy,
            "y_proxy",
            lower=metric_lower_bound,
            upper=metric_upper_bound,
            error_message=(
                f"'y_proxy' values must lie between 'metric_lower_bound'={metric_lower_bound!r} and "
                f"'metric_upper_bound'={metric_upper_bound!r}; got values in "
                f"[{y_proxy.min()!r}, {y_proxy.max()!r}]."
            ),
        )
        batch_identifiers, batch_codes = _unique_ordered_batches(batches)
        n_batches = len(batch_identifiers)
        batch_n_true = np.bincount(batch_codes[labeled_mask], minlength=n_batches)
        batch_n_proxy = np.bincount(batch_codes, minlength=n_batches)
        batch_n_unlabeled = batch_n_proxy - batch_n_true

        worst_labeled_position = np.argmin(batch_n_true)
        _validate_bounds(
            batch_n_true[worst_labeled_position],
            "y_true",
            lower=2,
            error_message=(
                f"'y_true' must have at least 2 labeled values per batch; got "
                f"{batch_n_true[worst_labeled_position]} in batch '{batch_identifiers[worst_labeled_position]}'."
            ),
        )
        worst_unlabeled_position = np.argmin(batch_n_unlabeled)
        _validate_bounds(
            batch_n_unlabeled[worst_unlabeled_position],
            "y_true",
            lower=2,
            error_message=(
                f"'y_true' must have at least 2 unlabeled values per batch; got "
                f"{batch_n_unlabeled[worst_unlabeled_position]} in batch "
                f"'{batch_identifiers[worst_unlabeled_position]}'."
            ),
        )

        risk_y_true = _scale_to_unit_risk(y_true, metric_lower_bound, metric_upper_bound, higher_is_better)
        risk_y_proxy = _scale_to_unit_risk(y_proxy, metric_lower_bound, metric_upper_bound, higher_is_better)
        risk_threshold = _scale_to_unit_risk(threshold, metric_lower_bound, metric_upper_bound, higher_is_better)
        return risk_y_true, risk_y_proxy, risk_threshold, batch_codes, batch_n_true, batch_n_proxy

    def _postprocess(
        self,
        risk_running_means: NDArray,
        risk_confidence_bounds: NDArray,
        risk_batch_mean_estimates: NDArray,
        higher_is_better: bool,
        metric_lower_bound: float,
        metric_upper_bound: float,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        running_means = _scale_from_unit_risk(
            risk_running_means, metric_lower_bound, metric_upper_bound, higher_is_better
        )
        confidence_bounds = _scale_from_unit_risk(
            risk_confidence_bounds, metric_lower_bound, metric_upper_bound, higher_is_better
        )
        batch_mean_estimates = _scale_from_unit_risk(
            risk_batch_mean_estimates, metric_lower_bound, metric_upper_bound, higher_is_better
        )
        return running_means, confidence_bounds, batch_mean_estimates

    def detect(
        self,
        y_true: NDArray,
        y_proxy: NDArray,
        batches: NDArray,
        higher_is_better: bool,
        threshold: float,
        metric_name: str = "Metric",
        confidence_level: float = 0.8,
        power_tuning: bool = True,
        metric_lower_bound: float = 0.0,
        metric_upper_bound: float = 1.0,
    ) -> PredictionPoweredMeanMonitoringResult:
        """Detect a drift of the running mean across a batched dataset.

        Splits the data by batch, computes a prediction-powered estimate per batch,
        and builds an anytime-valid empirical-Bernstein confidence sequence on the
        running mean of those estimates. An alarm is raised at every batch where the
        sequence crosses the user-supplied ``threshold``.

        Rows must be ordered oldest batch first and grouped into contiguous blocks;
        identifier values are not compared, so any label type works. Batches must be
        non-overlapping (no shared samples), and successive calls must be made on
        growing histories of the same data; passing the full accumulated dataset at
        every call makes the anytime-valid guarantee hold jointly over all calls.
        Alternatively the data may be restricted to the most recent batches, in which
        case the guarantee holds within each restriction but not across the moving
        history as a whole.

        Parameters
        ----------
        y_true : NDArray
            Array of labeled observations, shape ``(n_samples,)``.
            Labeled entries are finite; unlabeled entries are ``np.nan``.
        y_proxy : NDArray
            Array of proxy predictions, shape ``(n_samples,)``.
            Must be fully populated (no NaN).
        batches : NDArray
            Array of batch identifiers, shape ``(n_samples,)``. Rows must be ordered
            oldest batch first and grouped into contiguous blocks. Identifier values
            are not compared, so any hashable label type works (integers, dates,
            free-form strings).
        higher_is_better : bool
            ``False`` when the metric is a risk (drift means the metric increased),
            ``True`` when it is a performance (drift means the metric decreased).
        threshold : float
            The metric value the running mean is monitored against, in metric units:
            the worst level the user is willing to tolerate. An alarm fires once the
            anytime-valid bound proves the running metric has crossed it (the running
            risk exceeds it for a risk, the running performance falls below it for a
            performance). Must lie within ``[metric_lower_bound, metric_upper_bound]``.
        metric_name : str, optional
            Human-readable label for the metric. Defaults to ``"Metric"``.
        confidence_level : float, optional
            How confident each alarm should be. At the default ``0.8`` the monitor
            raises "80% confident" alarms: under no drift it has at most a 20% chance
            of raising a false alarm. Raising this value demands more evidence before
            alarming, so alarms become more trustworthy but arrive later; lowering it
            detects sooner at the cost of more false alarms.
        power_tuning : bool, optional
            If ``True`` (default), compute the power-tuning parameter of each batch on
            all previous batches (the first batch, having no predecessor, uses
            ``1.0``). If ``False``, use ``1.0`` everywhere (classic PPI).
        metric_lower_bound : float, optional
            Known lower bound of the metric. Defaults to ``0.0``.
        metric_upper_bound : float, optional
            Known upper bound of the metric. Defaults to ``1.0``.

        Returns
        -------
        PredictionPoweredMeanMonitoringResult
            Per-batch estimates, running means, anytime-valid confidence bounds,
            alarm flags, and the alarm threshold, all in the original metric
            orientation.

        Raises
        ------
        ValueError
            - If ``y_true`` is empty.
            - If ``y_true``, ``y_proxy`` and ``batches`` have different lengths.
            - If ``batches`` contains NaN values (numeric dtype) or None values (non-numeric dtype).
            - If ``confidence_level`` is not in (0, 1).
            - If ``metric_lower_bound >= metric_upper_bound``.
            - If ``threshold`` falls outside ``[metric_lower_bound, metric_upper_bound]``.
            - If any proxy value is NaN or all proxy values are identical.
            - If labeled ``y_true`` values are constant.
            - If any value falls outside ``[metric_lower_bound, metric_upper_bound]``.
            - If batches are interleaved rather than grouped into contiguous blocks.
            - If any batch has fewer than 2 labeled or fewer than 2 unlabeled samples.
            - If proxy values are constant across the prior batches (with ``power_tuning=True``).
        """
        risk_y_true, risk_y_proxy, risk_threshold, batch_codes, batch_n_true, batch_n_proxy = self._preprocess(
            y_true,
            y_proxy,
            batches,
            higher_is_better,
            threshold,
            confidence_level,
            metric_lower_bound,
            metric_upper_bound,
        )
        n_batches = len(batch_n_true)

        risk_batch_estimates = np.empty(n_batches)
        for position in range(n_batches):
            if position == 0 or (not power_tuning):
                tuning_parameter = 1.0
            else:
                earlier_mask = batch_codes < position
                y_true_earlier, y_proxy_labeled_earlier, y_proxy_unlabeled_earlier, _ = _split_labeled_unlabeled(
                    risk_y_true[earlier_mask], risk_y_proxy[earlier_mask]
                )
                tuning_parameter = _compute_tuning_parameter(
                    y_true_earlier, y_proxy_labeled_earlier, y_proxy_unlabeled_earlier, power_tuning
                )

            batch_mask = batch_codes == position
            y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, _ = _split_labeled_unlabeled(
                risk_y_true[batch_mask], risk_y_proxy[batch_mask]
            )
            risk_batch_estimates[position] = _compute_mean_estimate(
                y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, tuning_parameter
            )

        clipped_risk_batch_estimates = np.clip(risk_batch_estimates, 0.0, 1.0)

        miscoverage = 1.0 - confidence_level
        risk_running_means, risk_lower_bounds = _compute_empirical_bernstein_bounds(
            clipped_risk_batch_estimates, risk_threshold, miscoverage
        )
        running_means, confidence_bounds, batch_mean_estimates = self._postprocess(
            risk_running_means,
            risk_lower_bounds,
            risk_batch_estimates,
            higher_is_better,
            metric_lower_bound,
            metric_upper_bound,
        )
        confidence_sequence = EmpiricalBernsteinConfidenceSequence(
            running_mean_estimates=running_means,
            confidence_bounds=confidence_bounds,
        )
        result = PredictionPoweredMeanMonitoringResult(
            metric_name=metric_name,
            monitor_name=self.__class__.__name__,
            higher_is_better=higher_is_better,
            alarm_threshold=threshold,
            confidence_level=confidence_level,
            batch_mean_estimates=batch_mean_estimates,
            confidence_sequence=confidence_sequence,
            batch_n_true=batch_n_true,
            batch_n_proxy=batch_n_proxy,
        )
        return result
