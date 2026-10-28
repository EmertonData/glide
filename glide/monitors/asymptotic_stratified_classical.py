import numpy as np
from numpy.typing import NDArray

from glide.confidence_sequences import AsymptoticConfidenceSequence
from glide.engines.stratified_classical import StratifiedClassicalDataset, StratifiedClassicalMeanEngine
from glide.mean_monitoring_results import ClassicalMeanMonitoringResult
from glide.monitors.base import AsymptoticRM


class AsymptoticStratifiedClassicalRM(AsymptoticRM[StratifiedClassicalDataset, None]):
    """Anytime-valid label-only drift monitor over a stratified stream.

    Computes a per-batch Stratified classical estimate together with its standard
    error. Each batch's labeled samples are split into strata, a sample mean is
    computed independently within each stratum, and the per-stratum means are
    combined with population-proportional weights. The monitor then tracks the
    running mean of the per-batch estimates the same way ``AsymptoticClassicalRM``
    does for an unstratified stream. The false-alarm guarantee is asymptotic: each
    batch needs enough labeled samples within each of its strata for its estimate
    to be approximately Gaussian with a consistently estimated variance.

    References
    ----------
    Waudby-Smith, Ian, David Arbour, Ritwik Sinha, Edward H. Kennedy, and Aaditya
    Ramdas. "Time-uniform central limit theory and asymptotic confidence
    sequences." The Annals of Statistics 52, no. 6 (2024): 2613-2640.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.monitors import AsymptoticStratifiedClassicalRM
    >>> pre_drift_batch = np.array([0.0, 0.2, np.nan, np.nan, 0.1, 0.3, np.nan, np.nan])
    >>> post_drift_batch = np.array([0.8, 1.0, np.nan, np.nan, 0.7, 0.9, np.nan, np.nan])
    >>> y = np.hstack([pre_drift_batch, np.tile(post_drift_batch, 5)])
    >>> groups = np.tile([0, 0, 0, 0, 1, 1, 1, 1], 6)
    >>> batches = np.repeat(np.arange(6), 8)
    >>> monitor = AsymptoticStratifiedClassicalRM()
    >>> result = monitor.detect(y, groups, batches, higher_is_better=False, threshold=0.5)
    >>> result.drift_detected
    True
    >>> result.first_alarm_index
    2
    """

    _engine = StratifiedClassicalMeanEngine()

    def detect(
        self,
        y: NDArray,
        groups: NDArray,
        batches: NDArray,
        *,
        higher_is_better: bool,
        threshold: float,
        metric_name: str = "Metric",
        confidence_level: float = 0.8,
        tightest_at_batch: int = 10,
    ) -> ClassicalMeanMonitoringResult:
        """Detect a drift of the running mean across a batched, stratified dataset.

        Splits the data by batch, then by stratum within each batch, computes a
        stratified classical estimate and its standard error per batch, and builds
        an anytime-valid asymptotic confidence sequence on the running mean of those
        estimates. An alarm is raised at every batch where the sequence crosses the
        user-supplied ``threshold``. See ``AsymptoticClassicalRM.detect`` for the
        full explanation of batch ordering, growing-history semantics, and
        ``tightest_at_batch``, which all carry over unchanged.

        Parameters
        ----------
        y : NDArray
            Array of observations, shape ``(n_samples,)``. Unlabeled entries are
            ``np.nan`` and are dropped per stratum; every stratum of every batch
            must keep at least 2 labeled entries. Unlabeled entries still count
            towards the population weight of their stratum.
        groups : NDArray
            Array of stratum identifiers, shape ``(n_samples,)``. Unique values
            within a batch define that batch's strata; a stratum need not appear in
            every batch.
        batches : NDArray
            Array of batch identifiers, shape ``(n_samples,)``. Rows must be ordered
            oldest batch first and grouped into contiguous blocks. Identifier values
            are not compared, so any hashable label type works (integers, dates,
            free-form strings).
        higher_is_better : bool
            ``False`` when the metric is a risk (drift means the metric increased),
            ``True`` when it is a performance (drift means the metric decreased).
        threshold : float
            The metric value the running mean is monitored against, in metric units.
        metric_name : str, optional
            Human-readable label for the metric. Defaults to ``"Metric"``.
        confidence_level : float, optional
            How confident each alarm should be. Must be in ``(0.5, 1)``. Defaults to ``0.8``.
        tightest_at_batch : int, optional
            The batch index (1-indexed) at which the confidence sequence is tuned to
            be tightest. Defaults to ``10``.

        Returns
        -------
        ClassicalMeanMonitoringResult
            Per-batch estimates, running means, anytime-valid confidence bounds,
            alarm flags, and the alarm threshold, all in the original metric
            orientation. ``batch_n`` counts the labeled observations of each batch,
            aggregated across its strata.

        Raises
        ------
        ValueError
            - If ``batches`` is empty.
            - If ``y``, ``groups`` and ``batches`` have different lengths.
            - If ``batches`` or ``groups`` contains NaN values (numeric dtype) or None values (non-numeric dtype).
            - If ``confidence_level`` is not in ``(0.5, 1)``.
            - If batches are interleaved rather than grouped into contiguous blocks.
            - If any stratum in any batch has fewer than 2 labeled (non-NaN) samples.
            - If ``tightest_at_batch`` is not a positive integer.
            - If the accumulated variance of the batch estimates up to ``tightest_at_batch`` is zero.
        """
        batch_codes, batch_mean_estimates, running_means, confidence_bounds = self._detect(
            label_fields={"y": y},
            identifier_fields={"groups": groups},
            batches=batches,
            higher_is_better=higher_is_better,
            confidence_level=confidence_level,
            tightest_at_batch=tightest_at_batch,
            power_tuning=False,
        )
        confidence_sequence = AsymptoticConfidenceSequence(
            running_mean_estimates=running_means, confidence_bounds=confidence_bounds
        )
        labeled_mask = ~np.isnan(y)
        batch_n = np.bincount(batch_codes[labeled_mask])
        result = ClassicalMeanMonitoringResult(
            metric_name=metric_name,
            monitor_name=self.__class__.__name__,
            higher_is_better=higher_is_better,
            alarm_threshold=threshold,
            confidence_level=confidence_level,
            batch_mean_estimates=batch_mean_estimates,
            confidence_sequence=confidence_sequence,
            batch_n=batch_n,
        )
        return result
