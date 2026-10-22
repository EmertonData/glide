import numpy as np
from numpy.typing import NDArray

from glide.confidence_sequences import AsymptoticConfidenceSequence
from glide.core.validation import _validate_clusters_unique_across_batches
from glide.engines.clustered_classical import ClusteredClassicalMeanEngine
from glide.mean_monitoring_results import ClassicalMeanMonitoringResult
from glide.monitors.base import AsymptoticRM


class AsymptoticClusteredClassicalRM(AsymptoticRM[NDArray, None]):
    """Anytime-valid label-only drift monitor over clustered data.

    Computes a per-batch estimate by reducing each batch's labeled observations
    to per-cluster means and taking the classical mean of those means, together
    with its standard error, then tracks the running mean of the per-batch
    estimates the same way ``AsymptoticClassicalRM`` does for independent,
    unclustered samples. Samples of the same cluster are correlated, so treating
    each cluster as a single sampling unit keeps the standard errors, and hence
    the false-alarm guarantee, valid.

    References
    ----------
    Waudby-Smith, Ian, David Arbour, Ritwik Sinha, Edward H. Kennedy, and Aaditya
    Ramdas. "Time-uniform central limit theory and asymptotic confidence
    sequences." The Annals of Statistics 52, no. 6 (2024): 2613-2640.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.monitors import AsymptoticClusteredClassicalRM
    >>> pre_drift_batch = np.array([0.0, 0.2, 0.1, 0.3, np.nan, np.nan])
    >>> post_drift_batch = np.array([0.8, 1.0, 0.7, 0.9, np.nan, np.nan])
    >>> y = np.hstack([pre_drift_batch, np.tile(post_drift_batch, 5)])
    >>> clusters = np.repeat(np.arange(18), 2)
    >>> batches = np.repeat(np.arange(6), 6)
    >>> monitor = AsymptoticClusteredClassicalRM()
    >>> result = monitor.detect(y, clusters, batches, higher_is_better=False, threshold=0.5)
    >>> result.drift_detected
    True
    >>> result.first_alarm_index
    2
    """

    _engine = ClusteredClassicalMeanEngine()

    def detect(
        self,
        y: NDArray,
        clusters: NDArray,
        batches: NDArray,
        *,
        higher_is_better: bool,
        threshold: float,
        metric_name: str = "Metric",
        confidence_level: float = 0.8,
        tightest_at_batch: int = 10,
    ) -> ClassicalMeanMonitoringResult:
        """Detect a drift of the running mean across a batched, clustered dataset.

        Splits the data by batch, reduces each batch's labeled observations to
        per-cluster means, computes the classical mean and its standard error on
        those means, and builds an anytime-valid asymptotic confidence sequence on
        the running mean of the per-batch estimates. An alarm is raised at every
        batch where the sequence crosses the user-supplied ``threshold``. See
        ``AsymptoticClassicalRM.detect`` for the full explanation of batch ordering,
        growing-history semantics, and ``tightest_at_batch``, which all carry over
        unchanged.

        Parameters
        ----------
        y : NDArray
            Array of observations, shape ``(n_samples,)``. Unlabeled entries are
            ``np.nan`` and are dropped per batch; every batch must keep labeled
            entries in at least 2 clusters.
        clusters : NDArray
            Array of cluster identifiers, shape ``(n_samples,)``. Any hashable label
            type works (integers, strings, ...). Must be unique across the whole
            stream: an identifier reused in two different batches raises, since
            their rows would otherwise be silently pooled into a single artificial
            cluster once both fall inside the same prefix.
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
            aggregated across its clusters.

        Raises
        ------
        ValueError
            - If ``batches`` is empty.
            - If ``y``, ``clusters`` and ``batches`` have different lengths.
            - If ``batches`` contains NaN values (numeric dtype) or None values (non-numeric dtype).
            - If ``clusters`` contains NaN values (numeric dtype) or None values (non-numeric dtype).
            - If ``confidence_level`` is not in ``(0.5, 1)``.
            - If batches are interleaved rather than grouped into contiguous blocks.
            - If any batch has labeled observations in fewer than 2 clusters.
            - If a cluster identifier's rows span more than one batch.
            - If ``tightest_at_batch`` is not a positive integer.
            - If the accumulated variance of the batch estimates up to ``tightest_at_batch`` is zero.
        """
        _validate_clusters_unique_across_batches(clusters, batches)
        batch_codes, batch_mean_estimates, running_means, confidence_bounds = self._detect(
            label_fields={"y": y},
            identifier_fields={"clusters": clusters},
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
