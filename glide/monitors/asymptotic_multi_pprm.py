import numpy as np
from numpy.typing import NDArray

from glide.confidence_sequences import AsymptoticConfidenceSequence
from glide.engines.multi_ppi import MultiPPIDataset, MultiPPIMeanEngine
from glide.mean_monitoring_results import PredictionPoweredMeanMonitoringResult
from glide.monitors.base import AsymptoticRM


class AsymptoticMultiPPRM(AsymptoticRM[MultiPPIDataset, NDArray]):
    """Anytime-valid drift monitor combining human labels with multiple proxy predictors.

    Computes a per-batch Multi-PPI estimate (a small set of true labels and several
    sets of proxy labels, combined through a power-tuning weight vector fitted on the
    batches that strictly precede it), together with its standard error, and tracks
    the running mean of the per-batch estimates the same way ``AsymptoticPPRM`` does
    for a single proxy.

    References
    ----------
    Waudby-Smith, Ian, David Arbour, Ritwik Sinha, Edward H. Kennedy, and Aaditya
    Ramdas. "Time-uniform central limit theory and asymptotic confidence
    sequences." The Annals of Statistics 52, no. 6 (2024): 2613-2640.

    Zhang, Guangyi, Yunlong Cai, Guanding Yu, and Osvaldo Simeone. "Prediction-powered
    risk monitoring of deployed models for detecting harmful distribution shifts."
    arXiv preprint arXiv:2602.02229 (2026).

    Podkopaev, Aleksandr, and Aaditya Ramdas. "Tracking the risk of a deployed model
    and detecting harmful distribution shifts." International Conference on Learning
    Representations (ICLR), 2022.

    Shan, Jiawei, Zhifeng Chen, Yiming Dong, Yazhen Wang, and Jiwei Zhao. "SADA: Safe
    and Adaptive Aggregation of Multiple Black-Box Predictions in Semi-Supervised
    Learning." arXiv preprint arXiv:2509.21707 (2025).

    Examples
    --------
    >>> import numpy as np
    >>> from glide.monitors import AsymptoticMultiPPRM
    >>> pre_drift_y_true = np.array([0.0, 0.2, np.nan, np.nan])
    >>> pre_drift_y_proxies = np.array([[0.0, 0.1], [0.2, 0.3], [0.05, 0.25], [0.15, 0.05]])
    >>> post_drift_y_true = np.array([0.8, 1.0, np.nan, np.nan])
    >>> post_drift_y_proxies = pre_drift_y_proxies + 0.8
    >>> y_true = np.hstack([pre_drift_y_true, np.tile(post_drift_y_true, 5)])
    >>> y_proxies = np.vstack([pre_drift_y_proxies, np.tile(post_drift_y_proxies, (5, 1))])
    >>> batches = np.repeat(np.arange(6), 4)
    >>> monitor = AsymptoticMultiPPRM()
    >>> result = monitor.detect(y_true, y_proxies, batches, higher_is_better=False, threshold=0.5)
    >>> result.drift_detected
    True
    >>> result.first_alarm_index
    2
    """

    _engine = MultiPPIMeanEngine()

    def detect(
        self,
        y_true: NDArray,
        y_proxies: NDArray,
        batches: NDArray,
        higher_is_better: bool,
        threshold: float,
        metric_name: str = "Metric",
        confidence_level: float = 0.8,
        power_tuning: bool = True,
        tightest_at_batch: int = 10,
    ) -> PredictionPoweredMeanMonitoringResult:
        """Detect a drift of the running mean across a batched dataset.

        Splits the data by batch, computes a Multi-PPI estimate and its standard error
        per batch, and builds an anytime-valid asymptotic confidence sequence on the
        running mean of those estimates. An alarm is raised at every batch where the
        sequence crosses the user-supplied ``threshold``. See ``AsymptoticPPRM.detect``
        for the full explanation of batch ordering, growing-history semantics, and
        ``tightest_at_batch``, which all carry over unchanged.

        Parameters
        ----------
        y_true : NDArray
            Array of labeled observations, shape ``(n_samples,)``.
            Labeled entries are finite; unlabeled entries are ``np.nan``.
        y_proxies : NDArray
            2D array of proxy predictions, shape ``(n_samples, n_proxies)``.
            Must be fully populated (no NaN). Each column must have nonzero variance
            when ``power_tuning=True``.
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
        power_tuning : bool, optional
            If ``True`` (default), compute the power-tuning parameter vector of each
            batch on all previous batches (the first batch, having no predecessor,
            uses a fixed weight). If ``False``, use a fixed weight everywhere.
        tightest_at_batch : int, optional
            The batch index (1-indexed) at which the confidence sequence is tuned to
            be tightest. Defaults to ``10``.

        Returns
        -------
        PredictionPoweredMeanMonitoringResult
            Per-batch estimates, running means, anytime-valid confidence bounds,
            alarm flags, and the alarm threshold, all in the original metric
            orientation. ``batch_n_proxy`` counts the shared row count of
            ``y_proxies``, unaffected by the number of proxy columns.

        Raises
        ------
        ValueError
            - If ``batches`` is empty.
            - If ``y_true``, ``y_proxies`` and ``batches`` have different lengths.
            - If ``y_proxies`` is not a 2D array.
            - If ``batches`` contains NaN values (numeric dtype) or None values (non-numeric dtype).
            - If ``confidence_level`` is not in ``(0.5, 1)``.
            - If any proxy value is NaN.
            - If batches are interleaved rather than grouped into contiguous blocks.
            - If any batch has fewer than 2 labeled or fewer than 2 unlabeled samples.
            - If any proxy column is constant across a prefix set of batches (with ``power_tuning=True``).
            - If ``tightest_at_batch`` is not a positive integer.
            - If the accumulated variance of the batch estimates up to ``tightest_at_batch`` is zero.
        """
        batch_codes, batch_mean_estimates, running_means, confidence_bounds = self._detect(
            fields=[y_true, y_proxies],
            field_names=["y_true", "y_proxies"],
            batches=batches,
            higher_is_better=higher_is_better,
            confidence_level=confidence_level,
            tightest_at_batch=tightest_at_batch,
            power_tuning=power_tuning,
        )
        confidence_sequence = AsymptoticConfidenceSequence(
            running_mean_estimates=running_means, confidence_bounds=confidence_bounds
        )
        labeled_mask = ~np.isnan(y_true)
        batch_n_true = np.bincount(batch_codes[labeled_mask])
        batch_n_proxy = np.bincount(batch_codes)
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
