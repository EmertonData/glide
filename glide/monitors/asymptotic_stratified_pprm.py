import numpy as np
from numpy.typing import NDArray

from glide.confidence_sequences import AsymptoticConfidenceSequence
from glide.engines.stratified_ppi import StratifiedPPIDataset, StratifiedPPIMeanEngine, StratifiedTuningParameter
from glide.mean_monitoring_results import PredictionPoweredMeanMonitoringResult
from glide.monitors.base import AsymptoticRM


class AsymptoticStratifiedPPRM(AsymptoticRM[StratifiedPPIDataset, StratifiedTuningParameter]):
    """Anytime-valid drift monitor combining human and proxy labels over a stratified stream.

    Computes a per-batch Stratified PPI estimate (each batch's samples are split into
    strata, a power-tuned PPI++ estimate is computed independently within each stratum
    using a weight fitted on the batches that strictly precede it, and the per-stratum
    estimates are combined with population-proportional weights), together with its
    standard error, and tracks the running mean of the per-batch estimates the same
    way ``AsymptoticPPRM`` does for an unstratified stream. The false-alarm guarantee
    is asymptotic: each batch needs enough labeled and proxy samples, within each of
    its strata, for its Stratified PPI estimate to be approximately Gaussian with a
    consistently estimated variance.

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

    Fisch, Adam, Joshua Maynez, R. Alex Hofer, Bhuwan Dhingra, Amir Globerson, and
    William W. Cohen. "Stratified prediction-powered inference for effective hybrid
    evaluation of language models." Advances in Neural Information Processing
    Systems 37 (2024): 111489-111514.

    Fogliato, Riccardo, Pratik Patil, Mathew Monfort, and Pietro Perona. "A framework
    for efficient model evaluation through stratification, sampling, and estimation."
    In European Conference on Computer Vision, pp. 140-158. Cham: Springer Nature
    Switzerland, 2024.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.monitors import AsymptoticStratifiedPPRM
    >>> pre_drift_y_true = np.array([0.0, 0.2, np.nan, np.nan, 0.1, 0.3, np.nan, np.nan])
    >>> pre_drift_y_proxy = np.array([0.0, 0.2, 0.0, 0.2, 0.1, 0.3, 0.1, 0.3])
    >>> post_drift_y_true = np.array([0.8, 1.0, np.nan, np.nan, 0.7, 0.9, np.nan, np.nan])
    >>> post_drift_y_proxy = np.array([0.8, 1.0, 0.8, 1.0, 0.7, 0.9, 0.7, 0.9])
    >>> y_true = np.hstack([pre_drift_y_true, np.tile(post_drift_y_true, 5)])
    >>> y_proxy = np.hstack([pre_drift_y_proxy, np.tile(post_drift_y_proxy, 5)])
    >>> groups = np.tile([0, 0, 0, 0, 1, 1, 1, 1], 6)
    >>> batches = np.repeat(np.arange(6), 8)
    >>> monitor = AsymptoticStratifiedPPRM()
    >>> result = monitor.detect(y_true, y_proxy, groups, batches, higher_is_better=False, threshold=0.5)
    >>> result.drift_detected
    True
    >>> result.first_alarm_index
    2
    """

    _engine = StratifiedPPIMeanEngine()

    def detect(
        self,
        y_true: NDArray,
        y_proxy: NDArray,
        groups: NDArray,
        batches: NDArray,
        higher_is_better: bool,
        threshold: float,
        metric_name: str = "Metric",
        confidence_level: float = 0.8,
        power_tuning: bool = True,
        tightest_at_batch: int = 10,
    ) -> PredictionPoweredMeanMonitoringResult:
        """Detect a drift of the running mean across a batched, stratified dataset.

        Splits the data by batch, then by stratum within each batch, computes a
        Stratified PPI estimate and its standard error per batch, and builds an
        anytime-valid asymptotic confidence sequence on the running mean of those
        estimates. An alarm is raised at every batch where the sequence crosses the
        user-supplied ``threshold``. See ``AsymptoticPPRM.detect`` for the full
        explanation of batch ordering, growing-history semantics, and
        ``tightest_at_batch``, which all carry over unchanged.

        Parameters
        ----------
        y_true : NDArray
            Array of labeled observations, shape ``(n_samples,)``.
            Labeled entries are finite; unlabeled entries are ``np.nan``.
        y_proxy : NDArray
            Array of proxy predictions, shape ``(n_samples,)``.
            Must be fully populated (no NaN).
        groups : NDArray
            Array of stratum identifiers, shape ``(n_samples,)``. Unique values
            within a batch define that batch's strata; a stratum need not appear in
            every batch. One appearing for the first time anywhere in the stream
            falls back to an unpowered estimate for that batch only (see ``power_tuning``).
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
            If ``True`` (default), compute each stratum's power-tuning parameter on
            all previous batches (a stratum's first appearance in the stream uses a
            fixed weight, having no predecessor to fit on). If ``False``, use a fixed
            weight everywhere.
        tightest_at_batch : int, optional
            The batch index (1-indexed) at which the confidence sequence is tuned to
            be tightest. Defaults to ``10``.

        Returns
        -------
        PredictionPoweredMeanMonitoringResult
            Per-batch estimates, running means, anytime-valid confidence bounds,
            alarm flags, and the alarm threshold, all in the original metric
            orientation. ``batch_n_true``/``batch_n_proxy`` aggregate across all of a
            batch's strata.

        Raises
        ------
        ValueError
            - If ``batches`` is empty.
            - If ``y_true``, ``y_proxy``, ``groups`` and ``batches`` have different lengths.
            - If ``batches`` or ``groups`` contains NaN values (numeric dtype) or None values (non-numeric dtype).
            - If ``confidence_level`` is not in ``(0.5, 1)``.
            - If any proxy value is NaN.
            - If batches are interleaved rather than grouped into contiguous blocks.
            - If any stratum in any batch has fewer than 2 labeled or fewer than 2 unlabeled samples.
            - If a stratum's proxy values are constant across a prefix set of batches (with ``power_tuning=True``).
            - If ``tightest_at_batch`` is not a positive integer.
            - If the accumulated variance of the batch estimates up to ``tightest_at_batch`` is zero.
        """
        batch_codes, batch_mean_estimates, running_means, confidence_bounds = self._detect(
            fields=[y_true, y_proxy, groups],
            field_names=["y_true", "y_proxy", "groups"],
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
