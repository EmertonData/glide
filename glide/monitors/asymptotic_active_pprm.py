import numpy as np
from numpy.typing import NDArray

from glide.confidence_sequences import AsymptoticConfidenceSequence
from glide.engines.asi import ASIDataset, ASIMeanEngine
from glide.mean_monitoring_results import PredictionPoweredMeanMonitoringResult
from glide.monitors.base import AsymptoticRM


class AsymptoticActivePPRM(AsymptoticRM[ASIDataset, float]):
    """Anytime-valid drift monitor combining human and proxy labels under active sampling.

    Computes a per-batch Active Statistical Inference estimate together with its standard
    error. Each labeled sample is reweighted by the inverse of the known probability with
    which it was selected for annotation. A power-tuned weight balances the proxy against
    the labels; it is fitted on the batches that strictly precede the current one, which
    keeps it predictable as required for validity.

    The running mean of the per-batch estimates is then tracked the same way
    ``AsymptoticPPRM`` does for uniformly sampled labels.

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

    Zrnic, Tijana, and Emmanuel J. Candès. "Active statistical inference." In Proceedings
    of the 41st International Conference on Machine Learning, pp. 62993-63010. 2024.

    Gligorić, Kristina, Tijana Zrnic, Cinoo Lee, Emmanuel Candes, and Dan Jurafsky.
    "Can unconfident llm annotations be used for confident conclusions?." In Proceedings
    of the 2025 Conference of the Nations of the Americas Chapter of the Association for
    Computational Linguistics: Human Language Technologies (Volume 1: Long Papers),
    pp. 3514-3533. 2025.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.monitors import AsymptoticActivePPRM
    >>> pre_drift_y_true = np.array([0.0, 0.2, np.nan, np.nan])
    >>> pre_drift_y_proxy = np.array([0.0, 0.2, 0.0, 0.2])
    >>> post_drift_y_true = np.array([0.8, 1.0, np.nan, np.nan])
    >>> post_drift_y_proxy = np.array([0.8, 1.0, 0.8, 1.0])
    >>> y_true = np.hstack([pre_drift_y_true, np.tile(post_drift_y_true, 5)])
    >>> y_proxy = np.hstack([pre_drift_y_proxy, np.tile(post_drift_y_proxy, 5)])
    >>> sampling_probability = np.full(24, 0.5)
    >>> batches = np.repeat(np.arange(6), 4)
    >>> monitor = AsymptoticActivePPRM()
    >>> result = monitor.detect(
    ...     y_true, y_proxy, sampling_probability, batches, higher_is_better=False, threshold=0.5
    ... )
    >>> result.drift_detected
    True
    >>> result.first_alarm_index
    2
    """

    _engine = ASIMeanEngine()

    def detect(
        self,
        y_true: NDArray,
        y_proxy: NDArray,
        sampling_probability: NDArray,
        batches: NDArray,
        *,
        higher_is_better: bool,
        threshold: float,
        metric_name: str = "Metric",
        confidence_level: float = 0.8,
        power_tuning: bool = True,
        tightest_at_batch: int = 10,
    ) -> PredictionPoweredMeanMonitoringResult:
        """Detect a drift of the running mean across a batched, actively sampled dataset.

        Splits the data by batch, computes an Active Statistical Inference estimate and
        its standard error per batch, and builds an anytime-valid asymptotic confidence
        sequence on the running mean of those estimates. An alarm is raised at every
        batch where the sequence crosses the user-supplied ``threshold``. See
        ``AsymptoticPPRM.detect`` for the full explanation of batch ordering,
        growing-history semantics and ``tightest_at_batch``, which all carry over unchanged.

        Parameters
        ----------
        y_true : NDArray
            Array of labeled observations, shape ``(n_samples,)``.
            Labeled entries are finite; unlabeled entries are ``np.nan``.
        y_proxy : NDArray
            Array of proxy predictions, shape ``(n_samples,)``.
            Must be fully populated (no NaN).
        sampling_probability : NDArray
            Array of pre-determined probabilities of being selected for annotation, shape
            ``(n_samples,)``, each in ``[0, 1]``, typically the first output of
            ``ActiveSampler.sample`` applied batch by batch. Entries equal to ``0`` are
            excluded from the computation. A labeled entry must have a strictly positive
            probability and an unlabeled entry a probability strictly below ``1``.
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
            If ``True`` (default), compute the power-tuning weight from all previous
            batches pooled together (the first batch, having no predecessor, uses a fixed
            weight of ``1``). If ``False``, use a fixed weight of ``1`` everywhere.
        tightest_at_batch : int, optional
            The batch index (1-indexed) at which the confidence sequence is tuned to be
            tightest. Defaults to ``10``.

        Returns
        -------
        PredictionPoweredMeanMonitoringResult
            Per-batch estimates, running means, anytime-valid confidence bounds, alarm
            flags and the alarm threshold, all in the original metric orientation.
            ``batch_n_true`` counts labeled samples and ``batch_n_proxy`` counts samples
            with a strictly positive probability, i.e. the samples entering the estimate.

        Raises
        ------
        ValueError
            - If ``batches`` is empty.
            - If ``y_true``, ``y_proxy``, ``sampling_probability`` and ``batches`` have different lengths.
            - If ``batches`` contains NaN values (numeric dtype) or None values (non-numeric dtype).
            - If ``confidence_level`` is not in ``(0.5, 1)``.
            - If any proxy value is NaN, or if proxy values are constant within any batch.
            - If any probability is outside ``[0, 1]``.
            - If a labeled entry has probability ``0``, or an unlabeled entry has probability ``1``.
            - If batches are interleaved rather than grouped into contiguous blocks.
            - If any batch has fewer than 2 labeled or fewer than 2 unlabeled samples
              among those with a strictly positive probability.
            - If ``y_proxy * (labeled / sampling_probability - 1)`` is constant across a prefix set of batches
              (with ``power_tuning=True``).
            - If ``tightest_at_batch`` is not a positive integer.
            - If the accumulated variance of the batch estimates up to ``tightest_at_batch`` is zero.
        """
        batch_codes, batch_mean_estimates, running_means, confidence_bounds = self._detect(
            label_fields={"y_true": y_true, "y_proxy": y_proxy},
            batches=batches,
            identifier_fields={"sampling_probability": sampling_probability},
            higher_is_better=higher_is_better,
            confidence_level=confidence_level,
            tightest_at_batch=tightest_at_batch,
            power_tuning=power_tuning,
        )
        confidence_sequence = AsymptoticConfidenceSequence(
            running_mean_estimates=running_means, confidence_bounds=confidence_bounds
        )
        labeled_mask = ~np.isnan(y_true)
        selectable_mask = sampling_probability > 0
        batch_n_true = np.bincount(batch_codes[labeled_mask])
        batch_n_proxy = np.bincount(batch_codes[selectable_mask])
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
