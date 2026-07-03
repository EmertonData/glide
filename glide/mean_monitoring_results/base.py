from dataclasses import dataclass
from typing import List, Optional

from numpy.typing import NDArray

from glide.confidence_sequences import ConfidenceSequence


@dataclass(repr=False)
class MeanMonitoringResult:
    """Result of a drift-monitoring procedure over a batched dataset.

    Holds the embedded ``confidence_sequence`` and exposes its per-look arrays as
    properties. Every array is indexed by batch in chronological order and is
    expressed in the original metric units. For batch ``t``:

    - ``running_means[t]`` is the mean of the per-batch estimates up to ``t``;
    - ``confidence_bounds[t]`` is the anytime-valid bound on that running mean,
      on the side where drift is harmful (a lower bound when the metric is a
      risk, an upper bound when it is a performance);
    - ``alarms[t]`` is ``True`` once ``confidence_bounds[t]`` has crossed
      ``alarm_threshold``, signalling a drift.

    ``alarm_threshold`` is the user-supplied business threshold the confidence
    bound is tested against: the worst metric value the user is willing to
    tolerate, in metric units.

    Because the threshold is a known constant, the only false-alarm budget is
    the embedded sequence's, reported as ``confidence_level``: under no drift,
    the probability of ever raising a false alarm is at most
    ``1 - confidence_level``.
    """

    metric_name: str
    monitor_name: str
    higher_is_better: bool
    alarm_threshold: float
    confidence_level: float
    batch_identifiers: NDArray
    batch_mean_estimates: NDArray
    confidence_sequence: ConfidenceSequence

    @property
    def running_means(self) -> NDArray:
        result = self.confidence_sequence.running_mean_estimates
        return result

    @property
    def confidence_bounds(self) -> NDArray:
        result = self.confidence_sequence.confidence_bounds
        return result

    @property
    def alarms(self) -> NDArray:
        alternative = "smaller" if self.higher_is_better else "larger"
        result = self.confidence_sequence.test_null_hypothesis(self.alarm_threshold, alternative)
        return result

    @property
    def n_batches(self) -> int:
        result = len(self.batch_mean_estimates)
        return result

    @property
    def drift_detected(self) -> bool:
        result = bool(self.alarms.any())
        return result

    @property
    def first_alarm_index(self) -> Optional[int]:
        if not self.drift_detected:
            return None
        result = int(self.alarms.argmax())
        return result

    def __str__(self) -> str:
        lines: List[str] = [
            f"Metric: {self.metric_name}",
            f"Monitor: {self.monitor_name}",
            f"Number of Batches: {self.n_batches}",
            f"Drift Detected: {self.drift_detected}",
            f"First Alarm Index: {self.first_alarm_index}",
            f"Alarm Threshold: {self.alarm_threshold:.3f}",
            f"Running Mean: {self.running_means[-1]:.3f}",
            f"Confidence Bound: {self.confidence_bounds[-1]:.3f}",
            f"Confidence Level: {self.confidence_level:.2f}",
        ]
        return "\n".join(lines)

    def summary(self) -> str:
        """Return a formatted summary of the monitoring result."""
        return self.__str__()

    def __repr__(self) -> str:
        return self.__str__()
