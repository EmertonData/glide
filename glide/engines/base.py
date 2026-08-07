from typing import Protocol, Tuple, TypeVar

from numpy.typing import NDArray

DatasetT = TypeVar("DatasetT")
TuningParameterT = TypeVar("TuningParameterT")


class MeanEstimationEngine(Protocol[DatasetT, TuningParameterT]):
    def preprocess(self, *fields: NDArray) -> DatasetT: ...

    def fit_tuning_parameter(self, dataset: DatasetT, power_tuning: bool) -> TuningParameterT: ...

    def compute_mean_and_std(self, dataset: DatasetT, tuning_parameter: TuningParameterT) -> Tuple[float, float]: ...
