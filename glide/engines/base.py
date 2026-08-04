from typing import Protocol, Tuple, TypeVar, Union

from numpy.typing import NDArray

DatasetT = TypeVar("DatasetT")
TuningParameter = Union[float, NDArray, None]


class MeanEstimateEngine(Protocol[DatasetT]):
    def prepare(self, *fields: NDArray) -> DatasetT: ...

    def fit_tuning_parameter(self, dataset: DatasetT, power_tuning: bool) -> TuningParameter: ...

    def compute_mean_and_std(self, dataset: DatasetT, tuning_parameter: TuningParameter) -> Tuple[float, float]: ...
