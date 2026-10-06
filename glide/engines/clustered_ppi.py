from typing import Tuple

from numpy.typing import NDArray

from glide.engines.clustered_core import _preprocess
from glide.engines.ppi import PPIDataset, PPIMeanEngine

ClusteredPPIDataset = PPIDataset


class ClusteredPPIMeanEngine:
    def __init__(self) -> None:
        self._ppi_engine = PPIMeanEngine()

    def preprocess(self, y_true: NDArray, y_proxy: NDArray, clusters: NDArray) -> ClusteredPPIDataset:
        clustered_dataset = _preprocess(y_true, y_proxy, clusters)
        return clustered_dataset

    def fit_tuning_parameter(self, dataset: ClusteredPPIDataset, *, power_tuning: bool) -> float:
        tuning_parameter = self._ppi_engine.fit_tuning_parameter(dataset, power_tuning=power_tuning)
        return tuning_parameter

    def compute_mean_and_std(self, dataset: ClusteredPPIDataset, tuning_parameter: float) -> Tuple[float, float]:
        mean, std = self._ppi_engine.compute_mean_and_std(dataset, tuning_parameter)
        return mean, std
