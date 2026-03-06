from dataclasses import dataclass

from scipy.stats import norm


@dataclass
class CLTConfidenceInterval:
    mean: float
    std: float
    confidence_level: float = 0.95

    def _z_score(self) -> float:
        z_score = norm.ppf((1 + self.confidence_level) / 2)
        return z_score

    @property
    def lower_bound(self) -> float:
        result = self.mean - self.std * self._z_score()
        return result

    @property
    def upper_bound(self) -> float:
        result = self.mean + self.std * self._z_score()
        return result
