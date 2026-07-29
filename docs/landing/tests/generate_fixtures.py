import json
from pathlib import Path

import numpy as np

from glide.estimators import ClassicalMeanEstimator, PPIMeanEstimator
from glide.simulators import generate_binary_dataset

CONFIDENCE_LEVEL = 0.95

CASES = [
    {"totalSize": 20000, "humanSize": 4000, "trueMean": 0.77, "proxyMean": 0.76, "correlation": 0.6},
    {"totalSize": 20000, "humanSize": 2000, "trueMean": 0.82, "proxyMean": 0.70, "correlation": 0.3},
]


def build_fixtures() -> dict:
    simulate_fixtures = []
    for case in CASES:
        human_size = int(case["humanSize"])
        y_true, y_proxy = generate_binary_dataset(
            n_samples=int(case["totalSize"]),
            true_mean=case["trueMean"],
            proxy_mean=case["proxyMean"],
            correlation=case["correlation"],
            random_seed=0,
        )
        y_true_masked = y_true.copy()
        y_true_masked[human_size:] = np.nan
        ppi_result = PPIMeanEstimator().estimate(
            y_true_masked, y_proxy, power_tuning=True, confidence_level=CONFIDENCE_LEVEL
        )
        human_result = ClassicalMeanEstimator().estimate(y_true[:human_size], confidence_level=CONFIDENCE_LEVEL)
        simulate_fixtures.append(
            {
                "params": case,
                "ppi_half_width": ppi_result.confidence_interval.width / 2,
                "human_half_width": human_result.confidence_interval.width / 2,
                "effective_sample_size": ppi_result.effective_sample_size,
            }
        )
    fixtures = {"simulate": simulate_fixtures}
    return fixtures


if __name__ == "__main__":
    output_path = Path(__file__).parent / "fixtures.json"
    fixtures = build_fixtures()
    output_path.write_text(json.dumps(fixtures, indent=2))
