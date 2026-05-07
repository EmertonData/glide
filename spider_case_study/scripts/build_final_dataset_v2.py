"""
Build the final evaluation dataset by loading predictions.jsonl and simulating
correlated binary labels (llm_judge_label, human_label) using GLIDE's
generate_stratified_binary_dataset simulator.

Strata are assigned by splitting rows evenly across 5 groups. Label distributions
are calibrated to reflect a realistic ~80% accuracy system with a slightly
over-confident LLM judge and strong proxy-human correlation (~0.9).

Run from the repo root:
    python spider_case_study/scripts/build_final_dataset_v2.py
"""

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from glide.simulators import generate_stratified_binary_dataset

N_STRATA = 5
RANDOM_SEED = 42

# Per-stratum true (human) accuracy — all around 80%
TRUE_MEANS: List[float] = [0.77, 0.78, 0.77, 0.81, 0.78]
# Per-stratum proxy (LLM judge) accuracy — slightly over-optimistic
PROXY_MEANS: List[float] = [0.86, 0.89, 0.78, 0.89, 0.85]
# Proxy-human correlation per stratum
CORRELATIONS: List[float] = [0.73, 0.66, 0.92, 0.72, 0.79]

INPUT_PATH = Path(__file__).parent.parent / "data" / "predictions.jsonl"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "spider_ppi_dataset.parquet"

COLUMN_ORDER = [
    "example_id",
    "db_id",
    "question",
    "predicted_sql",
    "llm_judge_label",
    "human_label",
]


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Predictions file not found: {INPUT_PATH}\nRun generate_predictions.py first.")

    rows = []
    with open(INPUT_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    n = len(df)
    print(f"Loaded {n} predictions from {INPUT_PATH}")

    # Assign strata round-robin across rows
    df["db_id"] = [i % N_STRATA for i in range(n)]

    # Count rows per stratum (in stratum order, as expected by the simulator)
    n_per_stratum: List[int] = [int((df["db_id"] == k).sum()) for k in range(N_STRATA)]

    # Generate correlated binary labels — all rows labeled (n_unlabeled=0)
    y_true, y_proxy, _ = generate_stratified_binary_dataset(
        n_labeled=n_per_stratum,
        n_unlabeled=[0] * N_STRATA,
        true_mean=TRUE_MEANS,
        proxy_mean=PROXY_MEANS,
        correlation=CORRELATIONS,
        random_seed=RANDOM_SEED,
    )

    # Simulator outputs rows stratum-by-stratum; sort df to match, assign, restore order
    df["_order"] = np.arange(n)
    df_sorted = df.sort_values("db_id").reset_index(drop=True)
    df_sorted["human_label"] = y_true.astype("int8")
    df_sorted["llm_judge_label"] = y_proxy.astype("int8")
    df = df_sorted.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    df = df[COLUMN_ORDER]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nDataset saved to {OUTPUT_PATH}")
    print(f"\n{'=' * 50}")
    print(f"Total examples    : {len(df)}")
    print(f"Strata            : {df['db_id'].nunique()}")
    print(f"LLM judge rate    : {df['llm_judge_label'].mean():.1%}")
    print(f"Human label rate  : {df['human_label'].mean():.1%}")
    print("\nPer-stratum summary:")
    summary = df.groupby("db_id").agg(
        count=("example_id", "count"),
        judge_rate=("llm_judge_label", "mean"),
        human_rate=("human_label", "mean"),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
