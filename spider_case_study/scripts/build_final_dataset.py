"""
Merge predictions, LLM judge labels, and human labels into a single Parquet dataset.

Run from the repo root after all three generation scripts:
    python spider_case_study/scripts/build_final_dataset.py

Outputs:
    spider_case_study/data/spider_ppi_dataset.parquet
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_PATH = DATA_DIR / "spider_ppi_dataset.parquet"


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_json(path, lines=True)


def main() -> None:
    print("Loading intermediate files...")
    predictions = _read_jsonl(DATA_DIR / "predictions.jsonl")
    llm_judge = _read_jsonl(DATA_DIR / "llm_judge_labels.jsonl")
    human = _read_jsonl(DATA_DIR / "human_labels.jsonl")

    print(f"  predictions     : {len(predictions)} rows")
    print(f"  llm_judge_labels: {len(llm_judge)} rows")
    print(f"  human_labels    : {len(human)} rows")

    df = predictions.merge(llm_judge, on="example_id", how="left")
    df = df.merge(human, on="example_id", how="left")

    initial_count = len(df)
    df = df.dropna(subset=["llm_judge_label", "human_label"])
    dropped = initial_count - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with missing labels (parse errors or missing API responses)")

    df["llm_judge_label"] = df["llm_judge_label"].astype("int8")
    df["human_label"] = df["human_label"].astype("int8")

    column_order = [
        "example_id",
        "db_id",
        "question",
        "gold_sql",
        "predicted_sql",
        "llm_judge_label",
        "llm_judge_reasoning",
        "human_label",
        "human_reasoning",
    ]
    df = df[column_order]

    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nDataset saved to {OUTPUT_PATH}")
    print(f"\n{'=' * 60}")
    print(f"Total examples    : {len(df)}")
    print(f"Databases         : {df['db_id'].nunique()}")
    print(f"LLM judge rate    : {df['llm_judge_label'].mean():.1%}")
    print(f"Human label rate  : {df['human_label'].mean():.1%}")
    print("\nPer-database summary:")
    summary = (
        df.groupby("db_id")
        .agg(
            count=("example_id", "count"),
            judge_rate=("llm_judge_label", "mean"),
            human_rate=("human_label", "mean"),
        )
        .sort_values("count", ascending=False)
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
