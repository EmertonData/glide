import json
from pathlib import Path
from typing import List

import pandas as pd
from datasets import Dataset

# TODO: set to the agreed HuggingFace repo slug, e.g. "glide-py/spider-text-to-sql"
HF_REPO = "glide-py/spider-text-to-sql"


def _load_jsonl(path: Path) -> pd.DataFrame:
    records: List = [json.loads(line) for line in open(path) if line.strip()]
    df = pd.DataFrame(records)
    return df


def main() -> None:
    predictions = _load_jsonl(Path("data/predictions.jsonl"))
    judge_labels = _load_jsonl(Path("data/llm_judge_labels.jsonl"))
    human_labels = _load_jsonl(Path("data/human_labels.jsonl"))

    df = predictions.merge(judge_labels[["example_id", "llm_judge_label"]], on="example_id")
    df = df.merge(human_labels[["example_id", "human_label"]], on="example_id")
    df = df[["example_id", "db_id", "question", "gold_sql", "predicted_sql", "llm_judge_label", "human_label"]]

    df["agreement"] = (df["llm_judge_label"] == df["human_label"]).astype(int)
    summary = (
        df.groupby("db_id")
        .agg(
            count=("human_label", "count"),
            human_accuracy=("human_label", "mean"),
            judge_accuracy=("llm_judge_label", "mean"),
            agreement_rate=("agreement", "mean"),
        )
        .round(3)
    )
    print(summary.to_string())

    df = df.drop(columns=["agreement"])
    df.to_parquet("data/spider_dataset.parquet", index=False)

    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset.push_to_hub(HF_REPO)
    print(f"Pushed {len(df)} examples to {HF_REPO}.")


if __name__ == "__main__":
    main()
