import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
from datasets import Dataset


def _load_jsonl(path: Path) -> pd.DataFrame:
    records: List = [json.loads(line) for line in open(path) if line.strip()]
    df = pd.DataFrame(records)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge predictions with LLM judge and human labels, then push to HuggingFace Hub."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Path to the predictions JSONL file.",
    )
    parser.add_argument(
        "--judge-labels",
        type=Path,
        help="Path to the LLM judge labels JSONL file.",
    )
    parser.add_argument(
        "--human-labels",
        type=Path,
        help="Path to the human labels JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/spider_dataset.parquet"),
        help="Path for the output Parquet file. (default: data/spider_dataset.parquet)",
    )
    parser.add_argument(
        "--hf-repo",
        default="imerad-kv/spider-text-to-sql",
        help="HuggingFace Hub repository slug to push the dataset to. (default: glide-py/spider-text-to-sql)",
    )
    args = parser.parse_args()

    predictions = _load_jsonl(args.predictions)
    judge_labels = _load_jsonl(args.judge_labels)
    human_labels = _load_jsonl(args.human_labels)

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
    df.to_parquet(args.output, index=False)

    dataset = Dataset.from_pandas(df, preserve_index=False)
    dataset.push_to_hub(args.hf_repo)
    print(f"Pushed {len(df)} examples to {args.hf_repo}.")


if __name__ == "__main__":
    main()
