import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import openai
from _utils import (
    SQL_CORRECTNESS_CRITERIA,
    _call_with_retry,
    _load_checkpoint,
    _load_schemas,
    _strip_markdown_fence,
)

SYSTEM_PROMPT = (
    "You are an expert SQL writer. Given a natural language question and a database schema, "
    "write the SQL query that answers the question. Return ONLY the SQL query, no explanation."
)

USER_TEMPLATE = (
    "Database: {db_id}\n\n"
    "Schema:\n{schema}\n\n"
    "Question: {question}\n\n" + SQL_CORRECTNESS_CRITERIA + "\n\nReturn your SQL query:"
)


def _predictor(model: str, base_delay: float, max_retries: int) -> Callable[[List[Dict]], Optional[str]]:
    client = openai.OpenAI()

    def predictor(messages: List[Dict]) -> Optional[str]:
        system_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        return _call_with_retry(
            client,
            max_retries=max_retries,
            base_delay=base_delay,
            model=model,
            messages=system_messages + messages,
            max_completion_tokens=512,
            temperature=0.0,
        )

    return predictor


def _select_examples(
    train_data: List[Dict],
    schemas: Dict[str, str],
    n_databases: Optional[int],
    n_per_database: Optional[int],
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed)
    db_counts: Counter = Counter(row["db_id"] for row in train_data)
    selected_dbs = [db for db, _ in db_counts.most_common(n_databases) if db in schemas]

    rows_by_db: Dict[str, List] = {db: [] for db in selected_dbs}
    for idx, row in enumerate(train_data):
        if row["db_id"] in rows_by_db:
            rows_by_db[row["db_id"]].append((idx, row))

    selected: List[Dict] = []
    for db in selected_dbs:
        candidates = rows_by_db[db]
        if n_per_database is None:
            sampled = candidates
        else:
            n_sampled = min(n_per_database, len(candidates))
            sampled = rng.sample(candidates, n_sampled)
        for original_idx, row in sampled:
            selected.append(
                {
                    "example_id": f"spider_train_{original_idx}",
                    "db_id": row["db_id"],
                    "question": row["question"],
                    "gold_sql": row["query"],
                }
            )

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SQL predictions on the Spider dataset using an LLM. "
            "Outputs a checkpointed JSONL file under data/ and can be safely interrupted and resumed. "
            "Requires OPENAI_API_KEY."
        )
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4-mini",
        help="OpenAI model name. (default: gpt-5.4-mini)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for database and example sampling. (default: 42)",
    )
    parser.add_argument(
        "--n-databases",
        type=int,
        default=10,
        help="Number of Spider databases to include, selected by descending example count. "
        "Includes all if unset. (default: 10)",
    )
    parser.add_argument(
        "--n-per-database",
        type=int,
        help="Maximum number of examples to sample per database. Includes all examples per database if unset.",
    )
    parser.add_argument(
        "--base-delay",
        type=float,
        default=2.0,
        help="Base delay in seconds for exponential backoff between retries. (default: 2.0)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of API call retries on failure. (default: 3)",
    )
    parser.add_argument(
        "--spider-path",
        type=Path,
        default=Path("data/spider"),
        help="Path to the Spider 1.0 directory (must contain tables.json and train_spider.json). "
        "(default: data/spider)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=("Output JSONL file for predictions; used as a checkpoint. (default: data/predictions_by_<model>.jsonl)"),
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.3,
        help="Seconds to sleep between API calls to avoid rate limits. (default: 0.0)",
    )
    parser.add_argument(
        "--include-prompt",
        action="store_true",
        default=False,
        help="Include the full user prompt (with SQL schema) in each output record. (default: false)",
    )
    args = parser.parse_args()

    if args.output is not None:
        output_path = args.output
    else:
        model_slug = args.model.replace("/", "-")
        output_path = Path(f"data/predictions_by_{model_slug}.jsonl")

    schemas = _load_schemas(args.spider_path / "tables.json")
    train_data = json.loads((args.spider_path / "train_spider.json").read_text())
    examples = _select_examples(train_data, schemas, args.n_databases, args.n_per_database, args.seed)

    processed: Set[str] = _load_checkpoint(output_path)
    remaining = [ex for ex in examples if ex["example_id"] not in processed]
    print(f"Selected {len(examples)} examples -- already processed: {len(processed)}, remaining: {len(remaining)}")

    predictor = _predictor(args.model, args.base_delay, args.max_retries)

    n_written = 0
    with open(output_path, "a") as out_f:
        for i, ex in enumerate(remaining):
            print(f"  [{i + 1}/{len(remaining)}] {ex['example_id']} ({ex['db_id']})")
            user_prompt = USER_TEMPLATE.format(
                db_id=ex["db_id"],
                schema=schemas[ex["db_id"]],
                question=ex["question"],
            )
            predicted_sql = predictor([{"role": "user", "content": user_prompt}])
            if predicted_sql is None:
                continue
            record = {
                "example_id": ex["example_id"],
                "db_id": ex["db_id"],
                "question": ex["question"],
                "gold_sql": ex["gold_sql"],
                "predicted_sql": _strip_markdown_fence(predicted_sql),
            }
            if args.include_prompt:
                record["prompt"] = user_prompt
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            n_written += 1
            time.sleep(args.sleep)
    print(f"Done. {n_written} records written to {output_path.resolve()}")


if __name__ == "__main__":
    main()
