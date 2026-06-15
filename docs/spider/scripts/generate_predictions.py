import argparse
import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import anthropic
import openai
from _utils import _call_with_retry, _load_checkpoint, _load_schema
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You are an expert SQL writer. Given a natural language question and a database schema, "
    "write the SQL query that answers the question. Return ONLY the SQL query, no explanation."
)

USER_TEMPLATE = """\
Database: {db_id}

Schema:
{schema}

Question: {question}

Return your SQL query:"""


def anthropic_judge(model: str, base_delay: float, max_retries: int) -> Callable[[List[Dict]], Optional[str]]:
    client = anthropic.Anthropic()

    def judge(messages: List[Dict]) -> Optional[str]:
        return _call_with_retry(
            client,
            max_retries=max_retries,
            base_delay=base_delay,
            model=model,
            max_tokens=256,
            temperature=0.0,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

    return judge


def openai_judge(model: str, base_delay: float, max_retries: int) -> Callable[[List[Dict]], Optional[str]]:
    client = openai.OpenAI()

    def judge(messages: List[Dict]) -> Optional[str]:
        system_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=system_messages + messages,
                    max_tokens=256,
                    temperature=0.0,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2**attempt))
                else:
                    print(f"  Error after {max_retries} attempts: {e}")
        return None

    return judge


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
            "Requires ANTHROPIC_API_KEY or OPENAI_API_KEY depending on the provider."
        )
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="LLM provider to use. (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Model name passed to the provider API. (default: claude-haiku-4-5-20251001)",
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
        default=None,
        help="Number of Spider databases to include, selected by descending example count. Includes all if unset.",
    )
    parser.add_argument(
        "--n-per-database",
        type=int,
        default=None,
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
        help="Path to the Spider 1.0 directory (must contain tables.json and train_spider.json). (default: data/spider)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path to an existing JSONL file to resume from. Skips already-processed example IDs. "
            "Creates a new timestamped file if unset."
        ),
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Seconds to sleep between API calls to avoid rate limits. (default: 0.1)",
    )
    args = parser.parse_args()

    if args.output is not None:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"data/predictions_{args.provider}_{args.model}_{timestamp}.jsonl")

    schemas = _load_schema(args.spider_path / "tables.json")
    train_data = json.loads((args.spider_path / "train_spider.json").read_text())
    examples = _select_examples(train_data, schemas, args.n_databases, args.n_per_database, args.seed)

    processed: Set[str] = _load_checkpoint(output_path)
    remaining = [ex for ex in examples if ex["example_id"] not in processed]
    print(f"Selected {len(examples)} examples -- already processed: {len(processed)}, remaining: {len(remaining)}")

    if args.provider == "anthropic":
        judge = anthropic_judge(args.model, args.base_delay, args.max_retries)
    else:
        judge = openai_judge(args.model, args.base_delay, args.max_retries)

    with open(output_path, "a") as out_f:
        for i, ex in enumerate(remaining):
            print(f"  [{i + 1}/{len(remaining)}] {ex['example_id']} ({ex['db_id']})")
            user_prompt = USER_TEMPLATE.format(
                db_id=ex["db_id"],
                schema=schemas[ex["db_id"]],
                question=ex["question"],
            )
            predicted_sql = judge([{"role": "user", "content": user_prompt}])
            if predicted_sql is None:
                continue
            record = {
                "example_id": ex["example_id"],
                "db_id": ex["db_id"],
                "question": ex["question"],
                "gold_sql": ex["gold_sql"],
                "predicted_sql": predicted_sql.strip(),
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
