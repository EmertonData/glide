import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

import anthropic
from _utils import _call_with_retry, _load_checkpoint, _load_schema

MODEL = "claude-haiku-4-5-20251001"
RANDOM_SEED = 42
# Fill in after running explore_dataset.py:
N_DATABASES = ...
N_PER_DATABASE = ...
BASE_DELAY = 2.0
MAX_RETRIES = 3

SPIDER_PATH = Path("data/spider")
OUTPUT_PATH = Path("data/predictions.jsonl")

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


def _select_examples(train_data: List[Dict], schemas: Dict[str, str]) -> List[Dict]:
    rng = random.Random(RANDOM_SEED)
    db_counts: Counter = Counter(row["db_id"] for row in train_data)
    selected_dbs = [db for db, _ in db_counts.most_common(N_DATABASES) if db in schemas]

    rows_by_db: Dict[str, List] = {db: [] for db in selected_dbs}
    for idx, row in enumerate(train_data):
        if row["db_id"] in rows_by_db:
            rows_by_db[row["db_id"]].append((idx, row))

    selected: List[Dict] = []
    for db in selected_dbs:
        candidates = rows_by_db[db]
        sampled = rng.sample(candidates, min(N_PER_DATABASE, len(candidates)))
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
    schemas = _load_schema(SPIDER_PATH / "tables.json")
    train_data = json.loads((SPIDER_PATH / "train_spider.json").read_text())
    examples = _select_examples(train_data, schemas)

    processed: Set[str] = _load_checkpoint(OUTPUT_PATH)
    remaining = [ex for ex in examples if ex["example_id"] not in processed]
    print(f"Selected {len(examples)} examples -- already processed: {len(processed)}, remaining: {len(remaining)}")

    client = anthropic.Anthropic()
    with open(OUTPUT_PATH, "a") as out_f:
        for i, ex in enumerate(remaining):
            print(f"  [{i + 1}/{len(remaining)}] {ex['example_id']} ({ex['db_id']})")
            user_prompt = USER_TEMPLATE.format(
                db_id=ex["db_id"],
                schema=schemas[ex["db_id"]],
                question=ex["question"],
            )
            predicted_sql = _call_with_retry(
                client,
                max_retries=MAX_RETRIES,
                base_delay=BASE_DELAY,
                model=MODEL,
                max_tokens=256,
                temperature=0.0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
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
            time.sleep(0.3)


if __name__ == "__main__":
    main()
