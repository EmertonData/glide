"""
Generate candidate SQL predictions for Spider 1.0 examples using claude-haiku-4-5-20251001.

Run from the repo root:
    python spider_case_study/scripts/generate_predictions.py

Requires:
    ANTHROPIC_API_KEY environment variable
    uv sync --group tutorial
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import anthropic
from datasets import load_dataset

MODEL = "claude-haiku-4-5-20251001"
N_DATABASES = 50
EXAMPLES_PER_DATABASE = 20
RANDOM_SEED = 42
BASE_DELAY = 2.0
MAX_RETRIES = 3

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "predictions.jsonl"


def _call_with_retry(
    client: anthropic.Anthropic,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
    **kwargs,
) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            response = client.messages.create(**kwargs)
            return response.content[0].text
        except anthropic.RateLimitError:
            delay = base_delay * (2**attempt)
            print(f"  Rate limit hit, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                delay = base_delay * (2**attempt)
                print(f"  API error {exc.status_code}, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"  Non-retryable API error {exc.status_code}: {exc}")
                return None
    print(f"  Exhausted {max_retries} retries, skipping row.")
    return None


def _load_checkpoint(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    processed: Set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                processed.add(json.loads(line)["example_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return processed


def _select_examples(dataset) -> List[Dict]:
    import random
    from collections import Counter

    rng = random.Random(RANDOM_SEED)

    db_counts: Counter = Counter(row["db_id"] for row in dataset)
    top_databases: List[str] = [db for db, _ in db_counts.most_common(N_DATABASES)]

    rows_by_db: Dict[str, List] = {db: [] for db in top_databases}
    for idx, row in enumerate(dataset):
        db = row["db_id"]
        if db in rows_by_db:
            rows_by_db[db].append((idx, row))

    selected: List[Dict] = []
    for db in top_databases:
        candidates = rows_by_db[db]
        sampled = rng.sample(candidates, min(EXAMPLES_PER_DATABASE, len(candidates)))
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
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading Spider 1.0 dataset from HuggingFace...")
    dataset = load_dataset("xlangai/spider", split="train")
    print(f"  Train split: {len(dataset)} examples")

    examples = _select_examples(dataset)
    print(f"  Selected {len(examples)} examples from top {N_DATABASES} databases")

    processed = _load_checkpoint(OUTPUT_PATH)
    remaining = [ex for ex in examples if ex["example_id"] not in processed]
    print(f"  Already processed: {len(processed)} — remaining: {len(remaining)}")

    if not remaining:
        print("All examples already processed. Nothing to do.")
        return

    client = anthropic.Anthropic()

    with open(OUTPUT_PATH, "a") as out_f:
        for i, ex in enumerate(remaining):
            print(f"  [{i + 1}/{len(remaining)}] {ex['example_id']} ({ex['db_id']})")

            system_prompt = (
                "You are an expert SQL writer. Given a natural language question and a database name, "
                "write the SQL query that answers the question. Return ONLY valid SQL, no explanations."
            )
            user_prompt = f"Database: {ex['db_id']}\nQuestion: {ex['question']}\n\nReturn your SQL query:"

            predicted_sql = _call_with_retry(
                client,
                model=MODEL,
                max_tokens=256,
                temperature=0.0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            if predicted_sql is None:
                print(f"    Skipping {ex['example_id']} after exhausted retries.")
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

    total = sum(1 for _ in open(OUTPUT_PATH))
    print(f"\nDone. {OUTPUT_PATH} contains {total} records.")


if __name__ == "__main__":
    main()
