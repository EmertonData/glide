import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

INPUT_PATH = Path("data/predictions.jsonl")
OUTPUT_PATH = Path("data/human_labels.jsonl")
DB_ROOT = Path("data/spider/database")


def _execute_query(db_path: Path, sql: str) -> Tuple[Optional[List], Optional[str]]:
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        result = sorted(rows)
        return result, None
    except Exception as exc:
        return None, str(exc)


def _compute_label(db_path: Path, gold_sql: str, predicted_sql: str) -> Tuple[int, Optional[str]]:
    gold_rows, _ = _execute_query(db_path, gold_sql)
    predicted_rows, error = _execute_query(db_path, predicted_sql)

    if predicted_rows is None:
        return 0, error

    label = 1 if predicted_rows == gold_rows else 0
    return label, None


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


def main() -> None:
    examples = [json.loads(line) for line in open(INPUT_PATH) if line.strip()]
    processed: Set[str] = _load_checkpoint(OUTPUT_PATH)
    remaining = [ex for ex in examples if ex["example_id"] not in processed]
    print(f"Remaining: {len(remaining)} / {len(examples)}")

    with open(OUTPUT_PATH, "a") as out_f:
        for i, ex in enumerate(remaining):
            print(f"  [{i + 1}/{len(remaining)}] {ex['example_id']} ({ex['db_id']})")
            db_path = DB_ROOT / ex["db_id"] / f"{ex['db_id']}.sqlite"
            label, error = _compute_label(db_path, ex["gold_sql"], ex["predicted_sql"])

            record: Dict = {"example_id": ex["example_id"], "human_label": label}
            if error is not None:
                record["error"] = error

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()


if __name__ == "__main__":
    main()
