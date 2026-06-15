import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def _execute_query(db_path: Path, sql: str) -> Tuple[Optional[List], Optional[str]]:
    if not db_path.exists():
        return None, f"db_error: database file not found: {db_path}"

    try:
        conn = sqlite3.connect(str(db_path))
    except sqlite3.Error as exc:
        return None, f"db_error: {exc}"

    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        result = sorted(rows, key=str)
        return result, None
    except sqlite3.Error as exc:
        return None, f"sql_error: {exc}"
    finally:
        conn.close()


def _strip_markdown_fence(sql: str) -> str:
    sql = sql.strip()
    if sql.startswith("```"):
        lines = sql.splitlines()
        # drop opening fence line (```sql or ```)
        lines = lines[1:]
        # drop closing fence if present
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        sql = "\n".join(lines).strip()
    return sql


def _compute_label(db_path: Path, gold_sql: str, predicted_sql: str) -> Tuple[int, Optional[str]]:
    gold_rows, _ = _execute_query(db_path, gold_sql)
    predicted_rows, error = _execute_query(db_path, _strip_markdown_fence(predicted_sql))

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
    parser = argparse.ArgumentParser(description="Execute gold and predicted SQL queries and label each prediction.")
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to the input JSONL file containing predictions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/human_labels.jsonl"),
        help="Output JSONL file for human labels; also used as a checkpoint (default: data/human_labels.jsonl).",
    )
    parser.add_argument(
        "--db-root",
        type=Path,
        default=Path("data/spider"),
        help="Spider root or its 'database' subdirectory (default: data/spider).",
    )
    args = parser.parse_args()

    db_root = args.db_root / "database" if (args.db_root / "database").is_dir() else args.db_root

    examples = [json.loads(line) for line in open(args.input) if line.strip()]
    processed: Set[str] = _load_checkpoint(args.output)
    remaining = [ex for ex in examples if ex["example_id"] not in processed]
    print(f"Remaining: {len(remaining)} / {len(examples)}")

    n_written = 0
    with open(args.output, "a") as out_f:
        for i, ex in enumerate(remaining):
            print(f"  [{i + 1}/{len(remaining)}] {ex['example_id']} ({ex['db_id']})")
            db_path = db_root / ex["db_id"] / f"{ex['db_id']}.sqlite"
            label, error = _compute_label(db_path, ex["gold_sql"], ex["predicted_sql"])

            record: Dict = {"example_id": ex["example_id"], "human_label": label}
            if error is not None:
                record["error"] = error

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            n_written += 1
    print(f"Done. {n_written} records written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
