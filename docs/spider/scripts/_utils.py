import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import anthropic


def _load_schema(tables_path: Path) -> Dict[str, str]:
    with open(tables_path) as f:
        tables_data = json.load(f)

    schemas: Dict[str, str] = {}
    for db in tables_data:
        db_id: str = db["db_id"]
        table_names: List[str] = db["table_names_original"]
        column_names: List[List] = db["column_names_original"]
        column_types: List[str] = db["column_types"]
        primary_keys: List[int] = db["primary_keys"]
        foreign_keys: List[Tuple[int, int]] = db["foreign_keys"]

        table_columns: Dict[int, List[str]] = {i: [] for i in range(len(table_names))}
        for col_idx, (table_idx, col_name) in enumerate(column_names):
            if table_idx == -1:
                continue
            table_columns[table_idx].append(f"{col_name} {column_types[col_idx]}")

        lines: List[str] = []
        for table_idx, table_name in enumerate(table_names):
            cols = ", ".join(table_columns[table_idx])
            lines.append(f"Table {table_name}: ({cols})")

        pk_names = [
            f"{table_names[column_names[pk][0]]}.{column_names[pk][1]}"
            for pk in primary_keys
            if column_names[pk][0] != -1
        ]
        if pk_names:
            lines.append(f"Primary keys: {', '.join(pk_names)}")

        fk_pairs = [
            f"{table_names[column_names[src][0]]}.{column_names[src][1]}"
            f" -> {table_names[column_names[dst][0]]}.{column_names[dst][1]}"
            for src, dst in foreign_keys
        ]
        if fk_pairs:
            lines.append(f"Foreign keys: {', '.join(fk_pairs)}")

        schemas[db_id] = "\n".join(lines)

    return schemas


def _call_with_retry(
    client: anthropic.Anthropic,
    max_retries: int,
    base_delay: float,
    **kwargs,
) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            response = client.messages.create(**kwargs)
            result = response.content[0].text
            return result
        except anthropic.RateLimitError:
            delay = base_delay * (2**attempt)
            print(f"  Rate limit, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                delay = base_delay * (2**attempt)
                time.sleep(delay)
            else:
                return None
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
