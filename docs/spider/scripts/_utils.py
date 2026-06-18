import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import anthropic
import openai

_ANTHROPIC_LABEL_OUTPUT_FORMAT = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "label": {"type": "integer", "enum": [0, 1]},
        },
        "required": ["reasoning", "label"],
        "additionalProperties": False,
    },
}

_OPENAI_LABEL_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "binary_label_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "label": {"type": "integer", "enum": [0, 1]},
            },
            "required": ["reasoning", "label"],
            "additionalProperties": False,
        },
    },
}


SQL_CORRECTNESS_CRITERIA = (
    "A query correctly answers the question if it:\n"
    "- Returns the right columns: selects what the question asks for, using all necessary tables joined"
    " without producing spurious duplicates or missing rows.\n"
    "- Filters correctly: encodes all constraints stated or implied by the question,"
    " without adding conditions the question does not imply.\n"
    "- Aggregates and groups correctly: uses the right function (COUNT, SUM, AVG, MIN, MAX)"
    " at the right granularity when the question calls for one.\n"
    "- Respects cardinality and ordering: applies LIMIT only when the question explicitly bounds the result;"
    " sorts by the right key and direction when specified.\n"
    "Column order in SELECT, aliases, SQL keyword casing, choice of JOIN syntax vs subquery,"
    " and other structural variations that produce an equivalent result do not affect correctness."
)


def anthropic_judge(
    model: str, base_delay: float, max_retries: int, system_prompt: str
) -> Callable[[List[Dict]], Optional[Tuple[int, Optional[str]]]]:
    client = anthropic.Anthropic()

    def judge(messages: List[Dict]) -> Optional[Tuple[int, Optional[str]]]:
        text = _call_with_retry_anthropic(
            client,
            max_retries=max_retries,
            base_delay=base_delay,
            model=model,
            max_tokens=512,
            temperature=0.0,
            system=system_prompt,
            output_config={"format": _ANTHROPIC_LABEL_OUTPUT_FORMAT},
            messages=messages,
        )
        if text is None:
            return None
        parsed = json.loads(text)
        return (int(parsed["label"]), parsed.get("reasoning"))

    return judge


def openai_judge(
    model: str, base_delay: float, max_retries: int, system_prompt: str
) -> Callable[[List[Dict]], Optional[Tuple[int, Optional[str]]]]:
    client = openai.OpenAI()

    def judge(messages: List[Dict]) -> Optional[Tuple[int, Optional[str]]]:
        system_messages = [{"role": "system", "content": system_prompt}]
        text = _call_with_retry_openai(
            client,
            max_retries=max_retries,
            base_delay=base_delay,
            model=model,
            messages=system_messages + messages,
            temperature=0.0,
            response_format=_OPENAI_LABEL_JSON_SCHEMA,
        )
        if text is None:
            return None
        parsed = json.loads(text)
        return (int(parsed["label"]), parsed.get("reasoning"))

    return judge


def _load_schemas(tables_path: Path) -> Dict[str, str]:
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


def _call_with_retry_anthropic(
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


def _call_with_retry_openai(
    client: openai.OpenAI,
    max_retries: int,
    base_delay: float,
    **kwargs,
) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            result = response.choices[0].message.content
            return result
        except openai.RateLimitError:
            delay = base_delay * (2**attempt)
            print(f"  Rate limit, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
        except openai.APIStatusError as exc:
            if exc.status_code >= 500:
                delay = base_delay * (2**attempt)
                time.sleep(delay)
            else:
                return None
    return None


def _strip_markdown_fence(sql: str) -> str:
    sql = sql.strip()
    sql = sql.strip("`")
    if sql.lower().startswith("sql"):
        sql = sql[3:]
    return sql.strip()


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
