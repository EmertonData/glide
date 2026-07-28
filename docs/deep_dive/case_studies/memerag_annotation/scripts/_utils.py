import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from download_dataset import DEFAULT_REPO_DIR, download_memerag
from jinja2 import Environment, FileSystemLoader, Template

LANGUAGES = ["en", "de", "es", "fr", "hi"]
PROMPT_VARIANTS = ["zero_shot", "cot", "ag", "ag_cot"]
FACTUALITY_TO_BINARY = {"Supported": 1, "Not Supported": 0}
AMBIGUOUS_FACTUALITY_LABEL = "Challenging to determine"


def load_memerag_dataset(repo_dir: str = str(DEFAULT_REPO_DIR)) -> pd.DataFrame:
    """Load MEMERAG's per-sentence human annotations into a flat table.

    Downloads MEMERAG into `repo_dir` first if not already present there. Answer sentences whose
    human `factuality` label is `"Challenging to determine"` (neither `"Supported"` nor
    `"Not Supported"`) are dropped, since they carry no clean binary ground truth.

    Parameters
    ----------
    repo_dir : str
        Directory holding (or to hold) MEMERAG's data and prompts; see `download_dataset.py`.

    Returns
    -------
    pandas.DataFrame
        One row per answer sentence, with columns `query_id`, `language`, `query`, `context`
        (list of `{"text": ...}` passage dicts, as released), `sentence_id`, `sentence`,
        `factuality` (human ground-truth label, encoded as `1` for `"Supported"` and `0` for
        `"Not Supported"`), `fine_grained_factuality`, `relevance`.
    """
    download_memerag(Path(repo_dir))
    data_dir = Path(repo_dir) / "data" / "memerag"
    rows = []
    for language in LANGUAGES:
        with open(data_dir / f"{language}.jsonl", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                for answer_sentence in record["answer"]:
                    if answer_sentence["factuality"] == AMBIGUOUS_FACTUALITY_LABEL:
                        continue
                    rows.append(
                        {
                            "query_id": record["query_id"],
                            "language": language,
                            "query": record["query"],
                            "context": record["context"],
                            "sentence_id": answer_sentence["sentence_id"],
                            "sentence": answer_sentence["sentence"],
                            "factuality": encode_factuality_label(answer_sentence["factuality"]),
                            "fine_grained_factuality": answer_sentence["fine_grained_factuality"],
                            "relevance": answer_sentence["relevance"],
                        }
                    )
    dataset = pd.DataFrame(rows)
    return dataset


def build_example_id(row: pd.Series) -> str:
    """Build the unique identifier `<language>_<query_id>_<sentence_id>` for one answer sentence.

    Parameters
    ----------
    row : pandas.Series
        A row from `load_memerag_dataset`'s output (needs `language`, `query_id`, `sentence_id`).

    Returns
    -------
    str
        The unique example identifier.
    """
    example_id = f"{row['language']}_{row['query_id']}_{row['sentence_id']}"
    return example_id


def load_prompt_templates(prompt_variant: str, repo_dir: str = str(DEFAULT_REPO_DIR)) -> Tuple[Template, Template]:
    """Load MEMERAG's Jinja2 system/task judge prompt templates for one prompting strategy.

    Downloads MEMERAG into `repo_dir` first if not already present there.

    Parameters
    ----------
    prompt_variant : str
        One of `"zero_shot"`, `"cot"`, `"ag"`, `"ag_cot"`, matching the repository's
        `src/prompts/` subdirectories. `"ag_cot"` (annotation guidelines + chain-of-thought) is
        MEMERAG's best-performing strategy.
    repo_dir : str
        Directory holding (or to hold) MEMERAG's data and prompts; see `download_dataset.py`.

    Returns
    -------
    Tuple[Template, Template]
        The system prompt template and the task prompt template.
    """
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"'prompt_variant' must be one of {PROMPT_VARIANTS}; got {prompt_variant!r}.")
    download_memerag(Path(repo_dir))
    prompt_dir = Path(repo_dir) / "src" / "prompts" / prompt_variant
    environment = Environment(loader=FileSystemLoader(prompt_dir), autoescape=True)
    system_template = environment.get_template("sys_prompt.md")
    task_template = environment.get_template("task_prompt.md")
    return system_template, task_template


def render_messages(row: pd.Series, system_template: Template, task_template: Template) -> List[Dict[str, str]]:
    """Render one answer-sentence row into OpenAI chat messages.

    Mirrors MEMERAG's own `build_message` for the `open_ai` provider.

    Parameters
    ----------
    row : pandas.Series
        A row from `load_memerag_dataset`'s output (needs `query`, `context`, `sentence`).
    system_template : Template
        Rendered via `load_prompt_templates`.
    task_template : Template
        Rendered via `load_prompt_templates`.

    Returns
    -------
    List[Dict[str, str]]
        `[{"role": "system", "content": ...}, {"role": "user", "content": ...}]`.
    """
    template_dict = {"query": row["query"], "context": row["context"], "answer_segment": row["sentence"]}
    system_prompt = system_template.render(**template_dict)
    task_prompt = task_template.render(**template_dict)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt},
    ]
    return messages


def encode_factuality_label(label: str) -> int:
    """Encode a `"Supported"` / `"Not Supported"` factuality label as binary.

    Parameters
    ----------
    label : str
        `"Supported"` or `"Not Supported"`.

    Returns
    -------
    int
        `1` for `"Supported"`, `0` for `"Not Supported"`.
    """
    if label not in FACTUALITY_TO_BINARY:
        raise ValueError(f"'label' must be one of {list(FACTUALITY_TO_BINARY)}; got {label!r}.")
    encoded_label = FACTUALITY_TO_BINARY[label]
    return encoded_label


def extract_factuality_label(llm_output: str) -> Optional[str]:
    """Extract the "Supported" / "Not Supported" label from a raw LLM completion.

    Mirrors MEMERAG's own `extract_answer_factuality`: checks `<answer>` tags first. When an
    `<answer>` tag is found but its content isn't a clean label, checks `<rationale>` tags next.
    When no `<answer>` tag is found, falls back directly to fuzzy substring matching over the
    raw output.

    Parameters
    ----------
    llm_output : str
        The raw text completion returned by the judge model.

    Returns
    -------
    Optional[str]
        `"Supported"`, `"Not Supported"`, or `None` if no label could be extracted.
    """
    patterns = [r"<answer>([\s\S]*?)</answer>", r"<rationale>([\s\S]*?)</rationale>"]
    for pattern in patterns:
        matches = re.findall(pattern, llm_output)
        if matches:
            for match in matches:
                normalised_match = match.lower().strip()
                if normalised_match in ("supported", "not supported"):
                    label = "Not Supported" if normalised_match.startswith("not") else "Supported"
                    return label
        else:
            # fall back to fuzzy substring matching
            lowered_output = llm_output.lower()
            if "not supported" in lowered_output:
                return "Not Supported"
            elif "supported" in lowered_output:
                return "Supported"
            return None
    return None
