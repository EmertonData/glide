import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import openai
import pandas as pd
from _utils import (
    build_example_id,
    encode_factuality_label,
    extract_factuality_label,
    load_memerag_dataset,
    load_prompt_templates,
    render_messages,
)
from download_dataset import DEFAULT_REPO_DIR
from huggingface_hub import HfApi

DEFAULT_REPO_ID = "Glide-py/memerag"
README_PATH = Path(__file__).resolve().parent.parent / "README.md"


def load_checkpoint(output_path: Path) -> Set[str]:
    if not output_path.exists():
        return set()
    processed: Set[str] = set()
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                processed.add(json.loads(line)["example_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return processed


def get_annotation(
    client: openai.OpenAI,
    model: str,
    messages: list,
    max_retries: int,
    base_delay: float,
    **sampling_kwargs,
) -> Optional[Tuple[Optional[str], str, int, int]]:
    """Call the judge model once, retrying on transient API errors or unparseable output.

    Returns
    -------
    Optional[Tuple[Optional[str], str, int, int]]
        `(predicted_label, raw_output, prompt_tokens, completion_tokens)`, or `None` if a
        non-retryable API error occurred, or every retry produced no valid label.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(model=model, messages=messages, **sampling_kwargs)
        except openai.RateLimitError:
            delay = base_delay * (2**attempt)
            print(f"    Rate limit, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
            continue
        except openai.APIStatusError as error:
            if error.status_code >= 500:
                delay = base_delay * (2**attempt)
                print(
                    f"    Server error {error.status_code}, retrying in {delay:.0f}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)
                continue
            print(f"    Non-retryable API error (HTTP {error.status_code}): {error.message}")
            return None

        raw_output = response.choices[0].message.content
        predicted_label = extract_factuality_label(raw_output)
        if predicted_label is not None:
            return predicted_label, raw_output, response.usage.prompt_tokens, response.usage.completion_tokens
        print(f"    Could not extract a label from the response, retrying (attempt {attempt + 1}/{max_retries})")
    return None


def annotate_dataset(
    dataset: pd.DataFrame,
    repo_dir: str,
    prompt_variant: str,
    model: str,
    output_path: Path,
    max_output_tokens: int = 500,
    max_retries: int = 3,
    base_delay: float = 2.0,
    sleep: float = 0.3,
) -> int:
    """Annotate every sentence in `dataset` not already present in `output_path`.

    Appends one JSON record per newly annotated sentence to `output_path`, flushing after every
    write, so the file is a valid checkpoint even if the run is interrupted midway.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Output of `load_memerag_dataset` (optionally row-limited for a pilot run).
    repo_dir : str
        Directory holding MEMERAG's data and prompts, downloaded beforehand via `download_dataset.py`.
    prompt_variant : str
        One of `"zero_shot"`, `"cot"`, `"ag"`, `"ag_cot"`.
    model : str
        OpenAI model name to call as the judge, e.g. `"gpt-5.4"`.
    output_path : Path
        Append-only JSONL checkpoint file.
    max_output_tokens : int
        Maximum number of completion tokens allowed per request.
    max_retries : int
        Maximum attempts per sentence, covering both transient API errors and unparseable output.
    base_delay : float
        Base delay in seconds for exponential backoff between retries.
    sleep : float
        Seconds to sleep between successive API calls.

    Returns
    -------
    int
        Number of records newly written in this run.
    """
    client = openai.OpenAI()
    system_template, task_template = load_prompt_templates(prompt_variant, repo_dir)

    sampling_kwargs: Dict[str, float] = {"temperature": 0.0, "max_completion_tokens": max_output_tokens}

    example_ids = dataset.apply(build_example_id, axis=1)
    processed = load_checkpoint(output_path)
    remaining = dataset[~example_ids.isin(processed)]
    print(f"Selected {len(dataset)} sentences -- already processed: {len(processed)}, remaining: {len(remaining)}")

    num_written = 0
    with open(output_path, "a") as out_f:
        for i, (_, row) in enumerate(remaining.iterrows()):
            example_id = build_example_id(row)
            print(f"  [{i + 1}/{len(remaining)}] {example_id}")
            messages = render_messages(row, system_template, task_template)
            result = get_annotation(client, model, messages, max_retries, base_delay, **sampling_kwargs)
            if result is None:
                print(f"    Giving up on {example_id} after {max_retries} attempts, will retry on next run")
                continue
            predicted_label, raw_output, prompt_tokens, completion_tokens = result
            print(f"    prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}")
            record = {
                "example_id": example_id,
                "query_id": row["query_id"],
                "language": row["language"],
                "query": row["query"],
                "context": row["context"],
                "sentence_id": int(row["sentence_id"]),
                "sentence": row["sentence"],
                "factuality": row["factuality"],
                "fine_grained_factuality": row["fine_grained_factuality"],
                "relevance": row["relevance"],
                "llm_judge_label": encode_factuality_label(str(predicted_label)),
                "raw_llm_output": raw_output,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            num_written += 1
            time.sleep(sleep)

    print(f"Done. {num_written} records newly written to {output_path.resolve()}")
    return num_written


def push_to_hub(output_path: Path, repo_id: str = DEFAULT_REPO_ID) -> str:
    """Create (if needed) a public Hugging Face dataset repo and upload the annotations and README.

    Parameters
    ----------
    output_path : Path
        JSONL file produced by `annotate_dataset` to upload.
    repo_id : str
        Target Hugging Face dataset repository, e.g. `"Glide-py/memerag"`.

    Returns
    -------
    str
        URL of the pushed repository.
    """
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(README_PATH),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update dataset card",
    )
    api.upload_file(
        path_or_fileobj=str(output_path),
        path_in_repo=output_path.name,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload {output_path.name}",
    )
    repo_url = f"https://huggingface.co/datasets/{repo_id}"
    return repo_url


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Query an OpenAI model with MEMERAG's own judge prompts and save factuality predictions."
    )
    parser.add_argument(
        "--repo-dir",
        default=str(DEFAULT_REPO_DIR),
        help="Directory holding MEMERAG's data and prompts, downloaded beforehand via download_dataset.py.",
    )
    parser.add_argument("--prompt-variant", default="ag_cot", choices=["zero_shot", "cot", "ag", "ag_cot"])
    parser.add_argument("--model", default="gpt-5.4", help="OpenAI model name to call as the judge.")
    parser.add_argument("--limit", type=int, default=None, help="Annotate only the first N sentences (pilot run).")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=500,
        help="Maximum number of completion tokens allowed per request. (default: 500)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file, used as a checkpoint. (default: memerag_llm_judge_<prompt_variant>_<model>.jsonl)",
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
        help="Maximum attempts per sentence, covering transient API errors and unparseable output. (default: 3)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.3,
        help="Seconds to sleep between API calls to avoid rate limits. (default: 0.3)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="After annotating, upload the output JSONL file and README.md to a Hugging Face dataset repo.",
    )
    parser.add_argument("--hf-repo-id", default=DEFAULT_REPO_ID, help="Target Hugging Face dataset repository.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.output is None:
        model_slug = args.model.replace("/", "-")
        args.output = Path(f"memerag_llm_judge_{args.prompt_variant}_{model_slug}.jsonl")

    dataset = load_memerag_dataset(args.repo_dir)
    if args.limit is not None:
        dataset = dataset.head(args.limit)

    print(f"Annotating {len(dataset)} sentences with model={args.model!r}, prompt_variant={args.prompt_variant!r}")
    annotate_dataset(
        dataset,
        args.repo_dir,
        args.prompt_variant,
        args.model,
        args.output,
        max_output_tokens=args.max_output_tokens,
        max_retries=args.max_retries,
        base_delay=args.base_delay,
        sleep=args.sleep,
    )

    if args.push_to_hub:
        repo_url = push_to_hub(args.output, args.hf_repo_id)
        print(f"Pushed {args.output} and README.md to {repo_url}")
