---
license: cc-by-nc-4.0
task_categories:
  - text-classification
language:
  - en
  - de
  - es
  - fr
  - hi
tags:
  - rag
  - faithfulness
  - llm-evaluation
  - glide
---

# MEMERAG Faithfulness LLM-Judge Labels

This dataset extends [MEMERAG](https://github.com/amazon-science/MEMERAG) (Cruz Blandón et al., 2025, ACL 2025,
[arXiv:2502.17163](https://arxiv.org/abs/2502.17163)) with LLM-judge faithfulness predictions from an OpenAI model,
reusing MEMERAG's own data and judge prompt templates.

## Files

| File | Description |
|------|-------------|
| `memerag_llm_judge_<prompt_variant>_<model>.jsonl` | LLM-judge predictions (produced by `scripts/annotate_memerag.py`) |
| `scripts/` | Reproduction scripts (see below) |

## Dataset statistics

- **Source**: MEMERAG per-sentence human annotations, built on MIRACL
- **Languages**: English, German, Spanish, French, Hindi
- **Total records**: 2,322 answer sentences before dropping ambiguous labels (see below)
- **Labels**: binary (0 = Not Supported, 1 = Supported)

## Schema

| Column | Type | Description |
|--------|------|--------------|
| `example_id` | string | Unique identifier (`<language>_<query_id>_<sentence_id>`) |
| `query_id` | int | MEMERAG/MIRACL query identifier |
| `language` | string | One of `en`, `de`, `es`, `fr`, `hi` |
| `query` | string | The question |
| `context` | list | Retrieved passages, as `{"text": ...}` dicts, as released |
| `sentence_id` | int | Index of the sentence within its answer |
| `sentence` | string | The answer sentence being judged |
| `factuality` | int | Human ground-truth label: `1` for `"Supported"`, `0` for `"Not Supported"` |
| `fine_grained_factuality` | string | Finer-grained human category for the `factuality` verdict (e.g. `"Contradiction"`, `"Direct paraphrase"`) |
| `relevance` | string | How the sentence relates to the question (e.g. `"Directly answers the question"`) |
| `llm_judge_label` | int | Judge's predicted label, encoded the same way as `factuality` |
| `raw_llm_output` | string | Full raw completion returned by the judge model |

Answer sentences whose human label is `"Challenging to determine"` (neither `"Supported"` nor `"Not Supported"`) are
dropped before annotation, since they carry no clean binary ground truth.

## Reproduction

Install dependencies:

```bash
pip install -r scripts/requirements.txt
```

Then, from the repository root, generate the labels. The script downloads MEMERAG's per-sentence data and judge
prompt templates automatically on first use; `scripts/download_dataset.py` can also be run standalone
(`python scripts/download_dataset.py --output-dir data/MEMERAG`) to pre-download without doing anything else.

```bash
export OPENAI_API_KEY=...
python scripts/annotate_memerag.py --prompt-variant ag_cot --model gpt-5.4-mini \
    --output memerag_llm_judge.jsonl
```

Requires an OpenAI API key set as `OPENAI_API_KEY`. Checkpoints to `--output`: each record is keyed by `example_id`
and flushed to disk immediately after every API call, so the run can be resumed if accidentally interrupted. This
skips entries already present in the output file. Transient API errors (rate limits, 5xx) are retried with
exponential backoff (`--base-delay`, `--max-retries`); non-retryable errors and unparseable completions are skipped.

`ag_cot` (annotation guidelines + chain-of-thought) is MEMERAG's best-performing prompting strategy and is the
default, but all 4 are supported via `--prompt-variant` (`zero_shot`, `cot`, `ag`, `ag_cot`). `cot` and `ag_cot` ask
the model to write a free-text `<rationale>` block before its label, so their completions run longer than
`zero_shot` / `ag`, which only ask for the label.

Pass `--limit N` to annotate only the first N sentences, e.g. as a pilot run to sanity-check the actual completion
length before committing to a full run:

```bash
python scripts/annotate_memerag.py --prompt-variant ag_cot --model gpt-5.4-mini \
    --limit 20 --output pilot.jsonl
```

Pass `--push-to-hub` to additionally upload the resulting JSONL file and this `README.md` (as the dataset card) to a
Hugging Face dataset repository, named by `--hf-repo-id`, creating it as a public repo if it does not already exist.
Requires a Hugging Face access token with write access, set as `HF_TOKEN` or via a prior `huggingface-cli login`:

```bash
export HF_TOKEN=...
python scripts/annotate_memerag.py --prompt-variant ag_cot --model gpt-5.4-mini \
    --output memerag_llm_judge.jsonl --push-to-hub
```

## Source

```bibtex
@inproceedings{blandon2025memerag,
  title={Memerag: A multilingual end-to-end meta-evaluation benchmark for retrieval augmented generation},
  author={Bland{\'o}n, Mar{\'\i}a Andrea Cruz and Talur, Jayasimha and Charron, Bruno and Liu, Dong and Mansour, Saab and Federico, Marcello},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={22577--22595},
  year={2025}
}
```

`MEMERAG` is licensed CC BY-NC 4.0 (non-commercial); the underlying MIRACL passages/queries are Apache-2.0. Any
redistribution of a curated table derived from it must respect those terms.
