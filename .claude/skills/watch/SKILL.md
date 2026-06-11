---
name: watch
description: Monitor arXiv for new papers relevant to the GLIDE project (prediction-powered inference, active statistical inference, LLM evaluation debiasing, proxy annotation bias correction). Use whenever the user says "watch arXiv", "check for new PPI papers", "any new papers this week", "run the arXiv monitor", "what's new on arXiv", or invokes /watch with an optional time window like "3 days" or "1 week". Also triggers on "arXiv scan", "literature watch", "new papers on prediction-powered inference".
---

# arXiv GLIDE Watch

Monitor arXiv's statistics section for papers relevant to GLIDE's research areas over a user-specified time window.

## Time window

Parse the args:
- `/watch` with no args → default to **Today**
- `/watch N days` → N days (clamp to 1–7)
- `/watch 1 week` → 7 days

Announce at the start: "Scanning arXiv for papers from the last N days (YYYY-MM-DD – YYYY-MM-DD)."

## Date range → arXiv ID range

arXiv IDs use `YYMM.NNNNN` format — the YYMM prefix encodes the submission year and month. Using today's date from the system context, compute the earliest date in the window and determine which YYMM prefixes are relevant. If the window spans a month boundary (e.g., today is June 3 and N=5 → the window reaches back into May), include both month prefixes. Announce which prefixes are in scope.

## Step 1 — Fetch the stat/recent listing

Fetch `https://arxiv.org/list/stat/recent?skip=0&show=2000` and extract all paper entries: arXiv ID, title, and authors. This listing covers the last 1–2 announcement days and is the ground truth for the most recent papers.

## Step 2 — Parallel topic searches

In a **single message**, fire all 5 WebFetch calls simultaneously:

1. `https://arxiv.org/search/?query=prediction-powered+inference&searchtype=all&start=0&order=-announced_date_first`
2. `https://arxiv.org/search/?query=active+statistical+inference&searchtype=all&start=0&order=-announced_date_first`
3. `https://arxiv.org/search/?query=semi-supervised+inference+debiasing&searchtype=all&start=0&order=-announced_date_first`
4. `https://arxiv.org/search/?query=LLM+annotation+bias+correction+inference&searchtype=all&start=0&order=-announced_date_first`
5. `https://arxiv.org/search/?query=proxy+labels+imputation+inference&searchtype=all&start=0&order=-announced_date_first`

For each result extract: arXiv ID, title, authors, submission date.

## Step 3 — Parallel author searches

In a **single message**, fire all 10 WebFetch calls simultaneously:

1. `https://arxiv.org/search/?searchtype=author&query=Angelopoulos,+Anastasios&start=0&order=-announced_date_first`
2. `https://arxiv.org/search/?searchtype=author&query=Cand%C3%A8s,+Emmanuel&start=0&order=-announced_date_first`
3. `https://arxiv.org/search/?searchtype=author&query=Zrnic,+Tijana&start=0&order=-announced_date_first`
4. `https://arxiv.org/search/?searchtype=author&query=Kluger,+Dan&start=0&order=-announced_date_first`
5. `https://arxiv.org/search/?searchtype=author&query=Bates,+Stephen&start=0&order=-announced_date_first`
6. `https://arxiv.org/search/?searchtype=author&query=Gligoric,+Kristina&start=0&order=-announced_date_first`
7. `https://arxiv.org/search/?searchtype=author&query=Lei,+Lihua&start=0&order=-announced_date_first`
8. `https://arxiv.org/search/?searchtype=author&query=Song,+Yilin&start=0&order=-announced_date_first`
9. `https://arxiv.org/search/?searchtype=author&query=Jordan,+Michael&start=0&order=-announced_date_first`
10. `https://arxiv.org/search/?searchtype=author&query=Romano,+Yaniv&start=0&order=-announced_date_first`

For each result extract: arXiv ID, title, authors, submission date.

## Step 4 — Scholar citation lookup

**Goal**: find papers published within the time window that cite any of the reference papers listed in `README.md`.

### 4a — Read README references

Read `README.md` and extract every entry in the `### References` section. For each entry, record its short label (e.g. `[1]`) and full title + first author (e.g. "Prediction-powered inference — Angelopoulos et al.").

### 4b — Resolve Scholar cluster IDs in parallel

In a **single message**, fire one WebFetch call per reference paper to:

`https://scholar.google.com/scholar?q=<url-encoded title and first author>&hl=en`

From each result, extract the href of the "Cited by N" link for the matching paper. That href contains the Scholar cluster ID in the form `cites=XXXXXXXXXXXXXXXXX`. Record the cluster ID for each paper (skip papers where no match is found).

### 4c — Fetch citing papers in parallel

In a **single message**, fire one WebFetch call per resolved cluster ID to:

`https://scholar.google.com/scholar?cites=<ID>&as_sdt=2005&sciodt=0,5&hl=en&scisbd=1`

(`scisbd=1` sorts by date, most recent first.)

From each result, extract all visible entries: title, authors, publication year, and any arXiv or DOI link. Also note which README reference `[N]` each result came from.

### 4d — Add to candidate pool

For each citing paper found in 4c:
- If it carries an arXiv ID, add it to the candidate pool with the label `cites:[N]` so that Step 5 can verify its exact date.
- If it has no arXiv ID, check whether its publication year falls within the time window's year range. If yes, add it to the pool directly as a **non-arXiv citing paper** and skip the Step 5 date check (Scholar does not expose the exact submission date for non-arXiv papers).

Deduplicate: if the same arXiv ID or title was already gathered in Steps 1–3, do not add a duplicate — just annotate the existing entry with `cites:[N]`.

## Step 5 — Filter by time window

From all papers gathered in Steps 1–4, keep only those whose submission date falls within the computed window. When a paper's date is ambiguous from search results alone, fetch `https://arxiv.org/abs/XXXX.XXXXX` to confirm. Non-arXiv citing papers added in Step 4d bypass this check (they are kept as-is, with year-only date precision noted in the output).

Deduplicate across sources: if the same arXiv ID appears multiple times (e.g., once from the recent listing and once from a topic search), count it once.

Always exclude arXiv:2605.31278 (the GLIDE paper itself) unless a distinct new version or a follow-up paper appears.

## Step 6 — Flag matching papers

Flag a paper if it meets **any** of the following criteria.

### Topic match

For every candidate paper that survived Step 5, spawn one sub-agent per paper (using the Agent tool) to fetch `https://arxiv.org/abs/XXXX.XXXXX` and evaluate topic relevance. Fire all sub-agents in a **single message** so the fetches run in parallel. Each sub-agent returns only: `{ topic_match: bool, matched_phrase: str | null }` — the caller already holds the arXiv ID and title. Flag if the title or abstract mentions any of:

- Prediction-powered inference (PPI, PPI++)
- Active statistical inference / active inference (statistical sense — not reinforcement learning)
- Predict-then-debias (PTD)
- Semi-supervised inference or semi-supervised estimation
- Debiasing proxy labels / proxy annotations / LLM annotations
- Stratified estimation combined with debiasing or imputation
- IPW (inverse probability weighting) combined with imputed outcomes
- Confidence-driven inference
- Hybrid annotation / human-in-the-loop evaluation of language models
- LLM-as-judge bias correction / LLM evaluation under annotation noise
- Imputed covariates in inference / covariate imputation

When flagging for topic, quote the specific matching phrase from the title or abstract.

### Author match

Flag if any of the Step 3 authors appear in the author list. Always verify the full first name before flagging — last-name-only matches are not enough, especially for common names like Jordan or Song. Note: Kristina Gligoric is also spelled Gligorić; Michael Jordan should be statistics/ML only.

### Scholar citation match

Flag if the paper was found in Step 4 as citing a README reference paper. Note which reference it cites.

## Step 7 — Output

For each flagged paper:

```
[arXiv:XXXX.XXXXX] Title            (use [non-arXiv] if no arXiv ID)
Authors: ...
Reason flagged: <topic match: "exact phrase" | author match: "Author Name" | cites README [N]: "Short paper title">
Link: https://arxiv.org/abs/XXXX.XXXXX  (or DOI/URL for non-arXiv papers)
```

If a paper qualifies on multiple grounds, list all reasons on the same line separated by "; ".

For non-arXiv citing papers, append `(date: YYYY — exact day unavailable)` after the link.

Close with the summary line — always include it, even if nothing matched:

```
Flagged N paper(s) out of M total scanned, covering YYYY-MM-DD – YYYY-MM-DD.
(of which K found via Scholar citation lookup)
```

## Coverage note

The stat/recent listing with `show=2000` is sufficient to cover the full 1–7 day window: the stat section receives fewer than 200 papers per day, so a week's worth sits comfortably under 2000 entries. The listing is therefore the primary source of coverage; the topic and author searches serve as a complementary signal to catch papers from adjacent categories (cs.LG, econ.EM, etc.) that are cross-listed or not primarily filed under stat. The Scholar citation lookup (Step 4) is a separate signal that catches papers in any field that build on the README references but would not surface through arXiv keyword or author searches. When M cannot be determined precisely (because WebFetch summarizes large pages), give a best-effort estimate and note this briefly.
