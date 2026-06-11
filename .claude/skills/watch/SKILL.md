---
name: watch
description: Monitor arXiv for new papers relevant to the GLIDE project (prediction-powered inference, active statistical inference, LLM evaluation debiasing, proxy annotation bias correction). Use whenever the user says "watch arXiv", "check for new PPI papers", "any new papers this week", "run the arXiv monitor", "what's new on arXiv", or invokes /watch with an optional time window like "3 days" or "1 week". Also triggers on "arXiv scan", "literature watch", "new papers on prediction-powered inference".
---

# arXiv GLIDE Watch

Monitor arXiv's statistics section for papers relevant to GLIDE's research areas over a user-specified time window.

## Time window

Parse the args:
- `/watch` with no args → default to **last announced batch** (arXiv announces papers 1–2 days after submission; using the calendar day would often yield no results)
- `/watch N days` → N calendar days back from today
- `/watch N weeks` → N × 7 days back from today
- `/watch N months` → N calendar months back from today (e.g., `2 months` → from today back to the same day two months ago)
- `/watch last month` → 1 calendar month back from today (alias for `/watch 1 month`)
- `/watch last [month name]` (e.g., `last february`) → the most recent full occurrence of that named month
- `/watch in [month name] [YYYY]` or `/watch [month name] [YYYY]` (e.g., `in february 2025`) → that specific calendar month

There is no upper cap. For windows exceeding 3 months, note in the output that coverage from the topic and author searches (Steps 2–3) may be incomplete due to result volume.

Announce at the start: "Scanning arXiv for papers from YYYY-MM-DD – YYYY-MM-DD."

## Date range → arXiv ID range

arXiv IDs use `YYMM.NNNNN` format — the YYMM prefix encodes the submission year and month. Using today's date from the system context, compute the earliest date in the window and determine which YYMM prefixes are relevant. If the window spans a month boundary (e.g., today is June 3 and N=5 → the window reaches back into May), include both month prefixes. Announce which prefixes are in scope.

## Step 1 — Fetch the stat listing(s)

**Short window (≤ 2 days / last announced batch):** fetch `https://arxiv.org/list/stat/recent?skip=0&show=2000` and extract all paper entries: arXiv ID, title, and authors.

**Longer window (> 2 days):** for each month in scope (from the Date range section above), fetch `https://arxiv.org/list/stat/YYYY-MM?skip=0&show=5000` — where `YYYY-MM` is the four-digit year and two-digit month (e.g., `2025-02` for February 2025). If the window spans multiple months, fire all fetches in a **single message**. Extract all paper entries: arXiv ID, title, authors, and submission date. Date filtering to the exact window happens in Step 5.

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

### 4e — Scholar fallback

If any Scholar fetch in 4b or 4c returns a CAPTCHA, a bot-detection page, or an empty body, skip that reference and proceed. If **all** Scholar fetches in 4b fail, skip Step 4 entirely and add a note to the Step 7 output:

```
(Scholar citation lookup skipped — bot detection or empty results.)
```

## Step 5 — Fetch abstracts and filter by time window

In a **single message**, fire one WebFetch call to `https://arxiv.org/abs/XXXX.XXXXX` for every arXiv candidate gathered in Steps 1–4 (skip non-arXiv papers). Retain the full abstract text — it is used for both date confirmation and topic matching in Step 6.

Keep only papers whose submission date falls within the computed window. Non-arXiv citing papers added in Step 4d bypass this check (they are kept as-is, with year-only date precision noted in the output).

Deduplicate across sources: if the same arXiv ID appears multiple times (e.g., once from the recent listing and once from a topic search), count it once.

Always exclude arXiv:2605.31278 (the GLIDE paper itself) unless a distinct new version or a follow-up paper appears.

## Step 6 — Flag matching papers

Flag a paper if it meets **any** of the following criteria.

### Topic match

Using the abstract text already fetched in Step 5, evaluate each surviving candidate for topic relevance. Flag if the title or abstract mentions any of:

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

For short windows (≤ 2 days), the `stat/recent` listing with `show=2000` is the primary source; the stat section receives fewer than 200 papers per day, so it comfortably covers 1–2 announcement days. For longer windows, the monthly `stat/YYMM` listings are used instead — `show=5000` covers a full month of stat submissions. In both cases, the topic and author searches (Steps 2–3) serve as a complementary signal to catch papers from adjacent categories (cs.LG, econ.EM, etc.) that are cross-listed or not primarily filed under stat. The Scholar citation lookup (Step 4) is a separate signal for papers in any field that build on the README references. When M cannot be determined precisely (because WebFetch summarizes large pages), give a best-effort estimate and note this briefly.
