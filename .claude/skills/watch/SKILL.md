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

In a **single message**, fire all 9 WebFetch calls simultaneously:

1. `https://arxiv.org/search/?searchtype=author&query=Angelopoulos,+Anastasios&start=0&order=-announced_date_first`
2. `https://arxiv.org/search/?searchtype=author&query=Cand%C3%A8s,+Emmanuel&start=0&order=-announced_date_first`
3. `https://arxiv.org/search/?searchtype=author&query=Zrnic,+Tijana&start=0&order=-announced_date_first`
4. `https://arxiv.org/search/?searchtype=author&query=Kluger,+Dan&start=0&order=-announced_date_first`
5. `https://arxiv.org/search/?searchtype=author&query=Bates,+Stephen&start=0&order=-announced_date_first`
6. `https://arxiv.org/search/?searchtype=author&query=Gligoric,+Kristina&start=0&order=-announced_date_first`
7. `https://arxiv.org/search/?searchtype=author&query=Lei,+Lihua&start=0&order=-announced_date_first`
8. `https://arxiv.org/search/?searchtype=author&query=Song,+Yilin&start=0&order=-announced_date_first`
9. `https://arxiv.org/search/?searchtype=author&query=Jordan,+Michael&start=0&order=-announced_date_first`

For each result extract: arXiv ID, title, authors, submission date.

## Step 4 — Filter by time window

From all papers gathered in Steps 1–3, keep only those whose submission date falls within the computed window. When a paper's date is ambiguous from search results alone, fetch `https://arxiv.org/abs/XXXX.XXXXX` to confirm.

Deduplicate across sources: if the same arXiv ID appears multiple times (e.g., once from the recent listing and once from a topic search), count it once.

Always exclude arXiv:2605.31278 (the GLIDE paper itself — Martinon, Merad, Raki) unless a distinct new version or a follow-up paper appears.

## Step 5 — Flag matching papers

Flag a paper if it meets **any** of the following criteria.

### Topic match

Check the title first; fetch the abstract page only when the title is ambiguous. Flag if the title or abstract mentions any of:

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

Flag if any of the following authors appear in the author list. Always verify the first name before flagging — last-name-only matches are not enough, especially for common names like Jordan or Song.

| Author | Notes |
|--------|-------|
| Anastasios Angelopoulos | |
| Emmanuel Candès | |
| Tijana Zrnic | |
| Dan Kluger | |
| Stephen Bates | |
| Kristina Gligoric | also spelled Gligorić |
| Lihua Lei | |
| Yilin Song | verify first name carefully |
| Michael Jordan | statistics/ML only — not the basketball player |

## Step 6 — Output

For each flagged paper:

```
[arXiv:XXXX.XXXXX] Title
Authors: ...
Reason flagged: <topic match: "exact phrase" | author match: "Author Name">
Link: https://arxiv.org/abs/XXXX.XXXXX
```

If a paper qualifies on both grounds, list both reasons on the same line separated by "; ".

Close with the summary line — always include it, even if nothing matched:

```
Flagged N paper(s) out of M total scanned, covering YYYY-MM-DD – YYYY-MM-DD.
```

## Coverage note

The stat/recent listing with `show=2000` is sufficient to cover the full 1–7 day window: the stat section receives fewer than 200 papers per day, so a week's worth sits comfortably under 2000 entries. The listing is therefore the primary source of coverage; the topic and author searches serve as a complementary signal to catch papers from adjacent categories (cs.LG, econ.EM, etc.) that are cross-listed or not primarily filed under stat. When M cannot be determined precisely (because WebFetch summarizes large pages), give a best-effort estimate and note this briefly.
