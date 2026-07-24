---
name: newsletter
description: Drafts the monthly GLIDE LinkedIn newsletter (and, on request, its companion punchy LinkedIn post). Use this skill whenever the user asks to write, draft, or prepare the monthly newsletter, the LinkedIn issue, or the next GLIDE newsletter — even if phrased informally ("let's do the newsletter", "time for the monthly post", "draft this month's issue").
---

## Overview

This skill produces `newsletters/YYYY-MM.md` — the draft of the monthly GLIDE LinkedIn newsletter — and, if the user wants it, a second file `newsletters/YYYY-MM-linkedin-post.md`, a shorter and punchier companion post in a different register (see the "LinkedIn post companion" section below). The main newsletter is written in naked markdown: no `#` headers, just plain text titles with generous line breaks and ``` fences around code snippets. The user pastes it into LinkedIn's editor.

The newsletter has four sections plus a framing hook, with a target of at least 800 words:

- **Opening hook** — 2–3 sentences, no section title, connects the issue to a current theme or challenge in GenAI evaluation
- **Tutorial** — the centrepiece; a feature from the last two releases, told as a business problem first, with a short code snippet and paper citations
- **What's new in GLIDE** — changelog highlights rewritten for a broad audience, plus any events or milestones
- **What's next** — a living, high-level look at upcoming work
- **Get involved** — brief closing call to action (2–3 sentences)

At the bottom: a numbered reference list for any papers cited.

---

## Workflow

### Phase 1 — Auto-research (no user input needed)

Read the repository to build a picture of recent activity before asking the user anything.

1. Read `CHANGELOG.md`. Identify the last two released versions. Also read the `[Next release]` section: check `pyproject.toml`'s `current_version` and recent PR merge dates — if `[Next release]` looks substantial and recently active, a release may be imminent. Do not silently discard it: treat it as an in-scope candidate pool, flagged as "pending release", and let the user decide in Phase 2 whether to write the newsletter as if it has already shipped (this is common: the maintainer often ships the release right before publishing the newsletter).

2. Run `gh pr list --state merged --limit 30 --json number,title,mergedAt,body` to cross-reference with the changelog and get PR links.

3. Identify tutorial candidates: features from the last two released versions **and** from `[Next release]` if flagged as pending, that have corresponding documentation — a tutorial notebook in `docs/tutorials/`, a user guide page in `docs/user_guide/`, or a deep-dive notebook in `docs/deep_dive/`. List every candidate with its doc file path.

4. If no feature in the last two releases (or pending release) has usable documentation, stop here and tell the user: "I couldn't find a tutorial-ready feature in the last two releases. Could you point me to the topic you'd like to cover?" Do not proceed until the user responds.

### Phase 2 — Information gathering

Ask the user all of the following in a single message, not one question at a time:

1. **Tutorial topic.** Present every candidate you found (feature name + one-line description + doc path), including siblings. Ask which one to use, or whether to override with a different topic. If there is only one candidate, say so and ask for confirmation or an override.

2. **Release status.** If Phase 1 flagged a pending release, ask whether it will ship before the newsletter goes out and, if so, what version number to use. Treat that version's entries as already released for the rest of the workflow.

3. **Business case angle.** Ask if the user has a specific real-world scenario in mind for the tutorial, or whether to derive it from the documentation. Default to reusing the documentation's own framing as closely as possible (see Phase 3.3) rather than inventing a new angle.

4. **Events and milestones.** Ask if there are any external highlights to mention in the "What's new" section — accepted papers, talks, conference presentations, blog posts, awards — with links.

5. **Roadmap update.** Ask if the "What's next" section should be updated from the previous issue, or if there is a specific direction to emphasise this month.

6. **Companion post.** Ask whether the user also wants the punchy LinkedIn post companion (described below) drafted this month, alongside the main newsletter.

Do not write a word of the newsletter until the user has answered.

### Phase 3 — Deep research

With the tutorial topic confirmed:

1. Read the full documentation file(s) for the feature (tutorial notebook, user guide section, or deep-dive).

2. Extract:
   - The statistical or algorithmic problem being solved
   - The key public API calls (class names, method signatures, parameter names — note the exact parameter names, they must be reused verbatim in the newsletter prose, not paraphrased into synonyms)
   - Any paper citations already present in the docs, and which specific claim each one backs

3. Identify a real-world business scenario that motivates the problem. The tutorials in `docs/` often include a business framing — surface it and amplify it. If no framing exists at all, derive one from the problem structure (e.g. a sampler that minimises annotation cost → "your annotation budget is fixed and you need the most reliable estimate you can get").

4. Design the code snippet: 5–10 lines, no boilerplate, just the essential API call that demonstrates the value, with right imports. If the tutorial notebook has a representative example, adapt it rather than invent one. Reuse the exact parameter names extracted in step 2 in both the code and the sentence introducing it.

### Phase 4 — Write the newsletter

Follow the format and length guidance below precisely, then run through the quality checklist before saving.

### Phase 5 — Companion post (only if requested in Phase 2)

Draft `newsletters/YYYY-MM-linkedin-post.md` per the "LinkedIn post companion" section below, reusing the tutorial's problem/solution/result from the newsletter you just wrote rather than re-deriving them.

---

## Format and length

**File:** `newsletters/YYYY-MM.md` where YYYY-MM is the current year and month. Create the `newsletters/` directory if it does not exist.

**Encoding:** naked markdown. No `#` characters anywhere. Titles are plain text on their own line. Use the following line-break convention:

- 3 blank lines before and after top-level section titles
- 2 blank lines after subsection titles (if any)
- 1 blank line between paragraphs
- ``` fences around code snippets (the user removes them before pasting)

**Emojis on titles:** prefix every section title with a single leading emoji that fits the section (e.g. ✨ for "What's new in GLIDE", 🔭 for "What's next", 🙌 for "Get involved", 📚 for "References", and a topical one for the tutorial title). Exactly one emoji per title. Do NOT add emojis to the opening hook (it has no title), and do NOT scatter emojis inside body paragraphs or inline labels — titles only. A single emoji elsewhere is acceptable only when it lands naturally at the end of a sentence (e.g. a flag after a city name), never on the bold inline lead-ins of the tutorial body ("The problem.", "Why it is hard.", etc.).

**Tone:** clear, confident, accessible to a technical professional who is not a statistician. No jargon without a one-clause explanation — this includes mechanism-level vocabulary like "wealth process", "betting argument", "Central Limit Theorem", or named theorems: if a sentence needs one of these to make sense, cut the sentence and explain the *effect* instead (e.g. "it accumulates evidence across every batch observed so far" rather than naming the martingale construction that provides it). No exclamation marks. No filler phrases ("In today's world…", "It goes without saying…").

**Sentence length:** one idea per sentence. If a sentence has more than two commas or stacks more than one subordinate clause, split it. Read every paragraph out loud (mentally) before finalizing — if you would run out of breath, split it.

**Length:** at least 800 words. The tutorial section should be the longest, at roughly 400 words.

---

## Section-by-section guidance

### Opening hook

2–3 sentences. No title. Sets the scene: what challenge in GenAI evaluation does this issue speak to? Connects to something real and current without being clickbait. Should make a practitioner nod and keep reading.

Example register (not to be copied verbatim):
> Evaluating a language model is easy. Evaluating it reliably, at scale, without burning budget, is not. This month GLIDE adds a new piece to that puzzle.

---

### Tutorial section title

Format: a plain descriptive title that names the business problem, not the method. Good: "Cutting annotation costs without sacrificing confidence". Bad: "CostOptimalSampler tutorial".

**Structure of the tutorial body:**

1. **The problem** (2–3 sentences). Describe the business situation in concrete terms. A team, a constraint, a decision they need to make.

2. **Why it is hard** (1–2 sentences, each short — see sentence-length rule above). What breaks if you ignore it? What is the naive approach missing? If naming a statistical phenomenon (e.g. peeking, multiple testing), name it plainly and explain the consequence in the same breath, without stacking on the mechanism of why it happens.

3. **How GLIDE addresses it** (3–4 sentences). Explain the approach at the level of intuition, not mathematics. Name the relevant class/method naturally in prose. Do not name the underlying mathematical construction (wealth process, betting argument, CLT, etc.) — describe what it lets the user do instead.

4. **Code snippet.** Preceded by one sentence that says exactly what the snippet does, using the same parameter names that appear in the code.

```python
# 5–10 lines minimal comments
```

5. **What you get** (2–3 short sentences, one result per sentence). What does the output look like? What decision can the practitioner now make that they could not before? Refer to result fields and thresholds by their exact code names (e.g. `result.drift_detected`, `threshold`), not paraphrases like "the guarantee" or "the line" that don't map back to the snippet.

6. **Link to the full tutorial** in the GLIDE documentation. Keep this sentence purely about the link — do not attach citation numbers here.

7. **References**, inline, numbered in square brackets (e.g. [1]). Place each citation directly after the specific claim it supports (e.g. "...an anytime-valid confidence sequence [1]..." or "...the same principle behind prediction-powered risk monitoring [2]..."), never batched together at the end of an unrelated sentence like the documentation link.

---

### What's new in GLIDE

Title: "What's new in GLIDE"

Translate the changelog entries for the last two releases (using the confirmed version if a release was pending, see Phase 2.2) into plain language, focusing on what each change enables rather than how it was implemented. Group related items; do not list every bugfix individually unless one is particularly significant to users. If the tutorial covers one monitor/estimator in a family, this section is the place to mention its siblings shipped the same cycle, briefly and without repeating the tutorial's technical explanation.

---

### What's next

Title: "What's next"

A short, high-level paragraph (4–6 sentences) describing upcoming directions. Use the user's roadmap input. If no specific update was provided, derive from the `[Next release]` section of `CHANGELOG.md`. Do not make promises; frame as intentions and explorations.

---

### Get involved

Title: "Get involved"

2–3 sentences. Point to the GitHub repo (star it, open an issue, browse open issues). Mention the documentation. No bullet lists.

---

### Closing line

Before the References section, always include the following line as a standalone paragraph (no title, no blank lines above it other than the standard section spacing):

GLIDE is built by the R&D team at Emerton Data. This newsletter comes out monthly: one new capability, one tutorial, no filler.

---

### References

Title: "References"

Numbered list. One entry per cited paper, in Chicago author-date format — consistent with the rest of the GLIDE documentation:

[1] Last, First, First Last, and First Last. "Title." Venue/journal volume, no. number (year): pages. URL.

Example: Angelopoulos, Anastasios N., Stephen Bates, Clara Fannjiang, Michael I. Jordan, and Tijana Zrnic. "Prediction-powered inference." Science 382, no. 6671 (2023): 669-674. https://www.science.org/doi/10.1126/science.adi6000

Only include papers actually cited in the tutorial section, and check the relevant class's own docstring `References` section for the exact citation set to reuse — it is the canonical source, do not invent alternative references for the same method.

---

## Quality checklist before saving

- [ ] No `#` characters outside of code snippets (no markdown headers)
- [ ] All section titles are plain text on their own line with correct spacing
- [ ] Every section title (all but the opening hook) is prefixed with exactly one emoji; no emojis on body paragraphs or inline lead-ins
- [ ] Tutorial title names a business problem, not a method
- [ ] Business case is established before any technical content, and matches the documentation's own framing (no invented differentiator the user didn't ask for)
- [ ] No mechanism-level jargon (wealth process, betting argument, CLT, named theorems) anywhere in the tutorial body
- [ ] Every sentence is short: no more than two commas, no stacked subordinate clauses
- [ ] Code snippet is 5–10 lines, self-contained, no imports beyond the one shown
- [ ] Prose refers to code parameters and result fields by their exact names, not paraphrases
- [ ] Every inline citation sits next to the specific claim it supports, not batched at the end of an unrelated sentence
- [ ] Cited references match the exact set in the implementing class's docstring
- [ ] "What's new" is in plain language, no jargon
- [ ] At least 800 words total
- [ ] Closing "GLIDE is built by…" line present before References
- [ ] References section present if any paper was cited
- [ ] File saved to `newsletters/YYYY-MM.md`

---

## LinkedIn post companion

A second, optional deliverable: `newsletters/YYYY-MM-linkedin-post.md`. Only draft it if the user asked for it in Phase 2 (or asks for it afterwards). It is not a shorter version of the main newsletter — it is a different register entirely, built for skimmability and reaction rather than depth.

Before drafting, read the 2–3 most recent `newsletters/*-linkedin-post.md` files (if fewer exist, read all of them). Derive the structure, beat order, emoji choices, tone, and length from those examples rather than from a fixed template here — the format is free to evolve, and hardcoding it would make this skill drift out of sync with actual practice. Reuse this month's researched material (the tutorial's problem, fix, and result; the events from Phase 2) inside whatever structure those examples establish.

The one constant regardless of how the examples evolve: never invent facts. Every claim must trace back to the same researched material as the main newsletter.
