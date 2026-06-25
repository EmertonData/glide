---
name: newsletter
description: Drafts the monthly GLIDE LinkedIn newsletter. Use this skill whenever the user asks to write, draft, or prepare the monthly newsletter, the LinkedIn issue, or the next GLIDE newsletter — even if phrased informally ("let's do the newsletter", "time for the monthly post", "draft this month's issue").
---

## Overview

This skill produces a single file `newsletters/YYYY-MM.md` — the draft of the monthly GLIDE LinkedIn newsletter. It is written in naked markdown: no `#` headers, just plain text titles with generous line breaks and ``` fences around code snippets. The user pastes it into LinkedIn's editor.

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

1. Read `CHANGELOG.md`. Identify the last two released versions (ignore `[Next release]`). Extract the user-facing additions and changes from those two versions.

2. Run `gh pr list --state merged --limit 30 --json number,title,mergedAt,body` to cross-reference with the changelog and get PR links.

3. Identify tutorial candidates: features from the last two releases that have corresponding documentation — a tutorial notebook in `docs/tutorials/`, a user guide page in `docs/user_guide/`, or a deep-dive notebook in `docs/deep_dive/`. List the candidates with their doc file paths.

4. If no feature in the last two releases has usable documentation, stop here and tell the user: "I couldn't find a tutorial-ready feature in the last two releases. Could you point me to the topic you'd like to cover?" Do not proceed until the user responds.

### Phase 2 — Information gathering

Ask the user all of the following in a single message, not one question at a time:

1. **Tutorial topic.** Present the candidates you found (feature name + one-line description + doc path). Ask which one to use, or whether to override with a different topic. If there is only one candidate, say so and ask for confirmation or an override.

2. **Business case angle.** Ask if the user has a specific real-world scenario in mind for the tutorial, or whether to derive it from the documentation.

3. **Events and milestones.** Ask if there are any external highlights to mention in the "What's new" section — accepted papers, talks, conference presentations, blog posts, awards — with links.

4. **Roadmap update.** Ask if the "What's next" section should be updated from the previous issue, or if there is a specific direction to emphasise this month.

Do not write a word of the newsletter until the user has answered.

### Phase 3 — Deep research

With the tutorial topic confirmed:

1. Read the full documentation file(s) for the feature (tutorial notebook, user guide section, or deep-dive).

2. Extract:
   - The statistical or algorithmic problem being solved
   - The key public API calls (class names, method signatures, parameter names)
   - Any paper citations already present in the docs

3. Identify a real-world business scenario that motivates the problem. The tutorials in `docs/` often include a business framing — surface it and amplify it. If no framing exists, derive one from the problem structure (e.g. a sampler that minimises annotation cost → "your annotation budget is fixed and you need the most reliable estimate you can get").

4. Design the code snippet: 5–10 lines, no boilerplate, just the essential API call that demonstrates the value, with right imports. If the tutorial notebook has a representative example, adapt it rather than invent one.

### Phase 4 — Write the newsletter

Follow the format and length guidance below precisely.

---

## Format and length

**File:** `newsletters/YYYY-MM.md` where YYYY-MM is the current year and month. Create the `newsletters/` directory if it does not exist.

**Encoding:** naked markdown. No `#` characters anywhere. Titles are plain text on their own line. Use the following line-break convention:

- 3 blank lines before and after top-level section titles
- 2 blank lines after subsection titles (if any)
- 1 blank line between paragraphs
- ``` fences around code snippets (the user removes them before pasting)

**Emojis on titles:** prefix every section title with a single leading emoji that fits the section (e.g. ✨ for "What's new in GLIDE", 🔭 for "What's next", 🙌 for "Get involved", 📚 for "References", and a topical one for the tutorial title). Exactly one emoji per title. Do NOT add emojis to the opening hook (it has no title), and do NOT scatter emojis inside body paragraphs or inline labels — titles only. A single emoji elsewhere is acceptable only when it lands naturally at the end of a sentence (e.g. a flag after a city name), never on the bold inline lead-ins of the tutorial body ("The problem.", "Why it is hard.", etc.).

**Tone:** clear, confident, accessible to a technical professional who is not a statistician. No jargon without a one-clause explanation. No exclamation marks. No filler phrases ("In today's world…", "It goes without saying…").

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

2. **Why it is hard** (1–2 sentences). What breaks if you ignore it? What is the naive approach missing?

3. **How GLIDE addresses it** (3–4 sentences). Explain the approach at the level of intuition, not mathematics. Name the relevant class/method naturally in prose.

4. **Code snippet.** Preceded by one sentence that says exactly what the snippet does.

```python
# 5–10 lines minimal comments
```

5. **What you get** (2–3 sentences). What does the output look like? What decision can the practitioner now make that they could not before?

6. **Link to the full tutorial** in the GLIDE documentation.

7. **References** inline, numbered in square brackets (e.g. [1]), pointing to the reference list at the bottom of the newsletter.

---

### What's new in GLIDE

Title: "What's new in GLIDE"

Translate the changelog entries for the last two releases into plain language, focusing on what each change enables rather than how it was implemented. Group related items; do not list every bugfix individually unless one is particularly significant to users.

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

Only include papers actually cited in the tutorial section.

---

## Quality checklist before saving

- [ ] No `#` characters outside of code snippets (no markdown headers)
- [ ] All section titles are plain text on their own line with correct spacing
- [ ] Every section title (all but the opening hook) is prefixed with exactly one emoji; no emojis on body paragraphs or inline lead-ins
- [ ] Tutorial title names a business problem, not a method
- [ ] Business case is established before any technical content
- [ ] Code snippet is 5–10 lines, self-contained, no imports
- [ ] "What's new" is in plain language, no jargon
- [ ] At least 800 words total
- [ ] Closing "GLIDE is built by…" line present before References
- [ ] References section present if any paper was cited
- [ ] File saved to `newsletters/YYYY-MM.md`
