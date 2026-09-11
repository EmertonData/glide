# GenAI for Finance — Exercise Preparation TODO

Everything needed to get the two practice sessions running for 70 students. Slide preparation is out of scope (tracked separately). Organized as mini-tickets, roughly in dependency order.

The afternoon is scoped tightly around **faithfulness, LLM-as-judge, PPI, and GLIDE** — retriever evaluation (precision@k / MAP@k, retriever-vs-retriever comparison) has been dropped from the plan, which also removes the heaviest annotation burden from this list (no per-question, per-chunk ground-truth labeling needed).

The plan is ordered so the morning practice (Session 2 — Build, §4–§5) is fully deliverable, corrections included, before any afternoon practice (Session 4 — Evaluate Faithfulness, §6–§7) work starts. The hallucination taxonomy and claim dataset only matter for the afternoon, so there's no reason to touch them before the agentic RAG build is done.

---

## 1. Repository

- [ ] **1.1 Create a public GitHub repo** (e.g. `xhec-genai-finance-workshop`). Public so students can `git clone` / fork without an invite step.
- [ ] **1.2 Repo skeleton**:
  ```
  build/                          # instructor-only scripts that generate everything in data/
    generate_paragraph_claims.py  # taxonomy-driven claim generation (see §6.1)
    run_judge_scores.py           # precompute judge faithfulness scores (see §7.2)
  notebooks/
    01_build_agentic_rag.ipynb
    02_evaluate_faithfulness.ipynb
  corrections/
    correction_build.py
    correction_eval.py
  data/
    company_excerpt.pdf
    questions.json                # 15-20 due-diligence questions, used in the build session
    paragraph_claims.json         # (paragraph, claim, true_faithfulness_label, error_type) pairs — output of build/generate_paragraph_claims.py
    judge_scores.json             # precomputed judge faithfulness score per pair — output of build/run_judge_scores.py
  pyproject.toml
  .env.example
  README.md
  ```
  `build/` is instructor-only tooling (not shown to students as an exercise) — it's what turns the manual dataset-prep steps below into reproducible, rerunnable scripts, so a late change to the source PDF or the taxonomy doesn't mean redoing everything by hand.
- [ ] **1.3 `.env.example`** documenting the one variable every notebook needs: `ANTHROPIC_API_KEY` (the single shared key, distributed per §3). Keep this the *only* thing students configure.
- [ ] **1.4 README** with setup instructions written for someone opening the repo cold on an unfamiliar JupyterLab-like platform: clone, `pip install .` (from `pyproject.toml`), copy `.env.example` to `.env`, run the first cell.
- [ ] **1.5 Decide correction-release policy**: corrections live in the repo from day one (simplest, "cheating" is allowed by design per your own brief) vs. released progressively during the day. Recommend: ship everything from the start — matches your stated intent and removes a moving part on the day.
- [ ] **1.6 License / attribution note** for the financial document excerpt (public investor disclosure — fine to redistribute for teaching; still worth a one-line attribution in the README).

---

## 2. API budget: single shared key (simplified for year one)

First run of the course, very little prep time: one Anthropic API key, shared by every student and used for the instructor's own prep and judge runs. Console-side spend limit, manual monitoring. No per-student/per-group provisioning, no LiteLLM proxy, no host to provision.

- [ ] **2.1 Create a single API key in the Anthropic Console**, shared by all 70 students and also used for the instructor's own prep runs and the centralized judge batch (§6.1, §7.2).
- [ ] **2.2 Set a spend limit on the key**, sized for the full day: 70 students running both notebooks plus the centralized batch judge run. There's no per-key model restriction with a single shared key, so the notebooks themselves must pin students to Haiku by default; the spend cap is the backstop, not the primary control.
- [ ] **2.3 Monitor total spend manually via the Anthropic Console usage dashboard** throughout the day. Because everyone shares one key, a single runaway loop affects the whole room, not just one group — keep the dashboard open continuously (see §10.1) and be ready to revoke/reissue the key if spend spikes.
- [ ] **2.4 (Deferred — not for this run)** Per-student/per-group keys with individual spend limits and per-key model restriction, plus a LiteLLM proxy with live automated budget enforcement, would remove the single-point-of-failure and blast-radius risk of one shared key. Worth building for a second edition of the course; out of scope given this year's prep time.

---

## 3. API key distribution

- [ ] **3.1 Ship the shared key directly in `.env.example`** (or a one-line copy-pasteable setup snippet) — since every student uses the same key, there's no per-user mapping or distribution CSV to prepare.
- [ ] **3.2 Write the "my key doesn't work" runbook** (most likely causes: the shared key hit its daily spend limit, wrong key pasted, stale `.env`) so you're not debugging from scratch live.
- [ ] **3.3 Set up a Slack channel with the master organizers** to distribute and, if needed, rapidly re-share a rotated key mid-day — with a single key covering the whole room, this is the fastest path to get a fresh key in front of 70 students at once if the original gets revoked or exhausted.

---

## 4. Dataset for Session 2 (Build)

- [ ] **4.1 Pick one big tech company** (e.g. NVIDIA) and source its 10-K annual report or investor "key figures" brochure.
- [ ] **4.2 Select ~10 pages** spanning a **risk factors** section (prose, good for semantic search) and a **financial highlights table** (numeric, good for exact-match/regex retrieval) — the mix still makes hybrid search a real design choice in the build session, even though retrieval isn't separately graded in the afternoon.
- [ ] **4.3 Extract and clean** those pages into a standalone workshop PDF (trim, fix obvious OCR/formatting artifacts, keep the table structure legible).
- [ ] **4.4 Sanity-check chunking** against the extracted PDF using your own reference `chunk_text` implementation — confirm chunks are coherent and the financial table doesn't get mangled.
- [ ] **4.5 Write 15–20 due-diligence questions**, spanning factual/numeric lookup, risk synthesis, and at least one comparison question. Used to test the agent in the build session; not the basis for the faithfulness dataset in §6 (see the note in §6.1 on why).

---

## 5. Session 2 notebook — Build

Finish this entire section, corrections included — `chunk_text` (§5.1), `vectorize_text` (§5.2), and the ReAct loop assembly (§5.6) all need a written, tested correction — before starting any work in §6. The morning practice must stand on its own; nothing in the afternoon material should be touched until it does.

- [ ] **5.1 `chunk_text` exercise** + correction + inline unit test (assert on a small fixed example).
- [ ] **5.2 `vectorize_text` exercise** using a small local CPU embedding model (e.g. `sentence-transformers/all-MiniLM-L6-v2` or `multilingual-e5-small`) + correction + test.
- [ ] **5.3 `save_vectors` exercise**: store embeddings as a plain NumPy array in memory (no external vector DB) + correction + test.
- [ ] **5.4 `top_k_search` exercise**: vectorized cosine-similarity top-k + correction + test.
- [ ] **5.5 BM25 search exercise** (`rank_bm25`) and fusion with semantic search into one hybrid search tool + correction + test.
- [ ] **5.6 ReAct loop assembly exercise**: wire the search tool into a single simple ReAct agent (LangGraph or a manual loop — pick one, don't offer both) + correction + test.
- [ ] **5.7 Full top-to-bottom dry run** of the notebook against the real API using the shared key, to validate both correctness and real token/cost consumption per run — this run is also what produces the agent used for the optional spot-check in §6.4.

---

## 6. Dataset for Session 4 (Evaluate Faithfulness)

Starts only once §5 is fully done, corrections included — §6.4's optional spot-check needs the agent built there, and there's otherwise no reason to interleave afternoon dataset prep with the morning build.

**Hallucination taxonomy — use this to drive claim generation in §6.1.** Ten distortion patterns, each with the (source → hallucinated) shape a strong LLM should reproduce when asked to generate an "unfaithful" claim. Note how several of these (Contresens, Troncature, Simplification) keep the surface form almost identical to the source and only break one fact — that's exactly the "subtly wrong" quality the §6.2 quality-pass is checking for, so lean on these categories more than the more obviously-off ones (Invention, Accentuation) if the judge is catching everything too easily.

| Category | What breaks | Example: source → hallucinated |
|---|---|---|
| **Acronyme** | An acronym is expanded incorrectly | "PSTC (Plan Scientifique Technologique et Conseil)" → "PSTC (Plan Stratégique de Transformation de la Compagnie)" |
| **Troncature** | Information is cut short and distorted | "Quantmetry a quatre valeurs : A, B, C, D" → "A et B constituent les valeurs de Quantmetry" |
| **Synecdoque** | A whole is confused with one of its parts | "Le CSE de Quantmetry donne un avis favorable au projet d'intégration" → "Quantmetry est favorable au projet d'intégration" |
| **Contexte** | A property is inferred from the wrong context | "Data & Tech recrute des profils DA" → "Data & Tech recrute des profils Data Analyst" |
| **Agglomération** | A property is wrongly extended to another entity | "Quantmetry a mis en place un logiciel et un contrat spécifique RH" → "Quantmetry a pris des mesures spéciales RH dont un logiciel et un contrat" |
| **Invention** | A conclusion is asserted with no supporting evidence | "Quantmetry a un directeur financier" → "Quantmetry maintient sa stabilité financière grâce à un directeur financier" |
| **Simplification** | Information is simplified and distorted in the process | "Quantmetry automatise ses prédictions quand c'est possible" → "Quantmetry automatise ses prédictions" |
| **Amalgame** | One entity is confused with another | "Quantmetry prévoit les demandes de produits de ses clients" → "Quantmetry prévoit ses demandes de produits" |
| **Contresens** | A deduction is flatly reversed | "Quantmetry a vu sa rentabilité augmenter et passer de 10% à 12%" → "Quantmetry a vu sa rentabilité diminuer et passer de 10% à 12%" |
| **Accentuation** | Information is embellished or overstated | "Quantmetry a fait de la R&D" → "Quantmetry a investi dans la R&D" |

- [ ] **6.1 Implement `build/generate_paragraph_claims.py`: a paragraph-level faithfulness dataset with ground truth known by construction, using two fixed prompts to a strong LLM (Opus) per chunk.** For each of the document's ~40–80 chunks:
  - **Faithful-claim prompt**: "Paraphrase this paragraph in one short sentence, preserving every fact exactly."
  - **Applicability check**: first ask the LLM which of the ten taxonomy categories above plausibly apply to *this specific paragraph* (e.g. "Acronyme" only applies to a paragraph that actually contains an acronym) — don't force all ten onto every chunk.
  - **Unfaithful-claim prompt**, run once per applicable category: "Rewrite this paragraph into one short false sentence using this specific distortion: {category + description from the taxonomy above}. Keep the sentence plausible and close in form to the original — do not introduce an obviously absurd error."

  Generating one distorted claim per *applicable* category (typically 2–5 per chunk, sometimes more) rather than a single fixed distortion per chunk pushes the dataset well past the earlier ~100–200 estimate — realistically **200–400+ pairs** — for the same one-time script run. Since this is a centralized batch job (the shared key, run once by the instructor), the extra generations cost is negligible; it's not distributed across 70 students. The script writes every `(paragraph, claim, true_faithfulness_label, error_type)` triple to `paragraph_claims.json` (`error_type` is `null` for faithful claims). This is deliberately a *different, cheaper* source of ground truth than judging real end-to-end RAG answers: labeling here means constructing the claim, not reading and holistically judging an agent's output. It also directly fixes the sample-size problem: with only 15–20 whole Q&A pairs, PPI/GLIDE has nothing to visibly improve on — wide confidence intervals either way — while several hundred pairs gives real room for the debiased estimate to show a measurably tighter interval than the naive judge average. `error_type` isn't needed for the core exercise but, with this many examples per category, now supports a genuinely credible optional bonus slide on which distortion types the judge catches reliably and which it misses.
- [ ] **6.2 Quality-pass the unfaithful claims.** Skim every generated "unfaithful" claim and make sure it's *subtly* wrong in the spirit of the taxonomy's Contresens/Troncature/Simplification examples above, not absurdly wrong — a claim the judge model would obviously catch 100% of the time produces no interesting bias to correct, which kills the "aha" moment in §7.4. Rewrite any claim that's too easy, too ambiguous, or doesn't cleanly match its intended category.
- [ ] **6.3 No build step needed for the labeled/proxy split — it's drawn live in the Session 4 notebook using a GLIDE sampler.** `paragraph_claims.json` already carries `true_faithfulness_label` for every id (fully labeled by construction), so unlike a real deployment there's no annotation cost to precompute or bake into a static file. The exercise itself calls a `glide.samplers` sampler (e.g. `StratifiedSampler`, stratifying on `error_type` so the ~20–30 revealed ids span multiple distortion categories rather than being skewed toward whichever category happened to generate first; `UniformSampler` if a simpler baseline is preferred) to pick which ids get their true label "revealed" as the labeled set — the rest fall back to judge-score-only (proxy). This is exactly the annotation-simulation use case `glide.samplers` is built for, and doing it live (rather than shipping a precomputed `labeled_subset_ids.json`) means students see GLIDE's sampling API in action, not just its estimators.
- [ ] **6.4 (Optional, time permitting) Spot-check against real agent output.** Run the agent built in §5 on a couple of the due-diligence questions from §4.5 and eyeball whether its answers are faithful — a nice motivating aside, but not part of the graded statistical exercise, so skip it under time pressure.

---

## 7. Session 4 notebook — Evaluate Faithfulness with GLIDE

- [ ] **7.1 `score_faithfulness` exercise**: implement the LLM-as-judge call — given `(paragraph, claim)`, prompt the judge model and parse out a faithfulness score — + correction + test.
- [ ] **7.2 Implement `build/run_judge_scores.py`: precompute judge scores centrally.** Run `score_faithfulness` over every pair in `paragraph_claims.json` (§6.1 — now several hundred, since each paragraph yields one distortion per applicable category) yourself, once, using the shared key, and ship the results as `judge_scores.json`. Precomputing centrally — rather than having 70 students each call the judge on the same fixed data — protects the shared spend limit (§2.2) and guarantees everyone works from identical proxy labels; a few hundred one-time Opus calls is still cheap in absolute terms, since it's a single batch run, not 70 repetitions of it. Leave one or two cells where students make a *live* judge call themselves (on one pair of their choosing) purely so they've seen it happen, but the estimation exercise itself runs on the shipped data.
- [ ] **7.3 GLIDE exercise**: use a `glide.samplers` sampler (§6.3) on `paragraph_claims.json` to draw the labeled subset (its true labels are the labeled set), then feed those true labels alongside the full judge-score set (§7.2, the proxy-labeled set) into a GLIDE prediction-powered mean estimator, producing a debiased faithfulness-rate estimate with a confidence interval. This is the exercise students fill in — a short, well-scoped call into GLIDE's public sampler and estimator APIs, not a from-scratch implementation.
- [ ] **7.4 Interpretation exercise**: have students compute the naive estimate (just the mean of all judge scores, no debiasing) side by side with the GLIDE estimate, and explain the gap — with N≈150 and n≈25 this should now show a real, visible difference in both point estimate and interval width, unlike the whole-Q&A version. This is the intended "aha" moment of the day; if a dry run shows the gap is still too small to see, that's the signal to go back to §6.2 and make the unfaithful claims subtler (more judge error to correct), not to add more labeled examples.
- [ ] **7.5 Verify the GLIDE exercise against the actual installed `glide-py` API** — spot-check estimator names/signatures used in the notebook against the current package before finalizing, since the exercise text should mirror real usage.

---

## 8. Environment & dependencies

- [ ] **8.1 Pin dependencies in `pyproject.toml`**: `anthropic`, `sentence-transformers`, `rank_bm25`, `numpy`, `glide-py`, `pypdf` (or similar for PDF parsing), plus LangGraph if used for the ReAct loop. The `build/` scripts (§6.1, §7.2) share the same dependency set — no separate install path needed since they only run on your machine, not the students'.
- [ ] **8.2 Test a clean install** on an environment matching the target platform as closely as possible (Python version, OS) once the platform is known.
- [ ] **8.3 Pre-cache the embedding model weights** (bake into a platform image if possible, or have students download once before the session) to avoid 70 simultaneous Hugging Face downloads at exercise start.
- [ ] **8.4 Confirm CPU-only is sufficient** — no GPU request needed for either the embedding model or the API-based generation/judging.

---

## 9. Dry runs & rehearsal

- [ ] **9.1 Solo timing rehearsal** of both practice sessions end-to-end, notebook only, no lecture — validate the time budget holds, especially now that the freed-up afternoon time goes entirely to depth on GLIDE.
- [ ] **9.2 Beta test with a few colleagues** unfamiliar with the material, to catch notebook bugs and get a real difficulty read before 70 students hit it simultaneously.
- [ ] **9.3 Cost dry run**: simulate a realistic "3 attempts per exercise" pass through both notebooks and total the spend, to cross-check the $200 budget assumption against real measured usage rather than back-of-envelope estimates.

---

## 10. Day-of logistics

- [ ] **10.1 Keep the Anthropic Console usage dashboard open** throughout the day; be ready to revoke/reissue the shared key if trending over budget.
- [ ] **10.2 Have the key-distribution runbook (§3.2) on hand, and the organizers' Slack channel (§3.3) open** to push out a reissued key to everyone at once if it needs to be rotated mid-day.
- [ ] **10.3 Open a support channel** (shared doc / chat) for setup issues, separate from the main teaching flow, ideally staffed by a TA if one is available.

---

## 11. Certification quiz

- [ ] **11.1 Design a Google Form with 20 four-choice multiple-choice questions**, closing the day, covering both presentation sessions: LLM/RAG/agent fundamentals (Session 1) and faithfulness/LLM-as-judge/PPI/GLIDE fundamentals (Session 3). Timed for ~15 minutes.
- [ ] **11.2 Set up Google Forms auto-grading** (answer key + point values per question) so each student's score is available immediately after submission.
- [ ] **11.3 Confirm the pass threshold**: 80% (16/20 correct) required to obtain the GenAI for Finance certification from Emerton Data.
- [ ] **11.4 Dry-run the quiz** against the finalized slide content once Sessions 1 and 3 are locked, to confirm every question is answerable from what was actually taught and timing holds at ~15 minutes.
- [ ] **11.5 Decide how results and certificates are communicated** to students after the fact (e.g. Google Forms' built-in score release vs. a follow-up email from Emerton Data).
