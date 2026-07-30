# GLIDE Onboarding Checklist

Welcome to GLIDE. This checklist points a new contributor to the right material in order, rather than re-explaining what's already documented. Work through it top to bottom, checking items off as you go. The overall earning time following this roadmap should not exceed two days.

---

### Orientation

- [ ] Read GLIDE's [README](https://github.com/EmertonData/glide) — the project's main index page — in particular "What is GLIDE?", "Why GLIDE?", and the "Implemented Algorithms" table.
- [ ] Open the [online documentation](https://glide-py.readthedocs.io/en/latest/), linked from the README's Documentation section. This is where you will find the User Guide, Tutorials, and Deep Dive material referenced throughout this checklist.

### 1. The core idea: Prediction-Powered Inference (PPI)

Every method in GLIDE is a variation on one idea: you have **cheap, biased proxy labels** for an entire dataset (e.g. an LLM judge) and a **small, expensive, unbiased set of human labels**. PPI combines both to produce an estimate of the true mean that is unbiased and lower-variance than using human labels alone.

- [ ] Read the [Estimators user guide](https://glide-py.readthedocs.io/en/latest/user_guide/estimators/) until the end of the "Prediction-Powered Inference (PPI++)" section, including its "Mean estimation" and "Variance and CIs" subsections.
- [ ] Go through the [PPI tutorial](https://glide-py.readthedocs.io/en/latest/tutorials/ppi/) and make sure you understand how it works.

Make sure you understand the role of power tuning and how it depends on the correlation between human and proxy (LLM judge) labels.

### 2. Stratified PPI

A small conceptual step from PPI: the same correction, applied per stratum instead of over the whole dataset.

- [ ] Read the "Stratified PPI++" section of the [Estimators user guide](https://glide-py.readthedocs.io/en/latest/user_guide/estimators/).
- [ ] Go through the [Stratified PPI tutorial](https://glide-py.readthedocs.io/en/latest/tutorials/stratified_ppi/).

In summary, PPI can be applied to stratified data by computing plain PPI within each stratum and combining them with a weighted average.

### 3. Active Statistical Inference (ASI) and Inverse Probability Weighting

The goal of this section is to leave with a clear, intuitive understanding of the IPW mechanism, since it is an important foundation to understand samplers.

ASI is PPI where each sample also carries a **sampling probability** $\pi_i$: the probability that sample $i$ was selected for human annotation. Samples that were unlikely to be picked ($\pi_i$ small) but were picked anyway are up-weighted by $1/\pi_i$; samples that were very likely to be picked are barely up-weighted at all. In expectation over the random selection, this correction exactly cancels the bias introduced by non-uniform sampling — a commonly used principle in survey statistics.

- [ ] Read the "Active Statistical Inference (ASI)" section of the [Estimators user guide](https://glide-py.readthedocs.io/en/latest/user_guide/estimators/), including its "Mean estimation" subsection where $\pi_i$ enters the formula.
- [ ] Read the [ASI tutorial](https://glide-py.readthedocs.io/en/latest/tutorials/asi/).

### 4. Samplers

Samplers decide *which* samples get sent for human annotation, and are what actually produces the $\pi_i$ that ASI (and other IPW-based estimators) consume.

- [ ] Read the introduction and the "Stratified Sampler" and "Active Sampler" sections of the [Samplers user guide](https://glide-py.readthedocs.io/en/latest/user_guide/samplers/), including the intro table of $\pi_i$/$\xi_i$.
- [ ] Identify which of these produces an informative, heterogeneous per-sample probability meant to be consumed by IPW-based estimators (`ActiveSampler`), versus one whose sampling design is handled structurally rather than through per-unit weighting (`StratifiedSampler`).
- [ ] Read the [ASI scientific validation notebook](https://glide-py.readthedocs.io/en/latest/deep_dive/scientific_validation/estimators/asi/) to see a sampler and an estimator used together end to end, and how the resulting estimates are checked across many random seeds.

### 5. Full workflow checkpoint

Before moving to monitoring, consolidate everything above into one mental pipeline: **simulate → sample → estimate**.

- [ ] Read the "Summary: what each strategy contributes" section of the [Cost-Optimal Sampling tutorial](https://glide-py.readthedocs.io/en/latest/tutorials/cost_optimal/#summary-what-each-strategy-contributes): it runs the full simulate → sample → estimate pipeline end to end and compares a proxy-only, a human-only, and a cost-optimal strategy on cost and bias.

### Checkpoint: Quiz 1

Once chapters 1 to 5 are checked off, take the first validation quiz (20 questions, 90% pass threshold): `docs/onboarding_quizz/quizz1.html`. To load it, clone the repository, then run:

```bash
cd docs/onboarding_quizz && python3 -m http.server 8000
```

and open `http://localhost:8000/quizz1.html`.

---

### 6. Monitoring and the peeking problem

- [ ] Read "The Monitoring Problem" section of the [Monitors user guide](https://glide-py.readthedocs.io/en/latest/user_guide/monitors/): calling an estimator repeatedly as new data arrives ("peeking") inflates the false alarm rate beyond the nominal confidence level, because each call is only individually valid, not jointly valid across time.
- [ ] Read "Confidence Sequences" and "Asymptotic Confidence Sequences" in the same guide: a confidence sequence replaces a single confidence interval with a sequence of intervals that are simultaneously valid **at all times**, not just at one fixed sample size.
- [ ] Read the [Asymptotic PPRM monitor tutorial](https://glide-py.readthedocs.io/en/latest/tutorials/asymptotic_pprm_monitor/).

### 7. Local setup and software engineering practices

This is the point where you start actually running things yourself.

- [ ] Follow the "Setup" section of the online [Contributing guide](https://glide-py.readthedocs.io/en/latest/contributing/): install `uv`, run `make venv`, verify with `make tests`, install the pre-commit hooks with `uv run prek install`, and optionally run `make test-notebooks`.
- [ ] Read `.claude/CLAUDE.md` end to end: covers the module breakdown, how `tests/unit/` and `tests/functional/` mirror it, the coverage/mocking/naming/typing conventions, and the code hygiene and consistency/propagation expectations.
- [ ] Confirm you can run `make lint`, `make type-check`, and `make tests` locally without errors on a clean checkout.

### 8. Ways of working

- [ ] Read `CONTRIBUTING.md` in full: covers the issue → branch → PR → review → merge flow and the step-by-step recipes for adding a new estimator/sampler/monitor.

GLIDE also uses a set of Claude Code skills to automate repetitive parts of this workflow:

- [ ] `create-pr` — pushes the current branch, fills in the PR template from your diff, updates the CHANGELOG, attaches labels, and opens the PR via `gh`. Invoke with something like "create a PR for this".
- [ ] `/code-review` — reviews the pending diff on your branch across multiple dimensions before you request human review.
- [ ] `dependabot` — triages the oldest open Dependabot PR (classifies the bump, applies it, merges `main` to resolve conflicts); a maintainer still merges it manually.
- [ ] `ticket` — drafts a developer-ready ticket in `tickets/` from a paper, refactor request, or maintenance task (this very file was produced with it).
- [ ] `release` — automates version bump, CHANGELOG, release PR, TestPyPI, tag, and GitHub release for shipping a new version.

### 9. Documentation map

GLIDE's docs are built with MkDocs and organized by `mkdocs.yml`:

- [ ] **Getting Started** — installation and a quickstart notebook.
- [ ] **User Guide** — the mathematical/conceptual reference for samplers, estimators, and monitors (what you've been reading in this checklist).
- [ ] **Tutorials** — one notebook per estimator/sampler/monitor showing how to call the public API on a realistic scenario.
- [ ] **Deep Dive** — scientific validation notebooks (one per estimator/monitor, proving statistical correctness) and case studies (`r_judge`, `spider`, `memerag` — real evaluation workflows).
- [ ] **API Reference** — auto-generated from docstrings.

Note: `make doc` builds and serves the full site locally, but it executes every tutorial notebook and takes 10+ minutes — don't run it just to check a small doc change.

### Checkpoint: Quiz 2

Once chapters 6 to 9 are checked off, take the second validation quiz (20 questions, 90% pass threshold): `docs/onboarding_quizz/quizz2.html`. Load it the same way as Quiz 1, then open `http://localhost:8000/quizz2.html`.

---

## Further reading (not required to finish the checklist, useful later)

- Other tutorials not covered above: `clustered_ppi.ipynb`, `multi_ppi.ipynb`, `cost_optimal_random.ipynb` (all in `docs/tutorials/`).
- Other scientific validation notebooks: the PTD family and clustered/multi variants under `docs/deep_dive/scientific_validation/estimators/`.
- The README's `📖 References` list, if you want to go read the original papers behind a specific method.
