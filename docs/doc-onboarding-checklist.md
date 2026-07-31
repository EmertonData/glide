# GLIDE Onboarding Checklist

Welcome to GLIDE. This checklist points a new contributor to the right material in order, rather than re-explaining what's already documented. Work through it top to bottom, checking items off as you go. The overall learning time following this roadmap should not exceed two days.

---

### Orientation

- [ ] Read GLIDE's README — the repository's main index page — in particular "What is GLIDE?", "Why GLIDE?", and the "Implemented Algorithms" table.
- [ ] Open the online documentation, linked from the README's Documentation section. This is where you will find the User Guide, Tutorials, and Deep Dive material referenced throughout this checklist. You do not need to read all these assets now, simply know where to find them.
- [ ] In the Getting Started section of the documentation, skim the Installation guide and read through the Quickstart notebook: it estimates a bias-corrected metric with `PPIMeanEstimator` from a small labeled sample and a large proxy-labeled pool, giving you a first concrete example to anchor the theory in the next sections.

### 1. Local setup and software engineering practices

- [ ] Clone the GLIDE GitHub repository locally.
- [ ] Follow the "Setup" section of the Contributing guide in the online documentation: install `uv`, run `make venv`, verify with `make tests`, install the pre-commit hooks with `uv run prek install`, and optionally run `make test-notebooks`.
- [ ] Read `.claude/CLAUDE.md` which details the module breakdown, how `tests/unit/` and `tests/functional/` mirror it, the coverage/mocking/naming/typing conventions, and the code hygiene and consistency/propagation expectations.
- [ ] Confirm you can run `make lint`, `make type-check`, and `make tests` locally without errors on a clean checkout.

Note: `make doc` builds and serves the full site locally, but it executes every tutorial notebook and takes 10+ minutes — don't run it just to check a small doc change. If your change doesn't touch any notebook or code, you can speed this up a lot by temporarily setting `execute: false` on the `mkdocs-jupyter` plugin in `mkdocs.yml` before running `make doc` locally (revert it before committing).

### 2. Ways of working

- [ ] Read `CONTRIBUTING.md` in full: covers the issue → branch → PR → review → merge flow and the step-by-step recipes for adding a new estimator/sampler/monitor.

GLIDE also uses a set of Claude Code skills to automate repetitive parts of this workflow:

- [ ] `create-pr` — pushes the current branch, fills in the PR template from your diff, updates the CHANGELOG, attaches labels, and opens the PR via `gh`. Invoke with something like "create a PR for this".
- [ ] `code-review` — reviews the pending diff on your branch across multiple dimensions before you request human review.
- [ ] `dependabot` — triages the oldest open Dependabot PR (classifies the bump, applies it, merges `main` to resolve conflicts); a maintainer still merges it manually.
- [ ] `ticket` — drafts a developer-ready ticket in `tickets/` from a paper, refactor request, or maintenance task.
- [ ] `release` — automates version bump, CHANGELOG, release PR, TestPyPI, tag, and GitHub release for shipping a new version.

### Checkpoint: Quiz 1

Once chapters 1 and 2 are checked off, take the first validation quiz (80% pass threshold): `docs/onboarding_quizz/quizz1.html`. To load it, run:

```bash
cd docs/onboarding_quizz && python3 -m http.server 8000
```

and open `http://localhost:8000/quizz1.html`.

---

### 3. The core idea: Prediction-Powered Inference (PPI)

Every method in GLIDE is a variation on one idea: you have **cheap, biased proxy labels** for an entire dataset (e.g. an LLM judge) and a **small, expensive, unbiased set of human labels**. PPI combines both to produce an estimate of the true mean that is unbiased and has lower variance than an estimate using human labels alone.

- [ ] Go through the PPI tutorial and make sure you understand how it works.
- [ ] Read the Estimators user guide until the end of the "Prediction-Powered Inference (PPI++)" section, including its "Mean estimation" and "Variance and confidence intervals" subsections.
- [ ] Skim `glide/estimators/ppi.py` (the `PPIMeanEstimator` class) to see where the quantities from the user guide live in code; some of the heavy lifting is delegated to internal helpers imported from elsewhere in the package, no need to chase those down at this stage.

Make sure you understand the role of power tuning and how it depends on the correlation between human and proxy (LLM judge) labels.

### 4. Stratified PPI

A small conceptual step from PPI: the same correction, applied per stratum instead of over the whole dataset.

- [ ] Go through the Stratified PPI tutorial.
- [ ] Read the "Stratified PPI++" section of the Estimators user guide.
- [ ] Skim `glide/estimators/stratified_ppi.py` (the `StratifiedPPIMeanEstimator` class).

In summary, PPI can be applied to stratified data by computing plain PPI within each stratum and combining them with a weighted average.

### 5. Active Statistical Inference (ASI) and Inverse Probability Weighting

The goal of this section is to leave with a clear, intuitive understanding of the IPW mechanism, since it is an important foundation for understanding samplers.

In summary, ASI is PPI where each sample also carries a sampling probability, and IPW up-weights samples that were unlikely to be picked to cancel the resulting bias.

- [ ] Read the ASI tutorial.
- [ ] Read the "Active Statistical Inference (ASI)" section of the Estimators user guide, including its "Mean estimation" subsection where $\pi_i$ enters the formula.
- [ ] Skim `glide/estimators/asi.py` (the `ASIMeanEstimator` class).

### 6. Samplers

Samplers decide *which* samples get sent for human annotation, and are what actually produces the $\pi_i$ that ASI (and other IPW-based estimators) consume.

- [ ] Read the ASI scientific validation notebook (in the Deep Dive section) to see a sampler and an estimator used together end to end, and how the resulting estimates are checked across many random seeds.
- [ ] Read the introduction and the "Stratified Sampler" and "Active Sampler" sections of the Samplers user guide, including the intro table of $\pi_i$/$\xi_i$.
- [ ] Identify which of these produces an informative, heterogeneous per-sample probability meant to be consumed by IPW-based estimators (`ActiveSampler`), versus one whose sampling design is handled structurally rather than through per-unit weighting (`StratifiedSampler`).
- [ ] Skim `glide/samplers/active.py` (the `ActiveSampler` class) and `glide/samplers/stratified.py` (the `StratifiedSampler` class).

### 7. Full workflow checkpoint

Before moving to monitoring, consolidate everything above into one mental pipeline: **simulate → sample → estimate**.

- [ ] Read the "Summary: what each strategy contributes" section of the Cost-Optimal Sampling tutorial: it runs the full simulate → sample → estimate pipeline end to end and compares a proxy-only, a human-only, and a cost-optimal strategy on cost and bias.

Note: you do not need to understand the inner workings of the Cost-Optimal Sampler.

### 8. Monitoring and the peeking problem

- [ ] Read the Asymptotic PPRM monitor tutorial.
- [ ] Read "The Monitoring Problem" section of the Monitors user guide: calling an estimator repeatedly as new data arrives ("peeking") inflates the false alarm rate beyond the nominal confidence level, because each call is only individually valid, not jointly valid across time.
- [ ] Read "Confidence Sequences" and "Asymptotic Confidence Sequences" in the same guide: a confidence sequence replaces a single confidence interval with a sequence of intervals that are simultaneously valid **at all times**, not just at one fixed sample size.
- [ ] Skim `glide/monitors/asymptotic_pprm.py` (the `AsymptoticPPRM` class).

### Checkpoint: Quiz 2

Once chapters 3 to 8 are checked off, take the second validation quiz (80% pass threshold): `docs/onboarding_quizz/quizz2.html`. Load it the same way as Quiz 1, then open `http://localhost:8000/quizz2.html`.

---

## Further reading (not required to finish the checklist, useful later)

- There are more tutorials in the documentation beyond the ones covered above, walking through additional estimation methods and use cases.
- The Deep Dive section of the documentation has further scientific validation notebooks and real-world case studies applying the full simulate → sample → estimate pipeline to concrete benchmarks.
- The README's `📖 References` list, if you want to go read the original papers behind a specific method.

If you have the time, it's worth exploring this material at your own pace; there's no expectation to cover all of it right away.
