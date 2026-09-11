---
title: 'GLIDE: A Python library for prediction-powered evaluation of GenAI systems'
tags:
  - Python
  - prediction-powered inference
  - statistical inference
  - AI evaluation
  - LLM-as-judge
  - confidence intervals
  - anytime-valid inference
authors:
  - name: Grégoire Martinon
    orcid: 0009-0000-6035-460X
    corresponding: true
    affiliation: 1
  - name: Ibrahim Merad
    orcid: 0000-0003-0907-9096
    affiliation: 1
affiliations:
  - name: Emerton Data, France
    index: 1
date: 8 September 2026
bibliography: paper.bib
---

# Summary

Deciding whether a generative AI (GenAI) system behaves acceptably is a measurement problem: what fraction of answers contain hallucinations, or what fraction of an agent's trajectories are unsafe? On the one hand, human expert annotation on generated outputs answers such questions reliably but slowly and at high cost. On the other hand, the popular LLM-as-judge paradigm provides cheap but biased proxy annotations [@NEURIPS2023_91f18a12]. Prediction-powered inference (PPI) dissolves that trade-off: it combines a small human-labeled sample (also name ground truths) with a large pool of proxy-labeled observations into a debiased performance measurement. The obtained metric comes with a confidence interval that remains valid however poor the proxy is [@angelopoulos2023prediction; @angelopoulos2023ppiplusplus].

`GLIDE` (Generated Label Inference & Debiasing Engine) brings that literature into a single library. It specializes in risk and performance estimation of complex AI system producing large amounts of unstructued data, like AI agents. The vast majority of these metrics, be they accuracy, hallucination rate or an unsafe-trajectory rate, can be expressed as population means, estimated on samples of generated outputs. Since human annotation is highly limited, evaluating such systems requires to borrow ideas from survey theory, where a small sample is used to estiamte a population statistic. A common practice nowadays is to supplement this sampled, expensive annotation, with large scale LLM-as-Judge proxy annotations. Evaluating thus needs sampling and combining different kinds of labels: ground truth and proxy labels.

GLIDE addresses these challenges and exposes multiple sampling designs, performance estimators and drift monitors behind a scikit-learn-style API. Every estimator returns a result object carrying the point estimate, the confidence interval, and the effective sample size. The latter can be understood as the return on investment of the LLM-as-Judge: it is the number of human labels a classical survey estimator would have needed to reach the same precision. Monitors consume a stream of batches, each carrying a few human labels and many proxy labels, and return the running metric together with an anytime-valid bound. The library is documented with runnable tutorials, a mathematical user guide, a Monte Carlo validation notebook for every estimator and monitor, and several end-to-end case studies on public benchmarks.

![The evaluation workflow `GLIDE` implements. Left: prediction-powered inference debiases a cheap proxy metric using a small human-annotated control sample. Right: the resulting three-step decomposition around which the API is organized.\label{fig:workflow}](glide-workflow.png)

# Statement of need

Evaluating a deployed GenAI system under a fixed annotation budget accumulates complications, each addressed by a separate strand of the literature.

- **Cost asymmetry.** Expert review costs dollars per unit, an LLM-as-judge call cents. Cost-aware sampling spends a money budget rather than a label count [@angelopoulos2025cost].
- **Heterogeneous strata.** LLM-as-Judge quality varies across axes known in advance: domain, language, query type. Stratified sampling tunes evaluation per stratum [@NEURIPS2024_c9fcd02e; @fogliato2024framework].
- **Available judge uncertainty.** LLM-as-Judges expose per-observation uncertainty scores that reveal where a human label pays off. Active sampling concentrates the budget there [@pmlr-v235-zrnic24a; @gligoric2025can].
- **Correlated units.** Annotation is granted to whole groups: a reviewer labels every sentence of a paragraph at once, while the metric is the proportion of hallucinated sentences. Treating those sentences as independent understates the variance, so clustered inference works on group means.
- **Small labeled sets.** Below a few dozen labels per stratum, the central-limit approximation degrades. Bootstrap intervals stay valid [@kluger2025prediction].
- **Multiple judges.** Modern evaluation stacks accumulate different LLM-as-Judges of differing cost and skill. Multi-proxy aggregation weights them optimally, so an extra judge only narrows the interval [@shan2025sada; @cowen2026multiple].
- **Continuous monitoring.** Re-estimating a metric batch after batch inevitably produces false alarms. Anytime-valid confidence sequences make peeking safe under a single budget [@podkopaev2021tracking; @waudbysmith2024time; @zhang2026prediction].

These methods share one template: few reliable annotations combined with a large-scale proxy ones, but they are scattered across papers with heterogeneous notation and partial reference implementations [@song2026demystifying], some of them in R, none covering more than a slice of the grid. A practitioner who needs stratified sampling with a small per-stratum budget, or a multi-judge-powered metric tracked weekly in production, must currently stitch together code from several academic repositories and verify by hand that the sampling design and the estimator are mutually consistent.  `GLIDE` closes that gap by treating sampling, estimation and monitoring as one consistent family. It also quantifies the return on the judge: the effective sample size says how many extra expert annotations the proxy is worth, on top of those actually paid for.

# State of the field

`ppi_py` [@angelopoulos2023prediction] is the reference implementation of the PPI family and covers means, generalized linear models and M-estimators. The remaining methods above live in single-paper repositories, `ssepy` for stratified sampling and estimation [@fogliato2024framework], `active-inference` and `confidence-driven-inference` for active designs [@pmlr-v235-zrnic24a; @gligoric2025can], `PTDBoot` for the bootstrap variants [@kluger2025prediction], `sada` for multi-proxy aggregation [@shan2025sada], each self-consistent but with its own data conventions, no shared sampling layer, and no monitoring counterpart. Evaluation orchestration frameworks such as RAGAS [@es-etal-2024-ragas], DeepEval [@Ip_deepeval_2026], TruLens [@trulens] and Inspect [@UK_AI_Security_Institute_Inspect_AI_Framework_2024] solve the upstream problem of running an evaluation and producing LLM-as-Judge labels. `GLIDE` consumes their output, combines them with human experts labels, and supplies the downstream rigorous statistical estimation layer.

We built a new library rather than contributing these methods upstream for two reasons. The first is audience and scope. `ppi_py` established the reference implementation of PPI and remains the right tool for inference on general estimands; it starts from an already-labeled set, so sampling design falls outside its scope. `GLIDE` addresses the much broader community of engineers and evaluation teams, and buys that accessibility by narrowing the estimand to the mean, the workhorse of system evaluation. The narrowing also buys breadth: `ppi_py` predates most of the extensions listed above, whereas `GLIDE` brings cost-aware and active designs, stratified and clustered variants, multi-proxy aggregation, bootstrap intervals and anytime-valid monitoring into one consistent framework, several of them in their first public implementation.

The second reason follows from the first. Serving that audience means an API that reconciles sampling, estimation and monitoring into one modular architecture, in which a new estimator or sampler can be added without touching the rest of the code base.

# Software design

`GLIDE` organizes evaluation around three sequential steps inherited from survey theory (\autoref{fig:workflow}): sampling, which selects which observations deserve a human label; annotation, which only domain experts can perform; and estimation, which combines the labels and the proxy predictions into a debiased estimate with a confidence interval. Monitoring is a fourth component that wraps per-batch estimates in an anytime-valid bound. Samplers expose `sample`, estimators `estimate`, monitors `detect`, so a complete workflow fits in a handful of lines and a new component plugs in by implementing one of those three interfaces.

Two design decisions carry most of the library's behavior. The first is the data contract: one aligned array per signal, with `numpy.nan` marking the unlabeled entries of the ground-truth array, following the array conventions scikit-learn established across the data science community [@Pedregosa_Scikit-learn_Machine_Learning_2011]. The second is a middle layer of abstraction: each estimator family factors its computation into a `MeanEstimationEngine`, which an estimator calls once over the whole dataset and a monitor calls once per batch. Extending an estimator into its anytime-valid monitor counterpart therefore reuses the estimator's statistics instead of reimplementing them.

Every new estimator, sampler or monitor passes three lines of defense. Unit tests, with 100% coverage enforced and docstring examples executed as tests, pin the numerics. Functional tests assert statistical properties, among them the degeneracy identities that tie the family together: a stratified estimator applied to a single stratum, for instance, must return exactly what the pooled estimator returns. A scientific validation notebook then runs a Monte Carlo study of the properties the method claims, coverage at the nominal level, interval width against the labeled-only baseline, false-alarm and miscoverage rates for the monitors, and reports them as the figures that are standard in this literature. All three run in continuous integration, which also executes every notebook in the documentation, lints and type-checks the code base.

The package is also built to stay easy to depend on: `scipy` [@2020SciPy-NMeth] is its single runtime dependency, support windows follow SPEC 0, and releases follow semantic versioning.

# Research impact statement

The statistical framework behind the library is described in a companion methods paper presented at the ICML 2026 workshop on statistical frameworks for uncertainty in agentic systems [@martinon2026industrializing]. The validation notebooks described above cover all prediction-powered estimators and monitors: each attains its nominal coverage across proxy quality and confidence levels, never yields an interval wider than the human labeled-only baseline, and shows the effective sample size growing with proxy quality.

Three case studies then run the full workflow on public benchmarks whose proxy-labeled versions we release as documented datasets: agentic safety evaluation on R-Judge [@yuan-etal-2024-r]; text-to-SQL accuracy on Spider [@yu-etal-2018-spider]; and multilingual retrieval-augmented generation faithfulness on the MEMERAG dataset[@cruz-blandon-etal-2025-memerag]. Each case study is a reproducible notebook in the documentation, so the reported gains can be easily reproduced.

`GLIDE` has been developed in public since its first commit in March 2026, at roughly two releases per month, by seven contributors. It has attracted eighty-five stars and external forks, is disseminated through a monthly newsletter, and is the subject of a tutorial at PyData Amsterdam 2026 and a talk at Compute! Paris 2026, where practitioners are its intended audience.

# AI usage disclosure

`GLIDE` was developed in two-week agile sprints. Each sprint opens with a planning phase in which development tickets are co-designed with Claude in plan mode. These tickets are refined until every corner case and design decision is validated by the project's tech lead. Only then does a ticket reach a developer, who implements it with Claude Code, departing from the specification when unanticipated technical limits surface. Developers are assisted by dedicated skills (ticket writing, pull-request creation, renaming, releases, literature watch, dependency updates) reinforced whenever they fall short. Each pull request receives a first automated review that the developer resolves autonomously, followed by a mandatory line-by-line human review by the tech lead. The repository's `CLAUDE.md` file encodes the project's conventions and architecture and is updated continuously. This has cut review time significantly sprint after sprint. Every decision, and the ownership of it, remains human.

For this manuscript, Claude was used to brainstorm and draft structure and formulations, to format the file, and to audit the draft against the journal's author guidelines. The authors wrote and own its argument, verified every reference and every reported figure against the sources and the repository, and take full responsibility for the result.

# Acknowledgements

We thank Mohammed Raki, Guillaume D'Hérouville, Victor Woelffel and the other contributors to the repository for their work on estimators, samplers, documentation and tooling, and Emerton Data for supporting the development of the library as an open-source project. We also thank the authors of the methods and reference implementations `GLIDE` builds upon.

# References
