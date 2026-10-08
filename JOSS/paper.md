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

Evaluating a generative AI (GenAI) system, for example by measuring its hallucination rate or the share of unsafe agent trajectories, requires reliable labels. Human experts provide them at high cost, while LLM-as-judge proxies are cheap but biased [@NEURIPS2023_91f18a12]. `GLIDE` (Generated Label Inference & Debiasing Engine) is a Python library built on prediction-powered inference [@angelopoulos2023prediction; @angelopoulos2023ppiplusplus]. It combines a small set of human labels with a large set of proxy labels to produce debiased metric estimates whose confidence intervals stay valid however poor the proxy is. Behind a scikit-learn-style API, it offers sampling designs that choose which observations humans should label, and estimators that report the effective sample size, meaning the number of human labels saved by the proxy. It also provides monitors that track a metric over batches with anytime-valid bounds, which control false alarms under repeated evaluation [@podkopaev2021tracking; @waudbysmith2024time; @zhang2026prediction].

![The evaluation workflow `GLIDE` implements. Left: prediction-powered inference debiases a cheap proxy metric using a small human-annotated control sample. Right: the resulting three-step decomposition around which the API is organized.\label{fig:workflow}](glide-workflow.png)

# Statement of need

Evaluating a deployed GenAI system under a fixed annotation budget requires combining a few reliable human annotations with a large set of proxy annotations, such as LLM-as-judge labels. The literature has addressed the resulting complications one strand at a time: heterogeneous strata, handled by stratified sampling [@NEURIPS2024_c9fcd02e; @fogliato2024framework]; small labeled sets, handled by bootstrap intervals [@kluger2025prediction]; correlated units, handled by clustered inference; multiple judges, handled by multi-proxy aggregation [@shan2025sada; @cowen2026multiple]; judge uncertainty, exploited by active sampling [@pmlr-v235-zrnic24a; @gligoric2025can]; unequal human and LLM costs, handled by cost-aware sampling [@angelopoulos2025cost]; and continuous monitoring, made safe from false alarms by anytime-valid confidence sequences [@podkopaev2021tracking; @waudbysmith2024time; @zhang2026prediction].

These methods share one template, yet they are scattered across papers with heterogeneous notation and partial reference implementations [@song2026demystifying], some in R and some in Python, none covering more than a slice of the design space. Combining these methods currently means stitching together several repositories and verifying their consistency by hand. `GLIDE` closes that gap by treating sampling, estimation and monitoring as one consistent family, and it quantifies the return on the judge: the effective sample size says how many extra expert annotations the proxy is worth.

# State of the field

`ppi_py` [@angelopoulos2023prediction] has long been the reference implementation of the PPI family. This open-source Python package covers means, generalized linear models and M-estimators. The remaining methods above live in single-paper repositories: `ssepy` for stratified sampling and estimation [@fogliato2024framework], `active-inference` and `confidence-driven-inference` for active designs [@pmlr-v235-zrnic24a; @gligoric2025can], `PTDBoot` for the bootstrap variants [@kluger2025prediction], and `sada` for multi-proxy aggregation [@shan2025sada]. Each repository is self-consistent, but has its own data conventions, no shared sampling layer and no monitoring counterpart.

Evaluation orchestration frameworks such as RAGAS [@es-etal-2024-ragas], DeepEval [@Ip_deepeval_2026], TruLens [@trulens] and Inspect [@UK_AI_Security_Institute_Inspect_AI_Framework_2024] solve the engineering problem of running an evaluation and producing LLM-as-judge labels, but do not address the statistical reconciliation of these labels with ground-truth human expert annotations. `GLIDE` can thus consume the outputs of such pipelines, combine them with ground-truth labels, and supply the downstream rigorous statistical estimation layer.

We built a new library rather than contributing these methods upstream for two reasons.

The first is audience and scope. `ppi_py` is the reference implementation of the seminal PPI work and remains the right tool for inference on general estimands (e.g., quantiles, regression coefficients). It starts from an already-labeled set, so sampling design falls outside its scope. `GLIDE` addresses the broader community of engineers and evaluation teams, and buys that accessibility by narrowing the estimand to the mean, the workhorse of system evaluation. The narrowing also buys breadth: `ppi_py` predates most of the extensions listed above, whereas `GLIDE` brings cost-aware and active designs, stratified and clustered variants, multi-proxy aggregation, bootstrap intervals and anytime-valid monitoring into one consistent framework, several of them in their first public implementation.

The second reason follows from the first. Serving that audience requires an API that reconciles sampling, estimation and monitoring in one modular architecture. `GLIDE` provides a unified API that lets users switch easily between methods, and lets contributors add a new estimator, sampler or monitor without having to understand or modify the rest of the code base.

# Software design

`GLIDE` organizes evaluation around three sequential steps inherited from survey theory (\autoref{fig:workflow}): sampling, which selects which observations deserve a human label; annotation, which only domain experts can perform; and estimation, which combines the labels and the proxy predictions into a debiased estimate with a confidence interval. Monitoring is a fourth component that wraps per-batch estimates in an anytime-valid bound. Samplers expose `sample`, estimators `estimate`, monitors `detect`, so a complete workflow fits in a handful of lines and a new component plugs in by implementing one of those three interfaces.

Two design decisions carry most of the library's behavior.

The first is the data contract: one aligned array per collection of labels, with `numpy.nan` marking the unlabeled entries of the ground-truth array, following the conventions scikit-learn established across the data science community [@Pedregosa_Scikit-learn_Machine_Learning_2011].

The second is a middle layer of abstraction: each estimator family factors its computation into a `MeanEstimationEngine`, which an estimator calls once over the whole dataset and a monitor calls once per batch. Because estimators and monitors share this middle layer, contributors can readily extend any estimator into its anytime-valid monitor counterpart, without reimplementing its statistics.

Every new estimator, sampler or monitor passes three lines of defense. First, unit tests, with 100% coverage enforced and docstring examples executed as tests, pin the numerics. Second, functional tests assert statistical properties and equivalences, e.g., a stratified estimator applied to a single stratum must agree exactly with the uniform one. Third, scientific validation notebooks run Monte Carlo studies on synthetic data to assess the validity of the implementation, through coverage and interval width reduction for estimators, and false-alarm and miscoverage rates for monitors. These notebooks produce reproducible figures for the standard analyses of this literature. All three run in continuous integration, which not only lints, type-checks and tests the code base but also executes and renders every notebook in the documentation.

The package is also built to stay easy to depend on: `scipy` [@2020SciPy-NMeth] is its single runtime dependency. Dependency support windows follow SPEC 0, and releases follow semantic versioning.

# Research impact statement

The statistical framework behind the library is described in a companion methods paper presented at the ICML 2026 workshop on statistical frameworks for uncertainty in agentic systems [@martinon2026industrializing]. The validation notebooks described above cover all prediction-powered estimators and monitors: each attains its nominal coverage across proxy quality and confidence levels, never yields an interval wider than the human labeled-only baseline, and shows the effective sample size growing with proxy quality.

Additionally, three case studies run full workflows on public benchmarks whose proxy-labeled versions we release as documented datasets: agentic safety evaluation on R-Judge [@yuan-etal-2024-r]; text-to-SQL accuracy on Spider [@yu-etal-2018-spider]; and multilingual retrieval-augmented generation faithfulness on the MEMERAG dataset [@cruz-blandon-etal-2025-memerag]. Each case study is a reproducible notebook in the documentation, so the reported gains are easy to verify.

`GLIDE` has been developed in public since its first commit in March 2026, with roughly two releases per month and a dozen contributors. It has attracted more than a hundred stars and is disseminated through a monthly newsletter. It was the subject of a tutorial at PyData Amsterdam 2026 [@martinon2026pydata] and will be presented in a talk at Compute! Paris 2026 [@martinon2026compute], two venues whose audiences are practitioners. It will also be presented at the AI4Good workshop at NeurIPS 2026 in Paris [@martinon2026aigood].

# Code availability

The package can be installed from PyPI with `pip install glide-py`. The source is on GitHub (https://github.com/EmertonData/glide), together with the test suite, the scientific validation notebooks and the released datasets described above. The documentation, including tutorials, user guides and case studies, is at https://glide-py.readthedocs.io. A landing page (https://emertondata.github.io/glide/) serves as a reference hub linking to these resources, and hosts an interactive simulator in which users can see how the prediction-powered confidence interval narrows compared to the human-only one. Contributions are welcome by forking the repository and opening a pull request.

# AI usage disclosure

Generative AI was used in two places: in the development of the software, and in the writing of this manuscript. Because AI assistance in the software is embedded in the project's agile process, we describe that process here.

`GLIDE` was developed in two-week agile sprints. Each sprint opens with a planning phase in which development tickets are co-designed with Claude in plan mode. These tickets are refined until the project's tech lead has validated every corner case and design decision. Only then does a ticket reach a developer. Developers implement it with Claude Code, either alone or together with other developers during pair-programming sessions. They depart from the specification when unanticipated technical limits or opportunities surface. Developers are assisted by dedicated skills (ticket writing, pull-request creation, renaming, releases, literature watch, dependency updates). We refine these skills whenever they fall short.

Each pull request first receives an automated review from Claude's code-review plugin. Developer resolve it autonomously. A mandatory line-by-line human review by the tech lead follows. The repository's `CLAUDE.md` file encodes the project's conventions and architecture and is updated continuously. This setup has steadily reduced review time from one sprint to the next. Every decision, and the ownership of it, remains human.

For this manuscript, Claude was used to brainstorm and draft structure and formulations, to format the file, and to audit the draft against the journal's author guidelines. The authors wrote and own its argument, verified every reference and every reported figure against the sources and the repository, and take full responsibility for the result.

# Acknowledgements

We thank Anastasios Angelopoulos, Stephen Bates, Dan Kluger and Adam Fisch for useful discussions and guidance at the beginning of the project, and for their encouragement at ICML 2026. We are grateful to Pravallika Mavilla, Mohammed Raki, Guillaume D'Hérouville, Victor Woelffel and all contributors to the repository, and to Emerton Data for its financial support of the library as an open-source project. Finally, we thank the authors of the methods and reference implementations `GLIDE` builds upon.

# References
