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

`ppi_py` [@angelopoulos2023prediction] has long been the reference implementation of the PPI family. This open-source Python package covers means, generalized linear models and M-estimators. The more recent methods live in single-paper repositories: `ssepy` for stratified sampling and estimation [@fogliato2024framework], `active-inference` and `confidence-driven-inference` for active designs [@pmlr-v235-zrnic24a; @gligoric2025can], `PTDBoot` for bootstrap variants [@kluger2025prediction], and `sada` for multi-proxy aggregation [@shan2025sada]. Each is self-consistent, but has its own data conventions, no shared sampling layer and no monitoring counterpart. As for evaluation orchestration frameworks such as RAGAS [@es-etal-2024-ragas], DeepEval [@Ip_deepeval_2026], TruLens [@trulens] and Inspect [@UK_AI_Security_Institute_Inspect_AI_Framework_2024], they run evaluations and produce LLM-as-judge labels, but do not reconcile those labels with human expert annotations. `GLIDE` can thus consume their outputs and supplies the downstream statistical estimation layer.

`GLIDE` was built rather than contributed upstream to `ppi_py` for two reasons. First, audience and scope: `ppi_py` targets general estimands (e.g., quantiles, regression coefficients) from an already-labeled set, so sampling design is out of its scope, and it predates most of the extensions above. `GLIDE` narrows the estimand to the mean, the workhorse of system evaluation, to serve engineers and evaluation teams, and in exchange brings cost-aware and active sampling, stratified and clustered variants, multi-proxy aggregation, bootstrap intervals and anytime-valid monitoring into one framework, several in their first public implementation. Second, serving that audience requires one modular API that reconciles sampling, estimation and monitoring, so users can switch between methods and contributors can add an estimator, sampler or monitor without modifying the rest of the code base.

# Software design

`GLIDE`'s design is organized around three sequential steps inherited from survey theory (\autoref{fig:workflow}): sampling, which selects the observations worth a human label; annotation, which only domain experts can perform; and estimation, which combines labels and proxy predictions into a debiased estimate with a confidence interval. Additionally, monitoring wraps per-batch estimates in an anytime-valid bound. Samplers expose `sample`, estimators `estimate` and monitors `detect`, so a full workflow fits in a handful of lines and a new component plugs in by implementing one of these interfaces. Data follows a simple contract: one aligned array per collection of labels, with `numpy.nan` marking unlabeled entries of the ground-truth array, in the spirit of scikit-learn conventions [@Pedregosa_Scikit-learn_Machine_Learning_2011]. Each estimator family factors its computation into a shared `MeanEstimationEngine`, called once over the whole dataset by an estimator and once per batch by a monitor, so contributors can extend an estimator into its monitor counterpart without reimplementing its statistics. In terms of quality gates, every new component passes three lines of defense, all run in continuous integration: unit tests with 100% coverage and executed docstring examples, functional tests of statistical properties (e.g., a stratified estimator on a single stratum must equal the uniform one), and Monte Carlo validation notebooks assessing coverage, interval width, false-alarm and miscoverage rates.

Regarding maintenance philosophy, `scipy` [@2020SciPy-NMeth] is the single runtime dependency, our dependency support windows follow SPEC 0, and releases follow semantic versioning.

# Research impact statement

`GLIDE` has been developed in public since its first commit in March 2026, with roughly two releases per month and a dozen contributors. It has attracted more than a hundred stars and is disseminated through a monthly newsletter. Beyond the open-source community, the package has already been used in several missions delivered by Emerton Data, a consulting firm, to evaluate industrialized GenAI systems with limited human annotation budgets.

The statistical framework behind the library is described in a companion methods paper presented at the ICML 2026 workshop on statistical frameworks for uncertainty in agentic systems [@martinon2026industrializing]. Validation notebooks cover all prediction-powered estimators and monitors: each attains its nominal coverage across proxy quality and confidence levels, never yields an interval wider than the human labeled-only baseline, and shows the effective sample size growing with proxy quality. Three reproducible case studies in the documentation run full workflows on public benchmarks, whose proxy-labeled versions we release as documented datasets: agentic safety evaluation on R-Judge [@yuan-etal-2024-r], text-to-SQL accuracy on Spider [@yu-etal-2018-spider], and multilingual retrieval-augmented generation faithfulness on MEMERAG [@cruz-blandon-etal-2025-memerag].

`GLIDE` has reached practitioner and research audiences alike. It was the subject of a tutorial at PyData Amsterdam 2026 [@martinon2026pydata] and will be presented at Compute! Paris 2026 [@martinon2026compute] and at the AI4Good workshop at NeurIPS 2026 in Paris [@martinon2026aigood].

# Code availability

GLIDE can be installed from PyPI with `pip install glide-py`. The source, test suite, scientific validation notebooks and released datasets are on GitHub (https://github.com/EmertonData/glide). The documentation, including tutorials, user guides and case studies, is at https://glide-py.readthedocs.io, and a landing page (https://emertondata.github.io/glide/) links to these resources and hosts an interactive simulator comparing prediction-powered and human-only confidence intervals. Contributions are welcome via issues or pull requests.

# AI usage disclosure

Generative AI was used both to develop the software and to prepare this manuscript. `GLIDE` is developed in two-week agile sprints. Development tickets are co-designed with Claude in plan mode and validated by the tech lead before reaching a developer. Developers implement them with Claude Code, assisted by dedicated skills (ticket writing, pull-request creation, releases, and others), and the repository's `CLAUDE.md` file encodes the project's conventions. Each pull request receives an automated review from Claude's code-review plugin, followed by a mandatory line-by-line human review by the tech lead. For the manuscript, Claude was used to brainstorm structure and formulations, format the file, and audit the draft against the journal's guidelines. All decisions remain human: the authors wrote the argument, verified every reference and figure, and take full responsibility for the result.

# Acknowledgements

We thank Anastasios Angelopoulos, Stephen Bates, Dan Kluger and Adam Fisch for useful discussions and guidance at the beginning of the project, and for their encouragement at ICML 2026. We are grateful to Pravallika Mavilla, Mohammed Raki, Guillaume D'Hérouville, Victor Woelffel and all contributors to the repository, and to Emerton Data for its financial support of the library as an open-source project. Finally, we thank the authors of the methods and reference implementations `GLIDE` builds upon.

# References
