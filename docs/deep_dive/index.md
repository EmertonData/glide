# Overview

This section is concerned with the statistical validity of the algorithms implemented in GLIDE. If you are not yet familiar with how those algorithms work, read the [User Guide](../user_guide/index.md) first.

---

## Scientific Validation

The scientific validation notebooks verify that GLIDE's algorithms are statistically correct. Each notebook runs a Monte Carlo experiment sweeping the correlation between true and proxy labels, and measures coverage validity, confidence interval width, and effective sample size. The [Methodology](scientific_validation/estimators/methodology.md) page defines these measures and the experimental protocol in detail.

---

## Case Studies

The case studies apply GLIDE end-to-end to real benchmarks, using an LLM judge as the proxy annotator. Each notebook demonstrates a complete pipeline: loading a public dataset, running GLIDE's sampling and estimation workflows, verifying coverage validity, and quantifying efficiency gains over classical baselines. They address the central empirical question: do GLIDE's debiasing guarantees hold on realistic data, and how large is the reduction in confidence interval width relative to using human labels alone?

| Task | Workflows | Notebook |
|---|---|---|
| Agentic safety evaluation (R-Judge) | Uniform PPI++, Stratified PPI++, ASI | [Agentic System Evaluation](case_studies/r_judge.ipynb) |
| Text-to-SQL accuracy estimation (Spider) | Stratified PPI++ | [Text-to-SQL Accuracy Estimation](case_studies/spider.ipynb) |
