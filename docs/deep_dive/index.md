# Overview

This section verifies key properties of the algorithms implemented in GLIDE, including statistical validity, data efficiency, and practical performance, through numerical experiments on synthetic and real datasets. If you are not yet familiar with how those algorithms work, read the [User Guide](../user_guide/index.md) first.

---

## Scientific Validation

The scientific validation notebooks verify statistical properties of GLIDE's algorithms on synthetic data through Monte Carlo experiments. The metrics and protocol vary by algorithm family; details on the used measures and experimental protocol for estimators are provided in the [Methodology](scientific_validation/estimators/methodology.md) page.

---

## Case Studies

The case studies apply GLIDE end-to-end to real benchmarks, using an LLM judge as the proxy annotator. Each notebook demonstrates a complete pipeline: loading a public dataset, running GLIDE's sampling and estimation workflows, verifying coverage validity, and quantifying efficiency gains over classical baselines. They address the central empirical question: do GLIDE's debiasing guarantees hold on realistic data, and how large is the reduction in confidence interval width relative to using human labels alone?

| Task | Workflows | Notebook |
|---|---|---|
| Agentic safety evaluation (R-Judge) | Uniform PPI++, Stratified PPI++, ASI | [Agentic System Evaluation](case_studies/r_judge.ipynb) |
| Text-to-SQL accuracy estimation (Spider) | Stratified PPI++ | [Text-to-SQL Accuracy Estimation](case_studies/spider.ipynb) |
