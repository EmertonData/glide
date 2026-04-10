# Evaluation Workflow

This page explains how to plan and execute a rigorous evaluation with GLIDE — from raw data to a valid, debiased metric estimate.

---

## The evaluation problem

Suppose you have an AI system that produces outputs over a large dataset of $N$ items, and you want to measure its performance — for example, accuracy, relevance, or any scalar metric $\theta^*$.

Computing $\theta^*$ exactly requires reliable labels for every item. Human labels are reliable but expensive. Proxy labels (e.g., from an LLM-as-Judge) are cheap but biased: naively averaging them gives a systematically wrong answer.

GLIDE's approach is to use **proxy labels for all $N$ items** and **human labels for a small subset $n \ll N$** to produce an unbiased estimate $\hat{\theta}$ with a valid confidence interval. This is the prediction-powered inference paradigm.

The workflow has three stages:

```
Large proxy-labelled dataset
         │
         ▼
  [1] Sampler — which records to annotate?
         │
         ▼
  Human annotation of selected records
         │
         ▼
  [2] Estimator — what is the metric?
         │
         ▼
  Point estimate + confidence interval
```

---

## Stage 1 — Sampling: which records to annotate?

You start with a fully proxy-labelled dataset. The sampler's job is to select $b$ records for human annotation and to record the sampling probability $\pi_i$ used for each record. These probabilities are needed by the downstream estimator to correct for non-uniform selection.

### Do you need a sampler?

If you already have human labels for a uniformly drawn random subset of the data — for example, from a pre-existing annotation campaign — you can skip the sampling stage entirely and go straight to the estimator.

A sampler is useful when you still need to **allocate an annotation budget** and want to do so efficiently.

### Choosing a sampler

GLIDE provides two samplers. The choice depends on the structure of your data:

| Situation | Recommended sampler |
|---|---|
| No natural grouping in the data; you want to focus on uncertain predictions | [`ActiveSampler`](samplers.md#active-sampler) |
| Data naturally splits into groups (language, domain, topic, …) | [`StratifiedSampler`](samplers.md#stratified-sampler) |

**[`ActiveSampler`](samplers.md#active-sampler)** concentrates the budget on records where the proxy is least confident, by drawing each record with a probability proportional to its uncertainty score. This is variance-optimal when the dataset has no exploitable group structure.

**[`StratifiedSampler`](samplers.md#stratified-sampler)** partitions the dataset into strata and allocates the budget across them before sampling. Neyman allocation (the default) assigns more budget to strata where proxy variance is highest. Proportional allocation is a simpler baseline that replicates uniform random sampling within each stratum.

Both samplers produce two fields per record that the downstream estimator consumes directly:

| Field | Description |
|---|---|
| $\pi_i$ | Drawing probability used to select record $i$ |
| $\xi_i$ | Indicator: $1$ if selected for annotation, $0$ otherwise |

Records with $\xi_i = 1$ are the ones to send for human annotation.

---

## Stage 2 — Estimation: what is the metric?

Once human labels $Y_i$ have been collected for the $\xi_i = 1$ records, the estimator combines them with the proxy labels to produce an unbiased mean estimate and a confidence interval.

### Choosing an estimator

The right estimator depends on how the labeled subset was drawn:

| How were labels collected? | Recommended estimator |
|---|---|
| Uniform random sample (no sampler used, or `StratifiedSampler` with proportional allocation) | [PPI++](estimators.md#prediction-powered-inference-ppi) or [Stratified PPI++](estimators.md#stratified-ppi) |
| Non-uniform sampling via `ActiveSampler` or `StratifiedSampler` with Neyman allocation | [ASI](estimators.md#active-statistical-inference-asi) |
| No proxy labels available; human labels only | [Classical](estimators.md) |

**[PPI++](estimators.md#prediction-powered-inference-ppi)** is the right default when labels were drawn uniformly at random. It corrects for proxy bias using the labeled subset and automatically down-weights an uninformative proxy via power-tuning.

**[Stratified PPI++](estimators.md#stratified-ppi)** applies PPI++ independently within each stratum and combines the results. It is the natural companion to `StratifiedSampler` with proportional allocation, and outperforms global PPI++ when proxy quality varies across strata.

**[ASI](estimators.md#active-statistical-inference-asi)** handles the general case of non-uniform sampling. It uses the $\pi_i$ values produced by any sampler to correct for the sampling design via Inverse Probability Weighting. Use it whenever labels were not drawn uniformly — in particular after `ActiveSampler` or `StratifiedSampler` with Neyman allocation.

### Compatible sampler–estimator pairs

The table below summarises which combinations are valid:

| Sampler | Allocation | Compatible estimator |
|---|---|---|
| *(no sampler, uniform labels)* | — | PPI++, Stratified PPI++ |
| `StratifiedSampler` | Proportional | PPI++, Stratified PPI++, ASI |
| `StratifiedSampler` | Neyman | ASI |
| `ActiveSampler` | — | ASI |

!!! warning "Mismatch between sampler and estimator"
    Using PPI++ or Stratified PPI++ after non-uniform sampling (Neyman or active) produces **invalid confidence intervals**. The labeled subset is no longer a representative random sample, so the uniform-sampling assumption underlying PPI++ is violated. Always use ASI when sampling probabilities differ across records.

---

## Full workflow example

Here is a concrete end-to-end sequence for a scenario where the data has natural strata (e.g., languages) and you want to allocate an annotation budget of $b = 200$:

1. **Proxy-label all $N$ records** with your automated judge.
2. **Run `StratifiedSampler`** with Neyman allocation and $b = 200$. This sets $\pi_i$ and $\xi_i$ for every record.
3. **Collect human annotations** $Y_i$ for each record where $\xi_i = 1$.
4. **Run `ASI`** on the full dataset (which now carries $\tilde{Y}_i$, $\pi_i$, $\xi_i$, and $Y_i$ for labeled records). Power-tuning is on by default.
5. **Read off** the point estimate $\hat{\theta}$ and the confidence interval $[\hat{\theta} - \delta, \hat{\theta} + \delta]$.

If instead labels already exist for a uniform random subset, skip steps 2–3 and run Stratified PPI++ directly.

---

## Choosing the right method at a glance

```
Do you have proxy labels for all records?
├── No  → Classical estimator (human labels only)
└── Yes → Do you need to allocate an annotation budget?
          ├── No (labels already collected uniformly) → PPI++ or Stratified PPI++
          └── Yes → Does your data have natural strata?
                    ├── No  → ActiveSampler + ASI
                    └── Yes → StratifiedSampler + ASI
                               (proportional allocation also allows Stratified PPI++)
```
