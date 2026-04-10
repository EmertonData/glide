# Evaluation Workflow

This page explains how to plan and execute a rigorous evaluation with GLIDE — from raw data to a valid, unbiased metric estimate.

---

## The evaluation problem

Suppose you have an AI system that produces answers over a large dataset of $N$ items, and you want to measure its performance — for example, its accuracy, relevance score, or any other scalar metric $\theta^*$.

Computing $\theta^*$ exactly requires reliable annotations $Y$ for every item. Human annotations are reliable but expensive, so in practice you only label a small subset of $n \ll N$ items. A natural shortcut is to use **proxy labels** $\tilde{Y}$ — automated predictions from, for example, an LLM-as-Judge — to cover all $N$ items cheaply. The problem: proxy labels are generally **biased** ($E[\tilde{Y}] \neq \theta^*$), so naively averaging them gives a systematically wrong estimate of $\theta^*$.

GLIDE addresses this by combining large pools of cheap proxy labels with small sets of human labels to produce unbiased, reliable estimates of $\theta^*$. By combining these two sources, GLIDE can achieve the same statistical precision as a purely human-labeled approach — at a fraction of the annotation cost. Actual savings depend on the annotation effort required and how well the proxy aligns with human judgement, but the potential gains can be substantial. This makes rigorous performance evaluation tractable even for large-scale AI systems.

The workflow has three stages:

```
Large proxy-labelled dataset
         │
         ▼
  [1] Sampler — which items to annotate?
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

## Stage 1 — Sampling: which items to annotate?

You start with a fully proxy-labelled dataset. The sampler's job is to select $b$ items for human annotation and to assign the sampling probability $\pi_i$ to each item. These probabilities are needed by the downstream estimator to correct for non-uniform selection.

### Do you need a sampler?

If you already have human labels for a uniformly drawn random subset of the data — for example, from a pre-existing annotation campaign — you can skip the sampling stage entirely and go straight to the estimator.

A sampler is useful when you still need to **allocate an annotation budget** and want to do so efficiently.

### Uncertainty scores

GLIDE's samplers determine per-item sampling probabilities from **uncertainty scores** — a scalar value you supply for each item that quantifies how unreliable the proxy label is. A higher score means the proxy is less trustworthy for that record. Common choices include model confidence, ensemble disagreement, or any domain-specific signal that reflects label reliability.

Uncertainty scores do not need to be calibrated — only their relative ordering matters for sampling.

### Choosing a sampler

GLIDE provides two samplers. The choice depends on the structure of your data:

| Situation | Recommended sampler |
|---|---|
| No natural grouping in the data; you want to focus on uncertain predictions | [`ActiveSampler`](samplers.md#active-sampler) |
| Data naturally splits into groups (language, domain, topic, …) | [`StratifiedSampler`](samplers.md#stratified-sampler) |

See [Samplers](samplers.md) for a full description of each option.

Samplers compute two values per item that the downstream estimator uses directly:

| Value | Description |
|---|---|
| $\pi_i$ | Drawing probability used to select item $i$ |
| $\xi_i$ | Indicator: $1$ if selected for annotation, $0$ otherwise |

Items with $\xi_i = 1$ are the ones to send for human annotation.

---

## Stage 2 — Estimation: what is the metric?

Once human labels $Y_i$ have been collected for items with $\xi_i = 1$, the estimator combines them with the proxy labels to produce an unbiased mean estimate and a confidence interval.

### Choosing an estimator

The right estimator depends on how the labeled subset was drawn:

| How were labels collected? | Recommended estimator |
|---|---|
| Uniform random sample (no sampler used, or `StratifiedSampler` with proportional allocation) | [PPI++](estimators.md#prediction-powered-inference-ppi) or [Stratified PPI++](estimators.md#stratified-ppi) |
| Non-uniform sampling via `ActiveSampler` or `StratifiedSampler` with Neyman allocation | [ASI](estimators.md#active-statistical-inference-asi) |
| No proxy labels available; human labels only | [Classical](estimators.md) |

See [Estimators](estimators.md) for a full description of each option.

## Full workflow example

Here is a concrete end-to-end sequence for a scenario where the data has natural strata (e.g., languages) and you want to allocate an annotation budget of $b = 200$:

1. **Proxy-label all $N$ items** with your automated judge.
2. **Run `StratifiedSampler`** with Neyman allocation and $b = 200$. This sets $\pi_i$ and $\xi_i$ for every item.
3. **Collect human annotations** $Y_i$ for each item where $\xi_i = 1$.
4. **Run `ASI`** on the full dataset (where each item has values $\tilde{Y}_i$, $\pi_i$, $\xi_i$, and $Y_i$ for labeled items). Power-tuning is on by default.
5. **Collect** the point estimate $\hat{\theta}$ and the confidence interval $[\hat{\theta} - \delta, \hat{\theta} + \delta]$ for a desired confidence level.

If instead labels already exist for a uniform random subset, skip steps 2–3 and run Stratified PPI++ directly.
