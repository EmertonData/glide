# Performance Estimation Guide

This guide explains the statistical problem that GLIDE solves and how it works under the hood.

## The problem

Suppose you have an AI system that produces answers $X$ over a large dataset of $N$ items, and you want to measure its performance — for example, its accuracy, relevance score, or any other metric $\theta$.

The challenge is that computing the **true** metric $\theta^*$ requires reliable annotations $Y$ for every item. Human annotations are reliable, but expensive. So in practice, you only have human labels for a small subset of $n$ items.

A natural shortcut is to use an **LLM-as-Judge** to label all $N$ items cheaply. The problem: LLM judges are **biased** — their labels $\tilde{Y}$ satisfy $E[\tilde{Y}] \neq \theta^*$. Naively averaging them gives a systematically wrong estimate of $\theta^*$.

GLIDE addresses this using estimators that combine cheap LLM labels with a small set of human labels to produce unbiased, reliable estimates of $\theta^*$.

---

## What a good estimator looks like

A good estimator $\hat{\theta}$ of $\theta^*$ must satisfy two criteria:

**No bias** — the estimate should be correct in expectation:

$$E[\hat{\theta}] = \theta^*$$

**Small uncertainty** — the true value $\theta^*$ should fall within a confidence interval $C_\alpha$ at risk level $\alpha$:

$$\Pr(\theta^* \in C_\alpha) \geq 1 - \alpha$$

---

## Input data

| | LLM-as-Judge labels | Human labels |
|---|---|---|
| **Size** | $N$ (large) | $n$ (small) |
| **Cost** | Cheap | Costly |
| **Notation** | $\tilde{Y}_i$ | $Y_j$ |
| **Bias** | Biased: $E[\tilde{Y}] \neq \theta^*$ | Unbiased: $E[Y] = \theta^*$ |

The key insight: even though human labels are scarce, they can be used to **correct** the bias in the cheap LLM labels.

<p align="center">
  <img src="../../assets/schema-PPI.png" alt="PPI data schema" width="400">
</p>

<p align="center">
  <em>All $N$ items are evaluated by the LLM judge. A smaller subset of $n$ items receives human annotations as well, enabling bias measurement and correction.</em>
</p>

---

## The Prediction-Powered Inference (PPI) estimator

PPI ([Angelopoulos et al., *Science* 2023](https://www.science.org/doi/10.1126/science.adi6000)) combines both sources of data into a single estimate:

$$\hat{\theta} = \underbrace{\frac{1}{N} \sum_{i=1}^{N} \tilde{Y}_i}_{\text{Biased estimate}} + \underbrace{\frac{1}{n} \sum_{j=1}^{n} \left(Y_j - \tilde{Y}_j\right)}_{\text{Bias rectifier}}$$

- The **biased estimate** uses all $N$ LLM labels to get a low-variance but biased estimate.
- The **bias rectifier** uses the $n$ paired samples (items that have both a human label and an LLM label) to measure and subtract the average bias.

The result is an estimator that is **unbiased** ($E[\hat{\theta}] = \theta^*$) and leverages the full dataset for precision.

---

## Variance and confidence intervals

For large enough sample sizes (typically $n \geq 100$), the **Central Limit Theorem** applies and the variance of the PPI estimator decomposes cleanly:

$$\sigma^2_{\hat{\theta}} = \underbrace{\frac{\sigma^2_{\tilde{Y}}}{N}}_{\text{LLM-as-Judge variance}} + \underbrace{\frac{\sigma^2_{Y - \tilde{Y}}}{n}}_{\text{Residual variance}}$$

- The first term shrinks as $N$ grows — but since $N$ is typically large, this term is usually negligible in practice.
- The second term dominates and shrinks both as $n$ grows and as the LLM judge becomes more aligned with human annotations (smaller numerator $\sigma^2_{Y - \tilde{Y}}$).

This gives a confidence interval at level $1 - \alpha$:

$$\Pr\!\left(\theta^* \in \left[\hat{\theta} - z_{1-\alpha/2}\, \sigma_{\hat{\theta}},\; \hat{\theta} + z_{1-\alpha/2}\, \sigma_{\hat{\theta}}\right]\right) \geq 1 - \alpha$$

where $z_{1-\alpha/2}$ is the standard normal quantile (e.g. $z_{0.975} = 1.96$ for a 95% confidence interval).

---

## Why it matters

By combining a large pool of cheap LLM labels with a small set of human labels, PPI can achieve the same statistical precision as a purely human-labeled approach — at a fraction of the annotation cost. Actual savings depend on the annotation effort required and how well the LLM judge aligns with human judgement, but the potential gains can be substantial. This makes rigorous performance evaluation tractable even for large-scale AI systems.

---

## References

Angelopoulos, Anastasios N., Stephen Bates, Clara Fannjiang, Michael I. Jordan, and Tijana Zrnic. "Prediction-powered inference." *Science* 382, no. 6671 (2023): 669–674.
