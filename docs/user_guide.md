# User Guide

This guide explains the problems solved by GLIDE and the algorithms it implements for that.

## Performance Estimation

Suppose you have an AI system that produces answers $X$ over a large dataset of $N$ items, and you want to measure its performance — for example, its accuracy, relevance score, or any other metric $\theta$.

The challenge is that computing the **true** metric $\theta^*$ requires reliable annotations $Y$ for every item. Human annotations are reliable, but expensive. So in practice, you only have human labels for a small subset of $n$ items.

A natural shortcut is to use an **LLM-as-Judge** to label all $N$ items cheaply. The problem: LLM judges are **biased** — their labels $\tilde{Y}$ satisfy $E[\tilde{Y}] \neq \theta^*$. Naively averaging them gives a systematically wrong estimate of $\theta^*$.

GLIDE addresses this using estimators that combine cheap LLM labels with a small set of human labels to produce unbiased, reliable estimates of $\theta^*$.


---

## Why it matters

By combining a large pool of cheap LLM labels with a small set of human labels, PPI can achieve the same statistical precision as a purely human-labeled approach — at a fraction of the annotation cost. Actual savings depend on the annotation effort required and how well the LLM judge aligns with human judgement, but the potential gains can be substantial. This makes rigorous performance evaluation tractable even for large-scale AI systems.


---

## What a good estimator looks like

A good estimator $\hat{\theta}$ of $\theta^*$ must satisfy two criteria:

**No bias** — the estimate should be correct in expectation:

$$E[\hat{\theta}] = \theta^*$$

**Small and statistically valid uncertainty** — the true value $\theta^*$ should fall within a confidence interval $C_\alpha$ at risk level $\alpha$:

$$\Pr(\theta^* \in C_\alpha) \geq 1 - \alpha$$

Moreover, $C_\alpha$ should be as small as possible

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

## The Prediction-Powered Inference (PPI++) estimator

PPI ([Angelopoulos et al., *Science* 2023](https://www.science.org/doi/10.1126/science.adi6000)) combines both sources of data into a single unbiased estimate:

$$\hat{\theta} = \underbrace{\frac{1}{N} \sum_{i=1}^{N} \tilde{Y}_i}_{\text{Biased estimate}} + \underbrace{\frac{1}{n} \sum_{j=1}^{n} \left(Y_j - \tilde{Y}_j\right)}_{\text{Bias rectifier}}$$

- The **biased estimate** uses all $N$ LLM labels to get a low-variance but biased estimate.
- The **bias rectifier** uses the $n$ paired samples (items that have both a human label and an LLM label) to measure and subtract the average bias.

This method was subsequently extended to **PPI++** ([Angelopoulos et al., 2023](https://arxiv.org/abs/2311.01453)), which introduces power tuning through a weight $\lambda \in [0, 1]$ on the proxy labels:

$$\hat{\theta}_{\lambda} = \frac{1}{n} \sum_{j=1}^{n} Y_j + \lambda \left[\frac{1}{N} \sum_{i=1}^{N} \tilde{Y}_i - \frac{1}{n} \sum_{j=1}^{n} \tilde{Y}_j\right]$$

At $\lambda = 1$ this reduces exactly to the original PPI estimator.

---

## Variance and confidence intervals

For large enough sample sizes (typically $n \geq 100$), the **Central Limit Theorem** applies and the variance of the PPI++ estimator decomposes as:

$$\sigma^2_{\hat{\theta}}(\lambda) = \underbrace{\frac{\sigma^2_{Y - \lambda\tilde{Y}}}{n}}_{\text{Labeled residual variance}} + \underbrace{\frac{\lambda^2\,\sigma^2_{\tilde{Y}}}{N}}_{\text{Unlabeled proxy variance}}$$

- The first term shrinks both as $n$ grows and as the LLM judge aligns better with human annotations.
- The second term shrinks as $N$ grows and is usually negligible in practice since $N \gg n$.

This gives a confidence interval at level $1 - \alpha$:

$$\Pr\!\left(\theta^* \in \left[\hat{\theta}_{\lambda} - z_{1-\alpha/2}\, \sigma_{\hat{\theta}}(\lambda),\; \hat{\theta}_{\lambda} + z_{1-\alpha/2}\, \sigma_{\hat{\theta}}(\lambda)\right]\right) \geq 1 - \alpha$$

where $z_{1-\alpha/2}$ is the standard normal quantile (e.g. $z_{0.975} = 1.96$ for a 95% confidence interval).

### Optimal $\lambda$

PPI++ derives a closed-form plug-in estimator for the $\lambda$ that minimises the CI width:

$$\hat{\lambda} = \frac{\widehat{\text{Cov}}_n(Y,\, \tilde{Y})}{\left(1 + \tfrac{n}{N}\right)\widehat{\text{Var}}_{n+N}(\tilde{Y})}$$

where:

- $\widehat{\text{Cov}}_n$ is the sample covariance computed on the **$n$ labeled samples only**,

- $\widehat{\text{Var}}_{n+N}$ is the sample variance computed on **all $n + N$ proxy values pooled**,

- $n$ and $N$ are the numbers of labeled and unlabeled items respectively.

When the proxy is informative (high covariance with human labels), $\hat{\lambda}$ is close to 1 and the CI is narrower than standard PPI; when the proxy is uninformative, $\hat{\lambda}$ shrinks toward 0, down-weighting it and falling back to the classical human-only mean estimate. PPI++ uses optimal $\hat{\lambda}$ in GLIDE by default. This ensures the resulting estimate always has smaller variance than the classical estimate.


# Active Statistical Inference (ASI)

> **Note:** This section is a work in progress.


---

## References

Angelopoulos, Anastasios N., Stephen Bates, Clara Fannjiang, Michael I. Jordan, and Tijana Zrnic. "Prediction-powered inference." *Science* 382, no. 6671 (2023): 669–674.

Angelopoulos, Anastasios N., John C. Duchi, and Tijana Zrnic. "PPI++: Efficient prediction-powered inference." *arXiv preprint arXiv:2311.01453* (2023).
