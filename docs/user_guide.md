# User Guide

This guide explains the problems solved by GLIDE and the algorithms it implements for that.

## Performance and Risk Estimation

Suppose you have an AI system that produces answers $X$ over a large dataset of $N$ items, and you want to measure its performance — for example, its accuracy, relevance score, or any other metric $\theta$.

The challenge is that computing the **true** metric $\theta^*$ requires reliable annotations $Y$ for every item. Human annotations are reliable, but expensive. So in practice, you only have human labels for a small subset of $n$ items.

A natural shortcut is to use **proxy labels** — automated predictions (for example, from an **LLM-as-Judge**) — to label all $N$ items cheaply. The problem: proxy labels $\tilde{Y}$ are generally **biased** — so we have $E[\tilde{Y}] \neq \theta^*$. Naively averaging them gives a systematically wrong estimate of $\theta^*$.

GLIDE addresses this by combining large pools of cheap proxy labels (e.g., LLM-as-Judge) with small sets of human labels to produce unbiased, reliable estimates of $\theta^*$. By combining these two sources, GLIDE can achieve the same statistical precision as a purely human-labeled approach — at a fraction of the annotation cost. Actual savings depend on the annotation effort required and how well the proxy aligns with human judgement, but the potential gains can be substantial. This makes rigorous performance evaluation tractable even for large-scale AI systems.

---

## What a good estimator looks like

A good estimator $\hat{\theta}$ of $\theta^*$ must satisfy two criteria.

### Properties

**No bias** — the estimate should be correct in expectation:

$$E[\hat{\theta}] = \theta^*$$

**Small and statistically valid uncertainty** — the true value $\theta^*$ should fall within a confidence interval $C_\alpha$ at risk level $\alpha$:

$$\Pr(\theta^* \in C_\alpha) \geq 1 - \alpha$$

Moreover, $C_\alpha$ should be as small as possible.

### Input data

All estimators in GLIDE rely on two complementary sources of labels. Proxy labels $\tilde{Y}_i$ are available for all $N$ items at low cost but are biased ($E[\tilde{Y}] \neq \theta^*$). Human labels $Y_j$ are unbiased ($E[Y] = \theta^*$) but expensive, and only available for a small labeled subset of $n \ll N$ items. The key insight: even though human labels are scarce, they can be used to **correct** the bias in the cheap proxy labels.

<p align="center">
  <img src="../assets/schema-PPI.png" alt="Data schema" width="550">
</p>

<p align="center">
  <em>All $N$ items are evaluated by a proxy (automated prediction or model). A smaller subset of $n$ items receives human annotations as well, enabling bias measurement and correction.</em>
</p>

---

## Prediction-Powered Inference (PPI++)

PPI assumes that the labeled subset is drawn **uniformly at random** from the population. Under this assumption, it constructs an unbiased estimator by combining all available proxy labels with a small set of ground-truth annotations, correcting for the bias of the proxy at minimal cost.

In PPI, every record carries two values:

| Field | Present for | Description |
|---|---|---|
| $\tilde{Y}_i$ | All $N$ records | Proxy label |
| $Y_j$ | Labeled records only ($n < N$) | Ground-truth label |

### Mean estimation

**PPI++** [[2](#ref-2)] is an extension of the original PPI [[1](#ref-1)] that introduces a weight $\lambda \in [0, 1]$ on the proxy labels. The mean estimate is:

$$\hat{\theta}_{\lambda} = \frac{1}{n} \sum_{j=1}^{n} Y_j + \lambda \left[\frac{1}{N} \sum_{i=1}^{N} \tilde{Y}_i - \frac{1}{n} \sum_{j=1}^{n} \tilde{Y}_j\right]$$

This combines two components:

- The **human-label mean** $\frac{1}{n}\sum_{j} Y_j$, which is unbiased but high-variance due to the small labeled set.
- A **bias correction term** that uses all $N$ proxy labels to reduce variance, scaled by $\lambda$ to control how much weight the proxy receives.

At $\lambda = 1$, this recovers the original PPI estimator, which can equivalently be written as:

$$\hat{\theta} = \underbrace{\frac{1}{N} \sum_{i=1}^{N} \tilde{Y}_i}_{\text{Biased estimate}} + \underbrace{\frac{1}{n} \sum_{j=1}^{n} \left(Y_j - \tilde{Y}_j\right)}_{\text{Bias rectifier}}$$

The parameter $\lambda$ allows modulating the contribution of the proxy labels based on how informative they are. We will see that it can be set to an optimal value below.


### Variance and confidence intervals

For large enough sample sizes (typically $n \geq 100$), the **Central Limit Theorem** applies and the variance of the PPI++ estimator decomposes as:

$$\sigma^2_{\hat{\theta}}(\lambda) = \underbrace{\frac{\sigma^2_{Y - \lambda\tilde{Y}}}{n}}_{\text{Labeled residual variance}} + \underbrace{\frac{\lambda^2\,\sigma^2_{\tilde{Y}}}{N}}_{\text{Unlabeled proxy variance}}$$

- The first term shrinks both as $n$ grows and as the proxy aligns better with human annotations.
- The second term shrinks as $N$ grows and is usually negligible in practice since $N \gg n$.

This gives a confidence interval at level $1 - \alpha$:

$$\Pr\!\left(\theta^* \in \left[\hat{\theta}_{\lambda} - z_{1-\alpha/2}\, \sigma_{\hat{\theta}}(\lambda),\; \hat{\theta}_{\lambda} + z_{1-\alpha/2}\, \sigma_{\hat{\theta}}(\lambda)\right]\right) \geq 1 - \alpha$$

where $z_{1-\alpha/2}$ is the standard normal quantile (e.g. $z_{0.975} = 1.96$ for a 95% two-sided confidence interval).

### Power-tuning

The $\lambda$ parameter needs to be chosen wisely. If left at $\lambda = 1,$ low-quality proxy labels with weak or negative covariance to human labels could *degrade* the estimation by inducing larger confidence intervals compared to using human labels only ($\lambda = 0$). PPI++ derives a closed-form plug-in estimator for the $\lambda$ that minimises the CI width:

$$\hat{\lambda} = \frac{\widehat{\text{Cov}}_n(Y,\, \tilde{Y})}{\left(1 + \tfrac{n}{N}\right)\widehat{\text{Var}}_{n+N}(\tilde{Y})}$$

where:

- $\widehat{\text{Cov}}_n$ is the sample covariance computed on the **$n$ labeled samples only**,

- $\widehat{\text{Var}}_{n+N}$ is the sample variance computed on **all $n + N$ proxy values pooled**,

- $n$ and $N$ are the numbers of labeled and unlabeled items respectively.

When the proxy is informative (high covariance with human labels), $\hat{\lambda}$ is close to 1 and the CI is narrower than standard PPI; when the proxy is uninformative, $\hat{\lambda}$ shrinks toward 0, down-weighting it and falling back to the classical human-only mean estimate. PPI++ uses optimal $\hat{\lambda}$ in GLIDE by default. This ensures the resulting estimate always has smaller variance than the classical estimate.

---

## Stratified PPI++

Standard PPI++ assumes that labeled and unlabeled samples are drawn uniformly from a single population. In practice, the dataset is often naturally partitioned into **strata** — for example, by language, domain, or question type — and the proxy model may behave very differently across these groups. **Stratified PPI++** [[5](#ref-5), [6](#ref-6)] exploits this structure: rather than applying one global estimate, it runs PPI++ independently within each stratum and combines the results with population-proportional weights. Whenever strata differ in proxy quality or size, this yields narrower confidence intervals than applying PPI++ to the full dataset at once.

Let $K$ denote the number of strata. Stratum $k$ contains $N_k$ total records (labeled + unlabeled), with $n_k$ labeled and $N_k^u = N_k - n_k$ unlabeled. The total dataset size is $N = \sum_k N_k$ and the **population weight** of stratum $k$ is:

$$w_k = \frac{N_k}{N}$$

In Stratified PPI++, every record carries the same fields as PPI++ — a proxy label $\tilde{Y}_i$ and optionally a ground-truth label $Y_j$ — plus a **group identifier** indicating which stratum the record belongs to.

### Mean estimation

The Stratified PPI++ point estimate is a weighted average of the per-stratum PPI++ estimates:

$$\hat{\theta}_{\text{strat}} = \sum_{k=1}^{K} w_k \cdot \hat{\theta}_k(\lambda_k)$$

where $\hat{\theta}_k(\lambda_k)$ is exactly the PPI++ mean estimator applied to the data in stratum $k$ with its own weight $\lambda_k$. The weights $w_k$ are proportional to stratum size, so larger strata contribute more to the final estimate. Since each $\hat{\theta}_k(\lambda_k)$ is an unbiased estimator for the stratum-$k$ mean and the weights sum to one, $\hat{\theta}_{\text{strat}}$ is an unbiased estimator for the population mean $\theta^*$.

### Variance and confidence intervals

The asymptotic variance of $\hat{\theta}_{\text{strat}}$ is the sum of the per-stratum PPI++ variances, each scaled by its squared population weight:

$$\sigma^2_{\text{strat}} = \sum_{k=1}^{K} w_k^2 \cdot \sigma^2_k(\lambda_k)$$

where $\sigma^2_k(\lambda_k)$ is the PPI++ variance for stratum $k$. When strata are internally homogeneous but differ from one another, the stratified computation yields a smaller variance than applying PPI++ globally, leading to narrower confidence intervals. The reported standard deviation is $\sigma_{\text{strat}} = \sqrt{\sigma^2_{\text{strat}}}$, and a confidence interval at level $1 - \alpha$ is constructed via the CLT exactly as in PPI++.

The key benefit over global PPI++ becomes apparent when strata differ substantially in proxy quality. Strata where the proxy is accurate contribute a small $\sigma^2_k(\lambda_k)$, while strata where it is poor contribute a larger one — but each contribution is isolated to its own stratum instead of polluting the global estimate.

### Power-tuning

Each stratum $k$ receives its **own optimal weight** $\hat{\lambda}_k$, computed with the same closed-form formula as PPI++, restricted to the $n_k$ labeled and $N_k^u$ unlabeled records within that stratum:

$$\hat{\lambda}_k = \frac{\widehat{\text{Cov}}_{n_k}(Y_k,\, \tilde{Y}_k)}{\left(1 + \tfrac{n_k}{N_k^u}\right)\widehat{\text{Var}}_{n_k + N_k^u}(\tilde{Y}_k)}$$

This is the same formula as PPI++ power-tuning, applied stratum by stratum. In strata where the proxy is informative, $\hat{\lambda}_k$ is close to 1 and the stratum estimate benefits from the proxy signal. In strata where the proxy is weak or unreliable, $\hat{\lambda}_k$ shrinks toward 0, falling back to the classical human-only mean for that stratum — without affecting any other stratum.

Setting `power_tuning=False` forces $\lambda_k = 1.0$ for all strata, recovering stratified PPI without the variance-minimising tuning.

---

## Active Statistical Inference (ASI)

Standard approaches to combining proxy and human labels assume that the labeled subset is drawn **uniformly at random** from the population. In practice, annotation resources are often allocated strategically — for instance, prioritizing uncertain or difficult examples. **Active Statistical Inference (ASI)** [[3](#ref-3), [4](#ref-4)] handles this general case: each sample $X_i$ may have a distinct, pre-determined probability $\pi_i \in (0, 1]$ of being selected for human annotation. Inverse-Probability Weighting (IPW) corrects for this non-uniform selection, yielding valid confidence intervals under any fixed sampling rule.

In ASI, every record carries three values:

| Field | Present for | Description |
|---|---|---|
| $\tilde{Y}_i$ | All $n$ records | Proxy label |
| $\pi_i$ | All $n$ records | Known, pre-determined sampling probability |
| $\xi_i$ | All $n$ records | Sampling indicator such that $\Pr(\xi_i = 1) = \pi_i = 1 - \Pr(\xi_i = 0)$ |
| $Y_i$ | Labeled records only ($\xi_i = 1$) | Ground-truth label |

We define $\xi_i \in \{0, 1\}$ as the **sampling indicator**: $\xi_i = 1$ if a ground-truth label is present for record $i$, and $\xi_i = 0$ otherwise. Crucially, $\pi_i$ must be known for every record. It is a property of the sampling design, not derived from the data.

### Mean estimation

The core of ASI is a per-record **IPW-corrected effective label**:

$$z_i(\lambda) = \lambda\,\tilde{Y}_i + \xi_i\,\frac{Y_i - \lambda\,\tilde{Y}_i}{\pi_i}$$

Expanding by case:

- **Unlabeled** ($\xi_i = 0$): $\quad z_i = \lambda\,\tilde{Y}_i$
- **Labeled** ($\xi_i = 1$): $\quad z_i = \lambda\,\tilde{Y}_i + \dfrac{Y_i - \lambda\,\tilde{Y}_i}{\pi_i}$

For labeled samples, the residual $Y_i - \lambda\,\tilde{Y}_i$ is divided by $\pi_i$. This **up-weights** records that were less likely to be selected, ensuring each labeled sample represents its fair share of the population. The parameter $\lambda$ modulates how much weight the proxy label receives.


The ASI mean estimator is simply the average of the IPW-corrected labels:

$$\hat{\theta}_{\lambda} = \frac{1}{n}\sum_{i=1}^{n} z_i(\lambda)$$

This estimator is **unbiased** for the population mean under any fixed sampling design, provided $\pi_i > 0$ for all records.

At $\lambda = 0$, this reduces to the classical Horvitz–Thompson estimator, which uses only the labeled samples (each weighted by $1/\pi_i$). As $\lambda$ increases, the proxy labels contribute progressively more to the estimate.


### Variance and confidence intervals

The asymptotic variance is the sample variance of the corrected labels divided by $n$:

$$\hat{\sigma}^2_{\text{SE}}(\lambda) = \frac{\widehat{\text{Var}}\!\left(z(\lambda)\right)}{n}$$

where $\widehat{\text{Var}}$ denotes the sample variance with $\text{ddof} = 1$. By the Central Limit Theorem (for $n$ large enough, typically $n \geq 100$), this yields a confidence interval at level $1 - \alpha$:

$$\Pr\!\left(\theta^* \in \left[\hat{\theta}_{\lambda} - z_{1-\alpha/2}\,\hat{\sigma}_{\text{SE}},\; \hat{\theta}_{\lambda} + z_{1-\alpha/2}\,\hat{\sigma}_{\text{SE}}\right]\right) \geq 1 - \alpha$$

where $z_{1-\alpha/2}$ is the standard normal quantile (e.g. $z_{0.975} = 1.96$ for a 95% two-sided confidence interval).


### Power-tuning

The choice of $\lambda$ directly controls the width of the confidence interval. A poor value can increase variance relative to a human-only estimate. ASI derives a closed-form optimal $\lambda$ by minimising $\hat{\sigma}^2_{\text{SE}}(\lambda)$ analytically.

Define two per-record quantities:

$$a_i = \tilde{Y}_i\!\left(\frac{\xi_i}{\pi_i} - 1\right), \qquad b_i = Y_i \cdot \frac{\xi_i}{\pi_i}$$

- $a_i$ is computable for every record (requires only $\tilde{Y}_i$, $\xi_i$, and $\pi_i$).
- $b_i$ equals $Y_i / \pi_i$ for labeled records and $0$ for unlabeled records.

The variance-minimising $\lambda$ is:

$$\hat{\lambda} = \frac{\widehat{\text{Cov}}(a,\, b)}{\widehat{\text{Var}}(a)}$$

When the proxy is informative, $\hat{\lambda}$ is large and the IPW-corrected labels benefit from the proxy signal, narrowing the confidence interval. When the proxy is uninformative, $\hat{\lambda}$ shrinks toward 0, down-weighting it. Power tuning is enabled by default in GLIDE (`power_tuning=True`). Setting `power_tuning=False` fixes $\lambda = 1$, recovering the plain IPW estimator.

---

## References

<a id="ref-1"></a>[1] <a id="ref-1-link" href="https://www.science.org/doi/10.1126/science.adi6000">Angelopoulos, Anastasios N., Stephen Bates, Clara Fannjiang, Michael I. Jordan, and Tijana Zrnic. "Prediction-powered inference." *Science* 382, no. 6671 (2023): 669–674</a>.

<a id="ref-2"></a>[2] <a id="ref-1-link" href="https://arxiv.org/abs/2311.01453">Angelopoulos, Anastasios N., John C. Duchi, and Tijana Zrnic. "PPI++: Efficient prediction-powered inference." *arXiv preprint arXiv:2311.01453* (2023)</a>.

<a id="ref-3"></a>[3] <a id="ref-1-link" href="https://arxiv.org/abs/2403.03208">Zrnic, Tijana, and Emmanuel Candès. "Active statistical inference." *arXiv preprint arXiv:2403.03208* (2024)</a>.

<a id="ref-4"></a>[4] <a id="ref-1-link" href="https://aclanthology.org/2025.naacl-long.179.pdf">Gligorić, Kristina, Tijana Zrnic, Cinoo Lee, Emmanuel Candes, and Dan Jurafsky. "Can unconfident llm annotations be used for confident conclusions?." NAACL 2025: Human Language Technologies (Volume 1: Long Papers), pp. 3514-3533. 2025.</a>.

<a id="ref-5"></a>[5] <a id="ref-5-link" href="https://arxiv.org/abs/2406.04291">Fisch, Adam, Joshua Maynez, R. Hofer, Bhuwan Dhingra, Amir Globerson, and William W. Cohen. "Stratified prediction-powered inference for effective hybrid evaluation of language models." *Advances in Neural Information Processing Systems* 37 (2024): 111489–111514.</a>.

<a id="ref-6"></a>[6] <a id="ref-6-link" href="https://arxiv.org/abs/2406.07320">Fogliato, Riccardo, Pratik Patil, Mathew Monfort, and Pietro Perona. "A framework for efficient model evaluation through stratification, sampling, and estimation." *European Conference on Computer Vision*, pp. 140–158. Springer Nature Switzerland, 2024.</a>.
