# Estimators

GLIDE provides three estimators that each combine proxy labels and a small human-annotated subset to produce an unbiased mean estimate and a confidence interval. The right choice depends on how the labeled subset was collected (see [Evaluation Workflow](evaluation_workflow.md) for guidance).

---

## What a good estimator looks like

A good estimator $\hat{\theta}$ of $\theta^*$ must satisfy two criteria.

### Properties

**No bias**: the estimate should be correct in expectation:

$$E[\hat{\theta}] = \theta^*$$

**Small and statistically valid uncertainty**: the true value $\theta^*$ should fall within a confidence interval $C_\alpha$ at risk level $\alpha$:

$$\Pr(\theta^* \in C_\alpha) \geq 1 - \alpha$$

Moreover, $C_\alpha$ should be as small as possible.

### Input data

All estimators in GLIDE rely on two complementary sources of labels. Proxy labels $\tilde{Y}_i$ are available for $N$ samples at low cost but are biased ($E[\tilde{Y}] \neq \theta^*$). Human labels $Y_j$ are unbiased ($E[Y] = \theta^*$) but expensive, and only available for a small labeled set of $n \ll N$ samples. The key insight: even though human labels are scarce, they can be used to **correct** the bias in the cheap proxy labels.

<p align="center">
  <img src="../../assets/schema-PPI.png" alt="Data schema" width="550">
</p>

<p align="center">
  <em>All $N$ samples are evaluated by a proxy (automated prediction or model). A smaller subset of $n$ samples receives human annotations as well, enabling bias measurement and correction.</em>
</p>

---

## Prediction-Powered Inference (PPI++)

PPI assumes that the labeled subset is drawn **uniformly at random** from the population. Under this assumption, it constructs an unbiased estimator by combining all available proxy labels with a small set of ground-truth annotations, correcting for the bias of the proxy at minimal cost.

In PPI, each sample has two associated values:

| Value | Present for | Description |
|---|---|---|
| $\tilde{Y}_i$ | All $n+N$ samples | Proxy label |
| $Y_j$ | Labeled samples only ($n < N$) | Ground-truth label |

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

where $z_{1-\alpha/2}$ is the standard normal quantile (e.g. $z_{0.975} = 1.96$ for a $95\%$ two-sided confidence interval).

### Power-tuning

The $\lambda$ parameter needs to be chosen wisely. If left at $\lambda = 1,$ low-quality proxy labels with weak or negative covariance to human labels could *degrade* the estimation by inducing larger confidence intervals compared to using human labels only ($\lambda = 0$). PPI++ derives a closed-form plug-in estimator for the $\lambda$ that minimises the CI width:

$$\hat{\lambda} = \frac{\widehat{\text{Cov}}_n(Y,\, \tilde{Y})}{\left(1 + \tfrac{n}{N}\right)\widehat{\text{Var}}_{n+N}(\tilde{Y})}$$

where:

- $\widehat{\text{Cov}}_n$ is the sample covariance computed on the **$n$ labeled samples only**,

- $\widehat{\text{Var}}_{n+N}$ is the sample variance computed on **all $n + N$ proxy values pooled**,

- $n$ and $N$ are the numbers of labeled and unlabeled samples respectively.

When the proxy is informative (high covariance with human labels), $\hat{\lambda}$ is close to 1 and the CI is narrower than standard PPI; when the proxy is uninformative, $\hat{\lambda}$ shrinks toward 0, down-weighting it and falling back to the classical human-only mean estimate. It is standard to use optimal $\hat{\lambda}$. This ensures the resulting estimate always has smaller variance than the classical estimate.

---

## Stratified PPI++

Standard PPI++ assumes that labeled and unlabeled samples are drawn uniformly from a single population. In practice, the dataset is often naturally partitioned into **strata** (for example, by language, domain, or question type), and the proxy model may behave very differently across these strata. **Stratified PPI++** [[5](#ref-5), [6](#ref-6)] exploits this structure: rather than applying one global estimate, it runs PPI++ independently within each stratum and combines the results with population-proportional weights.

Let $K$ denote the number of strata. Stratum $k$ contains $n_k+N_k$ total samples (labeled + unlabeled), of which $n_k$ are labeled. We let $n = \sum_k n_k$ and $N = \sum_k N_k$ be the total numbers of labeled and unlabeled samples respectively. We assume that $n_k/n \approx N_k/N$ for all $k$ and compute the **population weight** of stratum $k$ as:

$$w_k = \frac{n_k+N_k}{n+N}$$

In Stratified PPI++, each sample has the same values as in PPI++ (a proxy label $\tilde{Y}_i$ and optionally a ground-truth label $Y_j$), plus a **stratum identifier** indicating which stratum the sample belongs to.

| Value | Present for | Description |
|---|---|---|
| $\tilde{Y}_i$ | All $n+N$ samples | Proxy label |
| $Y_j$ | Labeled samples only ($n < N$) | Ground-truth label |
| $g_j$ | All $n+N$ samples | Stratum identifier |


### Mean estimation

The Stratified PPI++ point estimate is a weighted average of the per-stratum PPI++ estimates:

$$\hat{\theta}_{\text{strat}} = \sum_{k=1}^{K} w_k \cdot \hat{\theta}_k(\lambda_k)$$

where $\hat{\theta}_k(\lambda_k)$ is exactly the PPI++ mean estimator applied to the data in stratum $k$ with its own weight $\lambda_k$. The weights $w_k$ are proportional to stratum size, so larger strata contribute more to the final estimate. Since each $\hat{\theta}_k(\lambda_k)$ is an unbiased estimator for the stratum-$k$ mean and the weights sum to one, $\hat{\theta}_{\text{strat}}$ is an unbiased estimator for the population mean $\theta^*$.

### Variance and confidence intervals

Keep in mind that Stratified PPI++ is designed for a small number of large strata. The theoretical guarantees assume that the number of strata $K$ stays fixed as sample size grows and that each stratum contains a non-vanishing share of the data. In practice, many small strata mean that per-stratum statistical estimates become unreliable, and the CLT approximation underlying the confidence interval may break down. When in doubt, prefer a coarser stratification with fewer, larger strata.

The asymptotic variance of $\hat{\theta}_{\text{strat}}$ is the sum of the per-stratum PPI++ variances, each scaled by its squared population weight:

$$\sigma^2_{\text{strat}} = \sum_{k=1}^{K} w_k^2 \cdot \sigma^2_k(\lambda_k)$$

where $\sigma^2_k(\lambda_k)$ is the PPI++ variance for stratum $k$. The reported standard deviation $\sigma_{\text{strat}}$ serves to construct a confidence interval at level $1 - \alpha$ via the CLT exactly as in PPI++.

The key benefit over global PPI++ becomes apparent when strata differ substantially in proxy quality. Strata where the proxy is accurate contribute a small $\sigma^2_k(\lambda_k)$, while strata where it is poor contribute a larger one, but each contribution is isolated to its own stratum instead of polluting the global estimate.

### Power-tuning

Each stratum $k$ receives its **own optimal weight** $\hat{\lambda}_k$, computed with the same closed-form formula as PPI++, restricted to the $n_k$ labeled and $N_k$ unlabeled samples within that stratum:

$$\hat{\lambda}_k = \frac{\widehat{\text{Cov}}_{n_k}(Y_k,\, \tilde{Y}_k)}{\left(1 + \tfrac{n_k}{N_k}\right)\widehat{\text{Var}}_{n_k + N_k}(\tilde{Y}_k)}$$

This is the same formula as PPI++ power-tuning, applied stratum by stratum. In strata where the proxy is informative, $\hat{\lambda}_k$ is close to 1 and the stratum estimate benefits from the proxy signal. In strata where the proxy is weak or unreliable, $\hat{\lambda}_k$ shrinks toward 0, falling back to the classical human-only mean for that stratum, without affecting any other stratum. It is standard to use optimal power tuning with the previous $\hat{\lambda}_k$ values.

---

## Active Statistical Inference (ASI)

Standard approaches to combining proxy and human labels assume that the labeled subset is drawn **uniformly at random** from the population. In practice, annotation resources are often allocated strategically, for instance, prioritizing uncertain or difficult examples. **Active Statistical Inference (ASI)** [[3](#ref-3), [4](#ref-4)] handles this general case: each sample $X_i$ may have a distinct, pre-determined probability $\pi_i \in (0, 1]$ of being selected for human annotation. Inverse-Probability Weighting (IPW) corrects for this non-uniform selection, yielding valid confidence intervals under any fixed sampling rule.

In this section, we assume we have a total of $n$ samples, labeled and unlabeled. In ASI, each sample has three associated values:

| Value | Present for | Description |
|---|---|---|
| $\tilde{Y}_i$ | All $n$ samples | Proxy label |
| $\pi_i$ | All $n$ samples | Known, pre-determined sampling probability |
| $\xi_i$ | All $n$ samples | Sampling indicator such that $\Pr(\xi_i = 1) = \pi_i = 1 - \Pr(\xi_i = 0)$ |
| $Y_i$ | Labeled samples only ($\xi_i = 1$) | Ground-truth label |

We define $\xi_i \in \{0, 1\}$ as the **sampling indicator**: $\xi_i = 1$ if a ground-truth label is present for sample $i$, and $\xi_i = 0$ otherwise. Crucially, $\pi_i$ must be known for every sample. It is a property of the sampling design, not derived from the data.

### Mean estimation

The core of ASI is a per-sample **IPW-corrected effective label**:

$$z_i(\lambda) = \lambda\,\tilde{Y}_i + \xi_i\,\frac{Y_i - \lambda\,\tilde{Y}_i}{\pi_i}$$

Expanding by case:

- **Unlabeled** ($\xi_i = 0$): $\quad z_i = \lambda\,\tilde{Y}_i$
- **Labeled** ($\xi_i = 1$): $\quad z_i = \lambda\,\tilde{Y}_i + \dfrac{Y_i - \lambda\,\tilde{Y}_i}{\pi_i}$

For labeled samples, the residual $Y_i - \lambda\,\tilde{Y}_i$ is divided by $\pi_i$. This **up-weights** samples that were less likely to be selected, ensuring each labeled sample represents its fair share of the population. The parameter $\lambda$ modulates how much weight the proxy label receives.


The ASI mean estimator is simply the average of the IPW-corrected labels:

$$\hat{\theta}_{\lambda} = \frac{1}{n}\sum_{i=1}^{n} z_i(\lambda)$$

This estimator is **unbiased** for the population mean under any fixed sampling design, provided $\pi_i > 0$ for all samples.

At $\lambda = 0$, this reduces to the classical Horvitz–Thompson estimator, which uses only the labeled samples (each weighted by $1/\pi_i$). As $\lambda$ increases, the proxy labels contribute progressively more to the estimate.


### Variance and confidence intervals

The asymptotic variance is the sample variance of the corrected labels divided by $n$:

$$\hat{\sigma}^2_{\text{SE}}(\lambda) = \frac{\widehat{\text{Var}}\!\left(z(\lambda)\right)}{n}$$

where $\widehat{\text{Var}}$ denotes the sample variance with $\text{ddof} = 1$. By the Central Limit Theorem (for $n$ large enough, typically $n \geq 100$), this yields a confidence interval at level $1 - \alpha$:

$$\Pr\!\left(\theta^* \in \left[\hat{\theta}_{\lambda} - z_{1-\alpha/2}\,\hat{\sigma}_{\text{SE}},\; \hat{\theta}_{\lambda} + z_{1-\alpha/2}\,\hat{\sigma}_{\text{SE}}\right]\right) \geq 1 - \alpha$$

where $z_{1-\alpha/2}$ is the standard normal quantile (e.g. $z_{0.975} = 1.96$ for a $95\%$ two-sided confidence interval).


### Power-tuning

The choice of $\lambda$ directly controls the width of the confidence interval. A poor value can increase variance relative to a human-only estimate. ASI derives a closed-form optimal $\lambda$ by minimising $\hat{\sigma}^2_{\text{SE}}(\lambda)$ analytically.

Define two per-sample quantities:

$$a_i = \tilde{Y}_i\!\left(\frac{\xi_i}{\pi_i} - 1\right), \qquad b_i = Y_i \cdot \frac{\xi_i}{\pi_i}$$

- $a_i$ is computable for every sample (requires only $\tilde{Y}_i$, $\xi_i$, and $\pi_i$).
- $b_i$ equals $Y_i / \pi_i$ for labeled samples and $0$ for unlabeled samples.

The variance-minimising $\lambda$ is:

$$\hat{\lambda} = \frac{\widehat{\text{Cov}}(a,\, b)}{\widehat{\text{Var}}(a)}$$

When the proxy is informative, $\hat{\lambda}$ is large and the IPW-corrected labels benefit from the proxy signal, narrowing the confidence interval. When the proxy is uninformative, $\hat{\lambda}$ shrinks toward 0, down-weighting it. Fixing $\lambda = 1$, recover the plain IPW estimator. It is standard to use optimal power tuning with the $\hat{\lambda}$ value above.

---

## Predict-Then-Debias (PTD)

**Predict-Then-Debias (PTD)** [[7](#ref-7)] constructs a confidence interval from the **empirical distribution of bootstrap estimates** rather than a normal approximation, making it reliable when $n$ is small or residuals are non-Gaussian. GLIDE implements Algorithm 3 from [[7](#ref-7)], which works on a uniformly drawn labeled sample and includes a speedup that avoids resampling the unlabeled data during the bootstrap.

In PTD, each sample has two associated values:

| Value | Present for | Description |
|---|---|---|
| $\tilde{Y}_i$ | All $n+N$ samples | Proxy label |
| $Y_j$ | Labeled samples only ($n \ll N$) | Ground-truth label |

Denote $(\tilde{Y}^\circ_i)_{i=1}^N$ the unlabeled proxies and $(\tilde{Y}^\bullet_i)_{i=1}^n$ the labeled ones.

### Mean estimation

The PTD mean estimate is the average of $B$ bootstrap estimates:

$$\hat{\theta}_{\text{PTD}} = \frac{1}{B}\sum_{b=1}^{B}\hat{\theta}^{(b)}_{\text{PTD}}$$

where each $\hat{\theta}^{(b)}_{\text{PTD}}$ is computed during the bootstrap procedure described below.

### Bootstrap procedure

For $b = 1, \dots, B$, sample a set of indices $\mathcal{I}^{(b)}$ of size $n$ uniformly with replacement from $\{1, \dots, n\}$ and compute the bootstrap means of the labeled ground-truth and proxy labels:

$$\hat{\mu}^{(b)}_{\text{true}} = \frac{1}{n}\sum_{i\in \mathcal{I}^{(b)}} Y_i, \qquad \hat{\mu}^{(b)}_{\text{proxy}} = \frac{1}{n}\sum_{i\in \mathcal{I}^{(b)}} \tilde{Y}^\bullet_i$$

The third ingredient needed is a perturbed draw of the unlabeled proxy mean, $\tilde{\gamma}^{(b)}$. Naively, this would require resampling all $N$ proxy labels on the unlabeled samples at each iteration. Algorithm 3 in [[7](#ref-7)] avoids this cost: by the CLT, the mean of $N$ i.i.d. proxy scores is approximately Gaussian with mean $\hat{\gamma}^\circ = \frac{1}{N}\sum_{i=1}^{N}\tilde{Y}^\circ_i$ and variance $\hat{S}_{\gamma}^\circ = \widehat{\text{Var}}(\tilde{Y}^\circ) / N$, so instead of resampling all $N$ unlabeled proxy scores at each iteration, we replace that expensive resample with a single standard gaussian draw, mimicking bootstrap randomness at a far lower computational cost:

$$\tilde{\gamma}^{(b)} = \hat{\gamma}^\circ + Z^{(b)} \cdot \sqrt{\hat{S}_{\gamma}^\circ}, \qquad Z^{(b)} \sim \mathcal{N}(0,\, 1)$$

The quantities $\hat{\gamma}^\circ$ and $\hat{S}_{\gamma}^\circ$ are computed once before the loop, reducing the per-iteration cost to $O(n)$ instead of $O(n+N)$. This approximation is reliable for large $N$ which is typically the case in practical scenarios where proxy labels are far cheaper than expensive human annotations.

Combining the labeled bootstrap means with this unlabeled draw gives:

$$\hat{\theta}^{(b)}_{\text{PTD}} = \lambda \cdot \tilde{\gamma}^{(b)} + \left(\hat{\mu}^{(b)}_{\text{true}} - \lambda \cdot \hat{\mu}^{(b)}_{\text{proxy}}\right)$$

where $\lambda$ is a power-tuning factor that controls the proxy labels' influence similarly to previous sections. The term $\hat{\mu}^{(b)}_{\text{true}} - \lambda \cdot \hat{\mu}^{(b)}_{\text{proxy}}$ captures the proxy bias measured on the labeled set, while $\lambda \cdot \tilde{\gamma}^{(b)}$ contributes the proxy signal on the full unlabeled population. Together they form a bias-corrected estimate of $\theta^*$ for each bootstrap replicate.

### Variance and confidence intervals

The variance of the PTD estimator is the sample variance of the bootstrap estimates:

$$\hat{\sigma}^2_{\text{PTD}} = \widehat{\text{Var}}_B\!\left(\hat{\theta}^{(1)}_{\text{PTD}},\, \ldots,\, \hat{\theta}^{(B)}_{\text{PTD}}\right)$$

where $\widehat{\text{Var}}_B$ is the sample variance computed across the $B$ bootstrap replicates.

The confidence interval at level $1 - \alpha$ is the interval between the $\alpha/2$ and $1 - \alpha/2$ empirical quantiles of $\bigl\{\hat{\theta}^{(1)}_{\text{PTD}},\, \ldots,\, \hat{\theta}^{(B)}_{\text{PTD}}\bigr\}$. This bootstrap percentile approach adapts to the actual shape of the residual distribution, making it reliable even when $n$ is small.

### Power-tuning

The optimal $\lambda$ is estimated from the **bootstrap covariances**. Let $\hat{\mu}_{\text{true}}$ and $\hat{\mu}_{\text{proxy}}$ be the vectors of values $\hat{\mu}^{(b)}_{\text{true}}$ and $\hat{\mu}^{(b)}_{\text{proxy}}$ for $b=1,\dots,B$ respectively. After running the bootstrap loop, it is computed as:

$$\hat{\lambda} = \frac{\widehat{\text{Cov}}_B\!\left(\hat{\mu}_{\text{true}},\; \hat{\mu}_{\text{proxy}}\right)}{\widehat{\text{Var}}_B\!\left(\hat{\mu}_{\text{proxy}}\right) + \hat{S}_{\gamma}^\circ}$$

where $\widehat{\text{Cov}}_B$ and $\widehat{\text{Var}}_B$ are computed across the $B$ bootstrap replicates of the labeled means, and $\hat{S}_{\gamma}^\circ$ is the estimated sampling variance of the unlabeled proxy mean. The denominator adds $\hat{S}_{\gamma}^\circ$ to account for the variance of the unlabeled proxies. This value can be readily used since a Gaussian approximation is made for the unlabeled mean.

When the proxy is informative (high bootstrap covariance with ground-truth means), $\hat{\lambda}$ is large and the estimate borrows heavily from the proxy signal, narrowing the interval. When the proxy is uninformative, $\hat{\lambda}$ shrinks toward 0, down-weighting it. Fixing $\lambda = 1$, recovers the unweighted PTD estimator. It is standard to use the optimal value $\hat{\lambda}$ in practice.

---

## Stratified PTD

**Stratified PTD** [[7](#ref-7)] extends PTD to datasets naturally partitioned into **strata** (for example, by language, domain, or data source). The PTD bootstrap is run independently within each stratum, each with its own tuning parameter, and the per-stratum results are combined with population-proportional weights into a single confidence interval. When strata differ in proxy quality, this yields narrower intervals than a single global PTD run.

GLIDE implements a stratified extension of Algorithm 3 from [[7](#ref-7)], applying the CLT speedup for the unlabeled mean independently within each stratum. This differs from Algorithm 6 of [[7](#ref-7)], which uses a single global power-tuning parameter. The per-stratum variant is statistically valid and tends to be more precise when strata differ in proxy quality.

Let $K$ denote the number of strata. Stratum $k$ contains $n_k + N_k$ total samples, of which $n_k$ are labeled and $N_k$ are unlabeled, with population weight:

$$w_k = \frac{n_k + N_k}{n + N}$$

where $n = \sum_k n_k$ and $N = \sum_k N_k$.

In Stratified PTD, each sample has the same values as in PTD, plus a stratum identifier:

| Value | Present for | Description |
|---|---|---|
| $\tilde{Y}_i$ | All $n+N$ samples | Proxy label |
| $Y_j$ | Labeled samples only ($n \ll N$) | Ground-truth label |
| $g_j$ | All $n+N$ samples | Stratum identifier |

### Mean estimation

The Stratified PTD point estimate is the mean of $B$ combined bootstrap estimates:

$$\hat{\theta}_{\text{SPTD}} = \frac{1}{B}\sum_{b=1}^{B}\hat{\theta}^{(b)}_{\text{SPTD}}$$

where each $\hat{\theta}^{(b)}_{\text{SPTD}}$ is produced during the bootstrap procedure described below. Since each per-stratum term is an unbiased estimate of the stratum-$k$ mean and the weights $w_k$ sum to one, $\hat{\theta}_{\text{SPTD}}$ is an unbiased estimator of the population mean $\theta^*$.

### Bootstrap procedure

Denote $(\tilde{Y}^\circ_{k,i})_{i=1}^{N_k}$ the unlabeled proxies in stratum $k$ and $(\tilde{Y}^\bullet_{k,j})_{j=1}^{n_k}$ the labeled ones. Before the bootstrap loop, compute for each stratum $k$ the mean and sampling variance of the unlabeled proxy scores:

$$\hat{\gamma}^\circ_k = \frac{1}{N_k}\sum_{i=1}^{N_k}\tilde{Y}^\circ_{k,i}, \qquad \hat{S}^\circ_{\gamma,k} = \frac{\widehat{\text{Var}}(\tilde{Y}^\circ_k)}{N_k}$$

These quantities are computed once and reused across all $B$ iterations, applying the same CLT speedup as PTD to each stratum independently.

For $b = 1, \dots, B$ and for each stratum $k$, sample $n_k$ indices $\mathcal{I}^{(b)}_k$ with replacement from $\{1, \dots, n_k\}$ and compute the bootstrap means of the labeled ground-truth and proxy labels:

$$\hat{\mu}^{(b)}_{\text{true},k} = \frac{1}{n_k}\sum_{i \in \mathcal{I}^{(b)}_k} Y_{k,i}, \qquad \hat{\mu}^{(b)}_{\text{proxy},k} = \frac{1}{n_k}\sum_{i \in \mathcal{I}^{(b)}_k} \tilde{Y}^\bullet_{k,i}$$

A perturbed draw of the unlabeled proxy mean for stratum $k$ is formed as:

$$\tilde{\gamma}^{(b)}_k = \hat{\gamma}^\circ_k + Z^{(b)}_k \cdot \sqrt{\hat{S}^\circ_{\gamma,k}}, \qquad Z^{(b)}_k \sim \mathcal{N}(0, 1)$$

where each $Z^{(b)}_k$ is drawn independently across strata and iterations. The per-stratum bootstrap estimates are then combined with population-proportional weights:

$$\hat{\theta}^{(b)}_{\text{SPTD}} = \sum_{k=1}^{K} w_k \left[\lambda_k \cdot \tilde{\gamma}^{(b)}_k + \left(\hat{\mu}^{(b)}_{\text{true},k} - \lambda_k \cdot \hat{\mu}^{(b)}_{\text{proxy},k}\right)\right]$$

The term $\hat{\mu}^{(b)}_{\text{true},k} - \lambda_k \cdot \hat{\mu}^{(b)}_{\text{proxy},k}$ captures the proxy bias in stratum $k$ on the labeled set, while $\lambda_k \cdot \tilde{\gamma}^{(b)}_k$ contributes the proxy signal on the stratum's unlabeled population.

### Variance and confidence intervals

The variance of the Stratified PTD estimator is the sample variance of the combined bootstrap estimates:

$$\hat{\sigma}^2_{\text{SPTD}} = \widehat{\text{Var}}_B\!\left(\hat{\theta}^{(1)}_{\text{SPTD}},\, \ldots,\, \hat{\theta}^{(B)}_{\text{SPTD}}\right)$$

The confidence interval at level $1 - \alpha$ is the interval between the $\alpha/2$ and $1 - \alpha/2$ empirical quantiles of $\bigl\{\hat{\theta}^{(b)}_{\text{SPTD}}\bigr\}_{b=1}^B$, inheriting the robustness to non-Gaussianity that characterises PTD.

Stratified PTD is designed for a small number of large strata. As the number of strata grows, each stratum's labeled set shrinks and the bootstrap distribution becomes unreliable. When in doubt, prefer a coarser stratification with fewer, larger strata.

### Power-tuning

Each stratum $k$ receives its own optimal tuning scalar $\hat{\lambda}_k$, estimated after the bootstrap loop from the bootstrap covariances within that stratum:

$$\hat{\lambda}_k = \frac{\widehat{\text{Cov}}_B\!\left(\hat{\mu}_{\text{true},k},\; \hat{\mu}_{\text{proxy},k}\right)}{\widehat{\text{Var}}_B\!\left(\hat{\mu}_{\text{proxy},k}\right) + \hat{S}^\circ_{\gamma,k}}$$

This is the same formula as PTD power-tuning, applied stratum by stratum. In strata where the proxy is informative, $\hat{\lambda}_k$ is close to 1 and the estimate benefits from the proxy signal. In strata where the proxy is weak, $\hat{\lambda}_k$ shrinks toward 0, falling back to the classical bootstrap mean for that stratum, without affecting any other stratum. It is standard to use optimal power tuning with the previous $\hat{\lambda}_k$ values.

---

## References

<a id="ref-1"></a>[1] <a id="ref-1-link" href="https://www.science.org/doi/10.1126/science.adi6000">Angelopoulos, Anastasios N., Stephen Bates, Clara Fannjiang, Michael I. Jordan, and Tijana Zrnic. "Prediction-powered inference." *Science* 382, no. 6671 (2023): 669–674</a>.

<a id="ref-2"></a>[2] <a id="ref-2-link" href="https://arxiv.org/abs/2311.01453">Angelopoulos, Anastasios N., John C. Duchi, and Tijana Zrnic. "PPI++: Efficient prediction-powered inference." *arXiv preprint arXiv:2311.01453* (2023)</a>.

<a id="ref-3"></a>[3] <a id="ref-3-link" href="https://proceedings.mlr.press/v235/zrnic24a.html">Zrnic, Tijana, and Emmanuel J. Candès. "Active statistical inference." Proceedings of the 41st International Conference on Machine Learning. 2024</a>.

<a id="ref-4"></a>[4] <a id="ref-4-link" href="https://aclanthology.org/2025.naacl-long.179.pdf">Gligorić, Kristina, Tijana Zrnic, Cinoo Lee, Emmanuel Candes, and Dan Jurafsky. "Can unconfident llm annotations be used for confident conclusions?." NAACL 2025: Human Language Technologies (Volume 1: Long Papers), pp. 3514-3533. 2025.</a>.

<a id="ref-5"></a>[5] <a id="ref-5-link" href="https://arxiv.org/abs/2406.04291">Fisch, Adam, Joshua Maynez, R. Hofer, Bhuwan Dhingra, Amir Globerson, and William W. Cohen. "Stratified prediction-powered inference for effective hybrid evaluation of language models." *Advances in Neural Information Processing Systems* 37 (2024): 111489–111514.</a>.

<a id="ref-6"></a>[6] <a id="ref-6-link" href="https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/12117.pdf">Fogliato, Riccardo, Pratik Patil, Mathew Monfort, and Pietro Perona. "A framework for efficient model evaluation through stratification, sampling, and estimation." *European Conference on Computer Vision*, pp. 140–158. Springer Nature Switzerland, 2024.</a>.

<a id="ref-7"></a>[7] <a id="ref-7-link" href="https://arxiv.org/abs/2501.18577">Kluger, Dan M., Kerri Lu, Tijana Zrnic, Sherrie Wang, and Stephen Bates. "Prediction-powered inference with imputed covariates and nonuniform sampling." *arXiv preprint arXiv:2501.18577* (2025).</a>.
