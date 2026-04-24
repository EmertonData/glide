# Samplers

Samplers sit **upstream of the estimators**: they decide *which* samples to send for human annotation and compute the sampling probabilities $\pi_i$ required by some estimators that rely on Inverse Probability Weighting (IPW). Choosing samples strategically (rather than uniformly at random) can substantially reduce the annotation budget needed to reach a target confidence-interval width.

A sampler takes a fully proxy-labelled dataset and a budget $b$, and computes two values per sample:

| Value | Description |
|---|---|
| $\pi_i$ | Drawing probability used to select sample $i$ for annotation ($0 < \pi_i \leq 1$) |
| $\xi_i$ | Bernoulli indicator: $1$ if the sample was selected, $0$ otherwise |

These output values can be leveraged by estimators supporting inverse probability weighting (IPW) to correct for non-uniform sampling bias.

---

## Stratified Sampler

When the dataset is naturally partitioned into **strata** (groups such as language, domain, or topic), the `StratifiedSampler` allocates the annotation budget across strata before performing per-stratum Bernoulli sampling. It supports two allocation strategies: **proportional** (baseline) and **Neyman** (default, variance-optimal).

Let $K$ denote the number of strata. Stratum $h$ contains $N_h$ samples. The total dataset size is $N = \sum_{h=1}^{K} N_h$ and the annotation budget is $b$.

### Proportional allocation

Proportional allocation assigns budget to each stratum in proportion to its size:

$$n_h^{\text{prop}} = b \cdot \frac{N_h}{N}$$

This yields the same sampling probability $\pi_i = b / N$ for every sample, regardless of stratum. It is equivalent to simple random sampling and serves as a baseline.

### Neyman allocation

Neyman allocation assigns more budget to strata with **higher proxy variance**, reducing the asymptotic variance of downstream estimators [[1](#ref-1)]. The raw allocation for stratum $h$ is:

$$n_h^{\text{ney}} = b \cdot \frac{N_h\, \hat{\sigma}_h}{\displaystyle\sum_{k=1}^{K} N_k\, \hat{\sigma}_k}$$

where $\hat{\sigma}_h$ is the sample standard deviation of the proxy labels within stratum $h$. Strata where the proxy is more dispersed receive a larger slice of the budget, because annotation effort there reduces total variance most efficiently.

When proxy variance is uniform across strata, Neyman allocation reduces to proportional allocation.

### Rounding

Both formulas produce fractional per-stratum counts. GLIDE resolves this with **largest-remainder rounding** (Hamilton's method): each stratum first receives $\lfloor n_h \rfloor$ slots, then the strata with the largest fractional remainders $n_h - \lfloor n_h \rfloor$ receive one extra slot each until the total reaches $b$. Per-stratum allocations are additionally capped at $N_h$, so the total allocated budget may be slightly less than $b$ when some strata are very small or have very high variance relatively to others.

### Sampling probabilities and procedure

Once the integer allocation $n_h$ is determined, every item in stratum $h$ receives the same drawing probability:

$$\pi_i = \min\!\left(\frac{n_h}{N_h},\; 1\right)$$

Each item is then independently selected via a Bernoulli trial:

$$\xi_i \sim \mathrm{Bernoulli}(\pi_i), \quad i = 1, \ldots, n$$

Each item receives values $(\pi_i, \xi_i)$. Samples with $\xi_i = 1$ are sent for human annotation.

---

## Active Sampler

The `ActiveSampler` concentrates the annotation budget on the samples with the **most uncertain proxy labels**. The intuition is that high-uncertainty samples are those where the proxy label is least reliable, so human annotation there yields the greatest reduction in variance.

### Sampling probabilities

Each sample $i$ has an **uncertainty score** $u_i > 0$. The raw drawing probability is set proportional to $u_i$, normalised so that the expected number of selected samples equals the budget $b$:

$$\tilde{\pi}_i = b \cdot \frac{u_i}{\sum_{j=1}^{n} u_j}$$

Because $\tilde{\pi}_i$ must be a valid Bernoulli probability, values are capped at 1:

$$\pi_i = \min\!\left(\tilde{\pi}_i,\; 1\right)$$

Samples whose raw probability exceeds 1 are selected with certainty ($\pi_i = 1$). As a result, the actual expected number of selected samples is at most $b$ (it may be slightly less when some values are capped).

### Sampling procedure

Given the probabilities $\pi_i$, each sample is independently selected via a Bernoulli trial:

$$\xi_i \sim \mathrm{Bernoulli}(\pi_i), \quad i = 1, \ldots, n$$

The draws are independent across samples. Each sample receives values $(\pi_i, \xi_i)$; samples with $\xi_i = 1$ are the ones to be sent for human annotation.

Note that uncertainty scores must be strictly positive. The `ActiveSampler` requires $u_i > 0$ for every sample. A zero or negative uncertainty score would result in a zero sampling probability, making that sample permanently unselectable, which violates the IPW assumption $\pi_i > 0$ required for valid inference.

---

## Cost Optimal Random Sampler

The `CostOptimalRandomSampler` addresses the following setting: two annotation sources are available, one **cheap but error-prone** (the proxy rater) and one **expensive but highly reliable** (the ground truth rater). The proxy is queried for every sample in the dataset. The sampler decides which samples to also send to the expensive rater, so that downstream estimation of the mean ground truth rating is as precise as possible within the available budget.

The sampler models two raters:

- **Proxy rater ($G$)**: cheap, always queried, cost $c_g$ per sample. Returns noisy labels.
- **Ground truth rater ($H$)**: expensive, cost $c_h$ per sample ($c_g < c_h$). Returns authoritative labels (e.g., a human annotator).

The simplest annotation policy queries the ground truth rater at a **fixed probability $\pi$** for every sample, regardless of its characteristics. When $\pi$ is too large the cost is too high; when $\pi$ is too small the downstream estimation variance blows up. The `CostOptimalRandomSampler` finds the optimal balance.

### The optimal sampling probability

The optimal probability minimises estimation error subject to the cost budget (see [[2](#ref-2), Proposition 1]). It depends on two quantities:

- $\text{Var}(H)$: variance of the ground truth labels
- $\text{MSE}(H, G) = \mathbb{E}[(H - G)^2]$: mean squared error of the proxy relative to the ground truth

The optimal probability $\pi^*$ is then:

$$
\pi^* = \begin{cases}
\sqrt{\dfrac{c_g}{c_h} \cdot \dfrac{\text{MSE}(H, G)}{\text{Var}(H) - \text{MSE}(H, G)}} & \text{if } \text{MSE}(H, G) < \dfrac{c_h}{c_h + c_g} \cdot \text{Var}(H) \\[0.5em]
1 & \text{otherwise}
\end{cases}
$$

**Interpretation:**

- If the first case condition holds, the proxy is accurate enough that querying the ground truth rater on only a fraction $\pi^* < 1$ of samples is cost-efficient. The optimal rate varies inversely with both the ratio $\text{Var}(H) / \text{MSE}(H, G)$ and the cost ratio $c_h / c_g$: a more accurate proxy or a more expensive ground truth rater both lead to a lower $\pi^*$.
- In the second case, the proxy is too unreliable and every sample must be sent to the expensive rater ($\pi^* = 1$).

The intuition: if $H$ has high variance but $G$ closely tracks it, the estimator can primarily exploit the proxy's cheap, high-quality predictions, querying the ground truth rater at a low rate only to correct for residual bias.

### Burn-in phase

To compute $\pi^*$, estimates of $\text{Var}(H)$ and $\text{MSE}(H, G)$ are needed. When no labeled data is available upfront, one can first annotate a number of initial samples unconditionally with ground truth ($\pi = 1$). This burn-in dataset is used to compute the required statistics. Once $\pi^*$ is determined, the burn-in data can be retained and reused by downstream estimators that support inverse probability weighting.

### Total cost and budget mapping

Since the proxy is queried for every sample, the expected cost per sample is:

$$\mathbb{E}[\text{Cost per sample}] = c_g + c_h \cdot \pi$$

Given a budget $b$ and optimal probability $\pi^*$, the maximum number of samples that can be processed is:

$$T = \left\lfloor \frac{b}{c_g + c_h \cdot \pi^*} \right\rfloor$$

### Sampling procedure

The annotation process proceeds in two stages. First, the sampler determines which samples to process, depending on how $T$ compares to the dataset size $N$:

- If $T < N$: then $T$ samples are drawn uniformly at random without replacement from the dataset.
- If $T \geq N$: all $N$ samples are used.

Second, each selected sample is independently sent to the expensive rater with probability $\pi^*$:

$$\xi_i \sim \mathrm{Bernoulli}(\pi^*), \quad i = 1, \ldots, \min(T, N)$$

where $\xi_i = 1$ means the sample additionally receives ground truth annotation and $\xi_i = 0$ means only the proxy annotation is used. All selected samples share the same drawing probability $\pi_i = \pi^*$.

---

## References

<a id="ref-1"></a>[1] <a id="ref-1-link" href="https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/12117.pdf">Fogliato, Riccardo, Pratik Patil, Mathew Monfort, and Pietro Perona. "A framework for efficient model evaluation through stratification, sampling, and estimation." *European Conference on Computer Vision*, pp. 140–158. Springer Nature Switzerland, 2024.</a>

<a id="ref-2"></a>[2] <a id="ref-2-link" href="https://arxiv.org/abs/2506.07949">Angelopoulos, Anastasios N., Jacob Eisenstein, Jonathan Berant, Alekh Agarwal, and Adam Fisch. "Cost-optimal active ai model evaluation." arXiv preprint arXiv:2506.07949 (2025).</a>
