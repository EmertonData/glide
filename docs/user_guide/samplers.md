# Samplers

Samplers sit **upstream of the estimators**: they decide *which* samples to send for human annotation. The simplest approach, implemented by `UniformSampler`, draws a budget $b$ of observations uniformly without replacement from a pool of $n$ samples and returns a binary selection vector $\xi \in \{0,1\}^n$. It is the appropriate baseline when no auxiliary signal is available.

When auxiliary signals are available, choosing samples strategically can substantially reduce the annotation budget needed to reach a target confidence-interval width. Other samplers (for example, `ActiveSampler`) go further: in addition to $\xi_i$, they compute a **drawing probability** $\pi_i$ for each sample, which allows downstream estimators to apply Inverse Probability Weighting (IPW) and correct for non-uniform sampling bias.

| Value | Description |
|---|---|
| $\pi_i$ | Drawing probability used to select sample $i$ for annotation ($0 < \pi_i \leq 1$) |
| $\xi_i$ | Bernoulli indicator: $1$ if the sample was selected, $0$ otherwise |

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

The `CostOptimalRandomSampler` addresses the following setting: two annotation sources are available, one **cheap but error-prone** (the proxy rater) and one **expensive but highly reliable** (the ground truth rater). A budget limit is imposed, potentially limiting the number of samples that can be annotated. The sampler determines how many samples are affordable and the proxy is queried for all of them. The sampler additionally decides which samples to also send to the expensive rater, so that downstream estimation of the mean ground truth rating is as precise as possible within the available budget.

The sampler models two raters:

- **Proxy rater ($\tilde{Y}$)**: cheap, always queried, cost $c_{\tilde{y}}$ per sample. Returns noisy labels.
- **Ground truth rater ($Y$)**: expensive, cost $c_y$ per sample ($c_{\tilde{y}} < c_y$). Returns authoritative labels (e.g., a human annotator).

The simplest annotation policy queries the ground truth rater at a **fixed probability $\pi$** for every sample, regardless of its characteristics. When $\pi$ is too large the cost is too high; when $\pi$ is too small the downstream estimation variance blows up. The `CostOptimalRandomSampler` finds the optimal balance.

### The optimal sampling probability

The optimal probability minimises estimation error subject to the cost budget (see [[2](#ref-2), Proposition 1]). It depends on two quantities:

- $\text{Var}(Y)$: variance of the ground truth labels
- $\text{MSE}(Y, \tilde{Y}) = \mathbb{E}[(Y - \tilde{Y})^2]$: mean squared error of the proxy relative to the ground truth

Note that [[2](#ref-2)] uses the notations $H$ and $G$ for $Y$ and $\tilde{Y}$ respectively, adapted here to harmonize with the rest of the documentation.

The optimal probability $\pi^*$ is then:

$$
\pi^* = \begin{cases}
\sqrt{\dfrac{c_{\tilde{y}}}{c_y} \cdot \dfrac{\text{MSE}(Y, \tilde{Y})}{\text{Var}(Y) - \text{MSE}(Y, \tilde{Y})}} & \text{if } \text{MSE}(Y, \tilde{Y}) < \dfrac{c_y}{c_y + c_{\tilde{y}}} \cdot \text{Var}(Y) \\[0.5em]
1 & \text{otherwise}
\end{cases}
$$

**Interpretation:**

- If the first case condition holds, the proxy is accurate enough that querying the ground truth rater on only a fraction $\pi^* < 1$ of samples is cost-efficient. The optimal rate varies inversely with both the ratio $\text{Var}(Y) / \text{MSE}(Y, \tilde{Y})$ and the cost ratio $c_y / c_{\tilde{y}}$: a more accurate proxy or a more expensive ground truth rater both lead to a lower $\pi^*$.
- In the second case, the proxy is too unreliable and every sample must be sent to the expensive rater ($\pi^* = 1$).

The intuition: if $H$ has high variance but $G$ closely tracks it, the estimator can primarily exploit the proxy's cheap, high-quality predictions, querying the ground truth rater at a low rate only to correct for residual bias.

### Burn-in phase

To compute $\pi^*$, estimates of $\text{Var}(Y)$ and $\text{MSE}(Y, \tilde{Y})$ are needed. When no labeled data is available upfront, one can first annotate a number of initial samples unconditionally with ground truth ($\pi = 1$). This burn-in dataset is used to compute the required statistics. Once $\pi^*$ is determined, the subsequent data is annotated with this probability and can be used by downstream estimators that support inverse probability weighting.

### Total cost and budget mapping

Since the proxy is queried for every sample, the expected cost per sample is:

$$\mathbb{E}[\text{Cost per sample}] = c_{\tilde{y}} + c_y \cdot \pi$$

Given a budget $b$ and optimal probability $\pi^*$, the maximum number of samples that can be processed is:

$$T = \left\lfloor \frac{b}{c_{\tilde{y}} + c_y \cdot \pi^*} \right\rfloor$$

### Sampling procedure

The annotation process proceeds in two stages. First, the sampler determines which samples to process, depending on how $T$ compares to the dataset size $N$:

- If $T < N$: then $T$ samples are drawn uniformly at random without replacement from the dataset.
- If $T \geq N$: all $N$ samples are used.

Second, each selected sample is independently sent to the expensive rater with probability $\pi^*$:

$$\xi_i \sim \mathrm{Bernoulli}(\pi^*), \quad i = 1, \ldots, \min(T, N)$$

where $\xi_i = 1$ means the sample additionally receives ground truth annotation and $\xi_i = 0$ means only the proxy annotation is used. All selected samples share the same drawing probability $\pi_i = \pi^*$.

---

## Cost Optimal Sampler

The `CostOptimalSampler` operates in the same two-rater setting as the `CostOptimalRandomSampler`, but replaces the **uniform annotation probability** with a **per-sample probability** that depends on how unreliable the proxy label is expected to be for each individual sample. Samples where the proxy is likely to err are annotated more frequently; samples where the proxy is reliable are annotated at a lower rate. This concentrates the annotation budget where it reduces variance the most.

The rater model is identical to the random variant:

- **Proxy rater ($\tilde{Y}$)**: cheap, always queried, cost $c_{\tilde{y}}$ per sample. Returns noisy labels.
- **Ground truth rater ($Y$)**: expensive, cost $c_y$ per sample ($c_{\tilde{y}} < c_y$). Returns authoritative labels.

Each sample $i$ additionally has a **per-sample uncertainty score** $u_i > 0$, a caller-supplied estimate of $\mathbb{E}[(Y_i - \tilde{Y}_i)^2 \mid X_i]$, the expected squared error of the proxy for that sample. These scores are not learned internally.

### Optimal active annotation policy

For a given threshold $\tau > 0$, the cost-optimal per-sample annotation probability takes the form (see [[2](#ref-2), Proposition 2]):

$$\pi_i(\tau) = \begin{cases} \gamma^*(\tau)\,\sqrt{u_i} & \text{if } \sqrt{u_i} \leq \tau \\ 1 & \text{if } \sqrt{u_i} > \tau \end{cases}$$

where the scale factor $\gamma^*(\tau)$ is:

$$\gamma^*(\tau) = \min\!\left(\sqrt{\dfrac{c_{\tilde{y}}/c_y + \Pr(U > \tau^2)}{\mathrm{Var}(Y) - \mathbb{E}[U \cdot \mathbf{1}_{U \leq \tau^2}]}},\;\dfrac{1}{\tau}\right)$$

Samples above the threshold are always sent to the ground truth rater ($\pi_i = 1$); below the threshold, the probability scales as $\sqrt{u_i}$, concentrating more effort on samples with higher expected proxy error.

The optimal threshold $\tau^*$ minimises the product of expected per-sample cost and per-sample estimation error:

$$\tau^* = \underset{\tau}{\operatorname{argmin}}\;\Bigl(c_y\,\mathbb{E}[\pi_i(\tau)] + c_{\tilde{y}}\Bigr)\cdot\Bigl(\mathrm{Var}(Y) + \mathbb{E}\!\left[U \cdot \!\left(\frac{1}{\pi_i(\tau)} - 1\right)\right]\Bigr)$$

The policy changes character only at the breakpoints $\tau \in \{\sqrt{u_i}\}$, so the minimisation is performed by exhaustive search over those distinct values.

**Interpretation:**

- Samples with small uncertainty ($\sqrt{u_i} \ll \tau^*$) have a reliable proxy: the ground truth rater is queried at a low rate, proportional to $\sqrt{u_i}$.
- Samples with large uncertainty ($\sqrt{u_i} > \tau^*$) have a proxy too unreliable to trust: the ground truth rater is always queried ($\pi_i = 1$).
- Compared to the `CostOptimalRandomSampler`, the active policy can achieve lower estimation variance for the same budget by exploiting heterogeneity in proxy quality across samples.

### Burn-in phase

The policy requires $\mathrm{Var}(Y)$. As with the random variant, this is estimated from a **burn-in dataset**: a set of samples annotated unconditionally by the ground truth rater before the active policy is applied. The burn-in labels are used to compute this variance estimate, after which the active policy governs subsequent annotation.

### Budget and sample selection

The expected cost of processing sample $i$ under the active policy is:

$$\mathbb{E}[\text{cost}_i] = c_y \cdot \pi_i + c_{\tilde{y}}$$

Because each sample has its own annotation probability, costs are accumulated in the order the samples appear in the input array. The budget $b$ determines the largest prefix of samples that can be processed:

$$T = \max\!\left\{t \,:\, \sum_{i=1}^{t} \bigl(c_y\,\pi_i + c_{\tilde{y}}\bigr) \leq b\right\}$$

A hard cutoff is applied: samples beyond index $T$ receive $\pi_i = 0$ and are excluded from sampling.

### Sampling procedure

For each selected sample $i \leq T$, the ground truth annotation is independently requested with probability $\pi_i$:

$$\xi_i \sim \mathrm{Bernoulli}(\pi_i), \quad i = 1, \ldots, T$$

where $\xi_i = 1$ means the sample is sent to the ground truth rater and $\xi_i = 0$ means only the proxy label is used. Unselected samples ($i > T$) receive $\pi_i = 0$ and $\xi_i = \mathrm{NaN}$, indicating they were not part of the sampling pool.

---

## References

<a id="ref-1"></a>[1] <a id="ref-1-link" href="https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/12117.pdf">Fogliato, Riccardo, Pratik Patil, Mathew Monfort, and Pietro Perona. "A framework for efficient model evaluation through stratification, sampling, and estimation." *European Conference on Computer Vision*, pp. 140–158. Springer Nature Switzerland, 2024.</a>

<a id="ref-2"></a>[2] <a id="ref-2-link" href="https://arxiv.org/abs/2506.07949">Angelopoulos, Anastasios N., Jacob Eisenstein, Jonathan Berant, Alekh Agarwal, and Adam Fisch. "Cost-optimal active ai model evaluation." arXiv preprint arXiv:2506.07949 (2025).</a>
