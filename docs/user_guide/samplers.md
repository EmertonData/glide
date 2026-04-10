# Samplers

Samplers sit **upstream of the estimators**: they decide *which* items to send for human annotation and compute the sampling probabilities $\pi_i$ required by estimators that rely on Inverse Probability Weighting (IPW). Choosing items strategically — rather than uniformly at random — can substantially reduce the annotation budget needed to reach a target confidence-interval width.

A sampler takes a fully proxy-labelled dataset and a budget $b$, and computes two values per item:

| Value | Description |
|---|---|
| $\pi_i$ | Drawing probability used to select item $i$ for annotation ($0 < \pi_i \leq 1$) |
| $\xi_i$ | Bernoulli indicator: $1$ if the item was selected, $0$ otherwise |

The downstream estimator uses these values to correct for non-uniform selection. Both samplers in GLIDE produce outputs that are directly compatible with the [ASI estimator](estimators.md#active-statistical-inference-asi).

---

## Active Sampler

The `ActiveSampler` concentrates the annotation budget on the **most uncertain records**. The intuition is that high-uncertainty records are those where the proxy label is least reliable, so human annotation there yields the greatest reduction in variance.

### Sampling probabilities

Each item $i$ has an **uncertainty score** $u_i > 0$. The raw drawing probability is set proportional to $u_i$, normalised so that the expected number of selected records equals the budget $b$:

$$\tilde{\pi}_i = b \cdot \frac{u_i}{\sum_{j=1}^{n} u_j}$$

Because $\tilde{\pi}_i$ must be a valid Bernoulli probability, values are capped at 1:

$$\pi_i = \min\!\left(\tilde{\pi}_i,\; 1\right)$$

Records whose raw probability exceeds 1 are selected with certainty ($\pi_i = 1$). As a result, the actual expected number of selected records is at most $b$ (it may be slightly less when some records are capped).

### Sampling procedure

Given the probabilities $\pi_i$, each item is independently selected via a Bernoulli trial:

$$\xi_i \sim \mathrm{Bernoulli}(\pi_i), \quad i = 1, \ldots, n$$

The draws are independent across items. Each item receives values $(\pi_i, \xi_i)$; items with $\xi_i = 1$ are the ones to be sent for human annotation.

Note that uncertainty scores must be strictly positive. The `ActiveSampler` requires $u_i > 0$ for every item. A zero or negative uncertainty score would result in a zero sampling probability, making that item permanently unselectable — which violates the IPW assumption $\pi_i > 0$ required for valid inference.

---

## Stratified Sampler

When the dataset is naturally partitioned into **strata** — groups such as language, domain, or topic — the `StratifiedSampler` allocates the annotation budget across strata before performing per-stratum Bernoulli sampling. It supports two allocation strategies: **proportional** (baseline) and **Neyman** (default, variance-optimal).

Let $K$ denote the number of strata. Stratum $h$ contains $N_h$ items. The total dataset size is $N = \sum_{h=1}^{K} N_h$ and the annotation budget is $b$.

### Proportional allocation

Proportional allocation assigns budget to each stratum in proportion to its size:

$$n_h^{\text{prop}} = b \cdot \frac{N_h}{N}$$

This yields the same sampling probability $\pi_i = b / N$ for every item, regardless of stratum. It is equivalent to simple random sampling and serves as a baseline.

### Neyman allocation

Neyman allocation assigns more budget to strata with **higher proxy variance**, minimising the asymptotic variance of downstream estimators [[1](#ref-1)]. The raw allocation for stratum $h$ is:

$$n_h^{\text{ney}} = b \cdot \frac{N_h\, \hat{\sigma}_h}{\displaystyle\sum_{k=1}^{K} N_k\, \hat{\sigma}_k}$$

where $\hat{\sigma}_h$ is the sample standard deviation of the proxy labels within stratum $h$. Strata where the proxy is more dispersed receive a larger slice of the budget, because annotation effort there reduces total variance most efficiently.

When proxy variance is roughly uniform across strata, Neyman allocation reduces approximately to proportional allocation.

### Rounding

Both formulas produce fractional per-stratum counts. GLIDE resolves this with **largest-remainder rounding** (Hamilton's method): each stratum first receives $\lfloor n_h \rfloor$ slots, then the strata with the largest fractional remainders receive one extra slot each until the total reaches $b$. Per-stratum allocations are additionally capped at $N_h$, so the total allocated budget may be slightly less than $b$ when some strata are very small or have very high variance relatively to others.

### Sampling probabilities and procedure

Once the integer allocation $n_h$ is determined, every item in stratum $h$ receives the same drawing probability:

$$\pi_i = \min\!\left(\frac{n_h}{N_h},\; 1\right)$$

Each item is then independently selected via a Bernoulli trial:

$$\xi_i \sim \mathrm{Bernoulli}(\pi_i), \quad i = 1, \ldots, n$$

Each item receives values $(\pi_i, \xi_i)$. Items with $\xi_i = 1$ are sent for human annotation.

Note that strata must have at least 2 items with non-zero variance. Neyman allocation requires computing a standard deviation, which needs at least 2 distinct proxy values per stratum. Strata that are too small or have zero proxy variance will raise a `ValueError`.

Note also that all strata must receive at least one annotation slot. If the budget is too small relative to the number of strata, some strata may receive a zero allocation. This violates the IPW assumption $\pi_i > 0$ and GLIDE raises an error. Either increase the budget or reduce the number of strata.

---

## References

<a id="ref-1"></a>[1] <a id="ref-1-link" href="https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/12117.pdf">Fogliato, Riccardo, Pratik Patil, Mathew Monfort, and Pietro Perona. "A framework for efficient model evaluation through stratification, sampling, and estimation." *European Conference on Computer Vision*, pp. 140–158. Springer Nature Switzerland, 2024.</a>

<a id="ref-2"></a>[2] <a id="ref-2-link" href="https://proceedings.mlr.press/v235/zrnic24a.html">Zrnic, Tijana, and Emmanuel J. Candès. "Active statistical inference." Proceedings of the 41st International Conference on Machine Learning. 2024</a>.
