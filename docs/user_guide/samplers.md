# Samplers

Samplers sit **upstream of the estimators**: they decide *which* samples to send for human annotation. The simplest approach, implemented by `UniformSampler`, draws a budget $b$ of observations uniformly without replacement from a pool of $n$ samples and returns a binary selection vector $\xi \in \{0,1\}^n$. It is the appropriate baseline when no auxiliary signal is available.

When auxiliary signals are available, choosing samples strategically can substantially reduce the annotation budget needed to reach a target confidence-interval width. Other samplers (for example, `ActiveSampler`) go further: in addition to $\xi_i$, they compute a **drawing probability** $\pi_i$ for each sample, which allows downstream estimators to apply Inverse Probability Weighting (IPW) and correct for non-uniform sampling bias.

The ultimate purpose is to estimate the mean $\mathbb{E}[Y]$ of ground truths $Y$ that are costly to obtain. Proxys label $\tilde{Y}$, available cheaply for all samples, serve as an auxiliary signal. When per-sample proxy errors $\mathbb{E}[(Y_i - \tilde{Y}_i)^2 \mid X_i]$ can be estimated for each input $X_i$, they reveal which samples the proxy is least reliable for and can guide the allocation of the annotation budget to reduce estimation variance.

| Value | Description |
|---|---|
| $Y_i$ | Ground truth label for sample $i$ |
| $\tilde{Y}_i$ | Proxy label for sample $i$ |
| $X_i$ | Input for sample $i$ |
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

Each sample $i$ has an **uncertainty score** $u_i > 0$, an estimate of the root mean squared error of the proxy label relative to the ground truth, $\sqrt{\mathbb{E}[(Y_i - \tilde{Y}_i)^2 \mid X_i]}$. The variance of an IPW-based estimator (see [Estimators](estimators.md) for examples) takes the form (see [[3](#ref-3), Equation (2)]):

$$\mathrm{Var}(Y) + \mathbb{E}\!\left[(Y - \tilde{Y})^2\left(\frac{1}{\pi(X)} - 1\right)\right]$$

The sampling probabilities are chosen to minimise this quantity. Because it depends on the probabilities only through $\mathbb{E}\!\left[\frac{(Y - \tilde{Y})^2}{\pi(X)}\right]$, the problem reduces to solving the optimisation:

$$\mathrm{minimize} \sum_i \frac{u_i^2}{\pi_i}$$

subject to $\pi_i \in (0, 1]$ for all $i$ and $\sum_i \pi_i = b$. When all resulting probabilities are valid ($\tilde{\pi}_i \leq 1$), the closed-form solution is:

$$\tilde{\pi}_i = b \cdot \frac{u_i}{\sum_{j=1}^{n} u_j}$$


When some uncertainty scores are large enough to push $\tilde{\pi}_i$ above 1, numerical optimisation finds the optimal solution satisfying the constraints.

Uncertainty scores must be strictly positive: $u_i > 0$ for every sample. A zero or negative score would assign a zero probability to that sample, making it unselectable.

### Sampling procedure

Given the probabilities $\pi_i$, each sample is independently selected via a Bernoulli trial:

$$\xi_i \sim \mathrm{Bernoulli}(\pi_i)$$

The draws are independent across samples. Each sample receives values $(\pi_i, \xi_i)$; samples with $\xi_i = 1$ are sent for human annotation.

Because the draws are random, the total number of selected samples equals the budget only in expectation. To prevent overshooting, the $\xi_i$ draws can be performed one by one, stopping as soon as the budget is reached. Samples not drawn when the budget is exhausted receive $\pi_i = 0$ and $\xi_i = \mathrm{NaN}$, indicating no Bernoulli draw was performed.

To ensure the cutoff does not depend on the input order, the samples can be shuffled before the draws begin and the results are returned in the original order.

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

The intuition: if $Y$ has high variance but $\tilde{Y}$ closely tracks it, the estimator can primarily exploit the proxy's cheap, high-quality predictions, querying the ground truth rater at a low rate only to correct for residual bias.

### Burn-in phase

To compute $\pi^*$, estimates of $\text{Var}(Y)$ and $\text{MSE}(Y, \tilde{Y})$ are needed. When no labeled data is available upfront, one can first annotate a number of initial samples unconditionally with ground truth ($\pi = 1$). This burn-in dataset is used to compute the required statistics. Once $\pi^*$ is determined, the subsequent data is annotated with this probability and can be used by downstream estimators that support inverse probability weighting.

### Sampling procedure

Each sample is sent to the expensive rater via a Bernoulli trial:

$$\xi_i \sim \mathrm{Bernoulli}(\pi_i)$$

where $\xi_i = 1$ means the sample additionally receives ground truth annotation and $\xi_i = 0$ means only the proxy label is used. All processed samples share the drawing probability $\pi_i = \pi^*$. 

Since the proxy is queried for every processed sample, the actual cost of sample $i$ is $c_{\tilde{y}} + c_y \cdot \xi_i$, with expected value:

$$\mathbb{E}[\text{cost}_i] = c_{\tilde{y}} + c_y \cdot \pi^*$$

The draws can be performed one by one, accumulating the actual cost of each sample until the budget $b$ is exhausted. Samples not reached when the budget runs out receive $\pi_i = 0$ and $\xi_i = \mathrm{NaN}$, indicating no draw was performed.

To ensure the cutoff does not depend on the input order, the samples can be shuffled before the draws begin and the results are returned in the original order.

---

## Cost Optimal Sampler

The `CostOptimalSampler` generalizes the `CostOptimalRandomSampler`. Two annotation sources are available: one **cheap but error-prone** (the proxy rater) and one **expensive but highly reliable** (the ground truth rater). A budget limit is imposed, potentially limiting the number of samples that can be annotated. Rather than querying the ground truth rater at a **fixed probability** for every sample, the `CostOptimalSampler` computes an **active policy**: a per-sample annotation probability that depends on how unreliable the proxy label is expected to be for that sample. Samples where the proxy is likely to err are annotated more frequently; samples where the proxy is reliable are annotated at a lower rate. This concentrates the annotation budget where it reduces variance the most.

The sampler models two raters:

- **Proxy rater ($\tilde{Y}$)**: cheap, always queried, cost $c_{\tilde{y}}$ per sample. Returns noisy labels.
- **Ground truth rater ($Y$)**: expensive, cost $c_y$ per sample ($c_{\tilde{y}} < c_y$). Returns authoritative labels.

Each sample $i$ is associated with an input $X_i$ and has a **per-sample uncertainty score** $u_i > 0$, a supplied estimate of $U_i := \sqrt{\mathbb{E}[(Y_i - \tilde{Y}_i)^2 \mid X_i]}$, the root mean squared error of the proxy for that sample. Note that the reference paper [[2](#ref-2)] defines the uncertainty score as the mean squared error; GLIDE uses RMSE instead, so that uncertainty scores are in the same units as the labels.

The goal is to query optimal amounts of proxy and ground truth ratings within the budget limit allowing to estimate the average $\mathbb{E}[Y]$ with the smallest possible variance.

### Optimal active annotation policy

The guiding intuition is that a sample where the proxy is likely to err should be sent to the ground truth rater more often than one where the proxy is reliable. The per-sample uncertainty score $u_i$ directly measures proxy unreliability, comparable to a standard deviation. The cost-optimal policy sets the annotation probability proportional to this uncertainty:

$$\pi_i \propto u_i$$

with two refinements. First, a proportionality constant $\gamma^*$ scales the probabilities and is chosen to optimise the precision of downstream estimators. Second, a threshold $\tau > 0$ acts as a risk tolerance: if the uncertainty $u_i$ exceeds $\tau$, the proxy is considered too unreliable to trust and the sample is always sent to the ground truth rater ($\pi_i = 1$).

Formally, the cost-optimal per-sample annotation probability takes the form (see [[2](#ref-2), Proposition 2]):

$$\pi_i = \pi(\tau, X_i) = \begin{cases} \gamma^*(\tau)\,u_i & \text{if } u_i \leq \tau \\ 1 & \text{otherwise} \end{cases}$$

- Samples with small uncertainty ($u_i \leq \tau^*$) have a reliable proxy: the ground truth rater is queried at a low rate, proportional to $u_i$.
- Samples with large uncertainty ($u_i > \tau^*$) have a proxy too unreliable to trust: the ground truth rater is always queried ($\pi_i = 1$).

The scale factor $\gamma^*(\tau)$ is:

$$\gamma^*(\tau) = \min\!\left(\sqrt{\dfrac{c_{\tilde{y}}/c_y + \Pr(U > \tau)}{\mathrm{Var}(Y) - \mathbb{E}[U^2 \cdot \mathbf{1}_{U \leq \tau}]}},\;\dfrac{1}{\tau}\right)$$

The optimal threshold $\tau^*$ minimises the product of expected per-sample cost and per-sample estimation error:

$$\tau^* = \underset{\tau}{\operatorname{argmin}}\;\Bigl(c_y\,\mathbb{E}[\pi(\tau, X)] + c_{\tilde{y}}\Bigr)\cdot\Bigl(\mathrm{Var}(Y) + \mathbb{E}\!\left[U^2 \cdot \!\left(\frac{1}{\pi(\tau, X)} - 1\right)\right]\Bigr)$$

The second term in the product can be shown to reflect the variance of an estimator of $\theta := \mathbb{E}[Y]$ (see [[3](#ref-3), Equation (2)]). Since this estimator would be unbiased, this also corresponds to its mean squared error.

Once $\tau^*$ is computed, the optimal annotation probabilities are obtained by plugging $\gamma^*(\tau^*)$ into $\pi_i$ for all $i$.

Compared to the `CostOptimalRandomSampler`, the active policy can achieve lower estimation variance for the same budget by exploiting heterogeneity in proxy quality across samples.

Note also that when all uncertainty scores are equal ($u_i = u$ for all $i$), the `CostOptimalSampler` recovers the same constant labelling probability $\pi^*$ computed in the `CostOptimalRandomSampler`.

### Burn-in phase

In order to compute the optimal policy, the ground truth label variance $\mathrm{Var}(Y)$ needs to be known. This is estimated from a **burn-in dataset**: a set of samples annotated unconditionally by the ground truth rater before the active policy is applied. The burn-in labels are used to compute this variance estimate, after which the active policy governs subsequent annotation.

### Sampling procedure

Each sample is sent to the expensive rater via a Bernoulli trial:

$$\xi_i \sim \mathrm{Bernoulli}(\pi_i)$$

where $\xi_i = 1$ means the sample additionally receives ground truth annotation and $\xi_i = 0$ means only the proxy label is used. Since the proxy is queried for every sample, the actual cost of sample $i$ is $c_{\tilde{y}} + c_y \cdot \xi_i$, with expected value:

$$\mathbb{E}[\text{cost}_i] = c_{\tilde{y}} + c_y \cdot \pi_i$$

The draws can be performed one by one, accumulating the actual cost of each sample until the budget $b$ is exhausted. Samples not reached when the budget runs out receive $\pi_i = 0$ and $\xi_i = \mathrm{NaN}$, indicating no draw was performed.

To ensure the cutoff does not depend on the input order, the samples can be shuffled before the draws begin and the results are returned in the original order.

---

## References

<a id="ref-1"></a>[1] <a id="ref-1-link" href="https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/12117.pdf">Fogliato, Riccardo, Pratik Patil, Mathew Monfort, and Pietro Perona. "A framework for efficient model evaluation through stratification, sampling, and estimation." In European Conference on Computer Vision, pp. 140-158. Cham: Springer Nature Switzerland, 2024.</a>.

<a id="ref-2"></a>[2] <a id="ref-2-link" href="https://arxiv.org/abs/2506.07949">Angelopoulos, Anastasios N., Jacob Eisenstein, Jonathan Berant, Alekh Agarwal, and Adam Fisch. "Cost-optimal active ai model evaluation." arXiv preprint arXiv:2506.07949 (2025).</a>

<a id="ref-3"></a>[3] <a id="ref-3-link" href="https://proceedings.mlr.press/v235/zrnic24a.html">Zrnic, Tijana, and Emmanuel J. Candès. "Active statistical inference." Proceedings of the 41st International Conference on Machine Learning. 2024</a>.
