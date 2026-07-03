# Monitors

Once a metric has been estimated validating an AI system's deployment, the question shifts from "what is the metric today?" to "has the metric drifted?" GLIDE's monitors answer this with **anytime-valid drift monitoring** [[1](#ref-1)]: a sequential procedure that watches a metric across successive batches of production data and raises an alarm the moment there is statistically valid evidence that it has crossed a threshold, no matter how many times it has already been checked.

---

## The monitoring problem

After deployment, the metric is re-estimated on successive batches of production data $t = 1, 2, \dots$, for instance a fresh monthly batch with new labeled samples. Each batch $t$ yields an estimate $\hat{\theta}_t$. The question is whether the sequence proves that the metric has drifted past a threshold $\tau$ that the user fixes in advance: the worst running value they are willing to tolerate.

The naive approach is statistically invalid. Comparing each $\hat{\theta}_t$ to $\tau$ with an ordinary confidence interval at level $1 - \alpha$ carries its own false-alarm probability $\alpha$ for that single comparison. Checking after every batch means accumulating many such chances to be wrong. More precisely, the probability of a false alarm after $t$ tests is $1-(1-\alpha)^t$, so a false alarm becomes almost certain over a long enough horizon. This is a form of **multiple testing**, here called **peeking**, and it is exactly what invalidates repeated per-batch significance testing.

The fix is a **confidence sequence**: a sequence of intervals $\{C_t\}_{t \ge 1}$ that covers the target *simultaneously at all times*,

$$\Pr\!\left(\forall t \ge 1:\; \bar{\theta}_t \in C_t\right) \ge 1 - \delta,$$

so the user may look after every batch and the total false-alarm probability over the entire monitoring horizon still stays below the single budget $\delta$. This **anytime-valid** guarantee is what makes peeking safe. It stands in contrast to fixed-sample confidence intervals which are valid only at a single, pre-committed sample size and lose their guarantee the moment they are checked repeatedly.

---

## Setting

Each batch $t$ carries the same inputs as [Prediction-Powered Inference (PPI++)](estimators.md#prediction-powered-inference-ppi): a small set of human labels and a larger set of proxy labels, both specific to that batch.

| Value | Present for | Description |
|---|---|---|
| $\tilde{Y}_{t,i}$ | All samples in batch $t$ | Proxy label |
| $Y_{t,j}$ | Labeled samples in batch $t$ only | Ground-truth label |

Every batch is monitored: there is no separate calibration batch, because the comparison target is the user-fixed threshold $\tau$ rather than a value estimated from the data.

The derivation below treats the metric as a **risk** $R$, where lower is better (for example an error rate). The running risk after $t$ batches is the average of the per-batch prediction-powered estimates:

$$\bar{R}_t = \frac{1}{t} \sum_{s=1}^{t} \hat{R}_s,$$

where $\hat{R}_s$ is the PPI++ estimate of the risk on batch $s$ alone. A performance metric, where higher is better (for example accuracy), is monitored by applying everything below to $1 - R$ instead of $R$; this single mirror transformation is the only adjustment needed, and it is not repeated elsewhere in this page.

The derivation also assumes $R \in [0, 1]$. This boundedness is required because the confidence sequence construction below relies on an empirical-Bernstein boundary, which in turn requires a known, bounded range for the quantity being monitored. Section [Normalization onto the unit interval](#normalization-onto-the-unit-interval) shows how a raw PPI++ estimate, which is not itself confined to $[0, 1]$, is mapped onto it.

---

## From Markov to Ville: the anytime-valid guarantee

This section builds the anytime-valid guarantee from first principles, in three steps: a classical tail bound, its sequential upgrade, and a betting interpretation that makes the upgrade constructive.

**Markov's inequality.** For a nonnegative random variable $W$ with $E[W] \le 1$,

$$\Pr(W \ge 1/\delta) \le \delta \, E[W] \le \delta.$$

Any such $W$ turns the event $\{W \ge 1/\delta\}$ into a test of level $\delta$: it is unlikely to occur if $E[W] \le 1$. But this bound is only about $W$ at a single, fixed point; it says nothing about whether $W$ might cross $1/\delta$ at some earlier or later time even if it is back below threshold now.

**Ville's inequality.** The sequential upgrade. For a nonnegative supermartingale $\{W_t\}_{t \ge 0}$ with $W_0 = 1$ and $E[W_t \mid \mathcal{F}_{t-1}] \le W_{t-1}$,

$$\Pr\!\left(\exists t \ge 1:\; W_t \ge 1/\delta\right) \le \delta.$$

The difference from Markov is essential: the probability bounded here is that of *ever* crossing $1/\delta$, over the entire, unbounded horizon, not the probability at one fixed time. This is exactly the anytime-valid property that a confidence sequence needs.

**The betting / wealth reading.** Ville's inequality becomes constructive once $W_t$ is read as the wealth of a gambler betting against the null hypothesis "no drift", starting with one unit of capital. A bet that is fair under the null keeps $W_t$ a supermartingale, so under the null the gambler cannot expect to get rich: Ville's inequality caps how much luck they can have, over the whole sequence of bets. Under genuine drift, however, the bets are informative and tend to pay off, so the wealth grows; reaching $W_t \ge 1/\delta$ is therefore calibrated evidence of drift, not a coincidence.

The **betting parameter** $\beta \in (0, 1)$ controls how aggressively the gambler bets at each step. For a fixed $\beta$, the wealth process is

$$W_t(\beta) = \exp\!\left(\beta \, S_t - \psi_E(\beta) \, V_t\right),$$

where:

- $S_t = \sum_{s=1}^{t} (X_s - c_s)$ is the cumulative deviation of the normalized per-batch estimates $X_s$ (Section [Normalization onto the unit interval](#normalization-onto-the-unit-interval)) from their **predictable centers** $c_s$,
- $V_t = \sum_{s=1}^{t} (X_s - c_s)^2$ is the running empirical variance of those deviations,
- $\psi_E(\beta) = -\log(1 - \beta) - \beta$ is a cumulant-generating-function penalty, chosen exactly so that $W_t(\beta)$ is a supermartingale under the null.

A **predictable center** $c_s$ is a quantity fixed before batch $s$ is observed, most naturally the running mean of the normalized estimates from all previous batches, $c_s = \bar{X}_{s-1}$ for $s \ge 2$. Fixing $c_s$ this way makes the increment $X_s - c_s$ conditionally mean-zero under the null, which is what secures the supermartingale property.

There is no previous batch before the first one, so $c_1$ must be seeded by a constant fixed in advance. This is where the guide seeds with the threshold $\tau$: predictability alone is what secures validity, so *any* constant fixed in advance ($0$, the midpoint of the range, $\tau$, ...) gives a valid sequence, and the choice of seed affects only the width of the very first look, never coverage. The guide seeds with $\tau$ specifically because it is the null-hypothesized running risk at the decision boundary: it is the pre-data analogue of the running mean that later batches center on, and it makes the first contribution to $V_t$ smallest exactly when an early batch lands near the threshold, which is the regime the test cares about most. Because the seed enters only the first term of $V_t$, its influence washes out as batches accumulate; a poorly chosen seed can only inflate $V_t$, which widens the early boundary and *delays* an alarm, but it can never create a false one.

**The method of mixtures.** The betting parameter that would extract the most evidence fastest depends on the (unknown) size of the drift and on $V_t$, so rather than commit to a single $\beta$, the guide averages the wealth process over a density $q(\beta)$ on $(0, 1)$:

$$W_t = \int_0^1 W_t(\beta) \, q(\beta) \, d\beta.$$

A mixture of nonnegative supermartingales, each starting at $1$, is itself a nonnegative supermartingale starting at $1$, so Ville's inequality applies for *any* choice of $q$; the density affects only the tightness of the resulting bound, never its validity. GLIDE takes $q$ uniform on $(0, 1)$: parameter-free and horizon-agnostic, and it drops out of the integrand entirely. A conjugate choice of $q$ would yield a closed-form mixture and a slightly tighter boundary, at the price of introducing a sharpness hyperparameter that would have to be tuned.

---

## The empirical-Bernstein boundary

Fix the variance process at a value $v$. The **boundary** is the largest cumulative deviation still consistent with the null,

$$u(v) = \sup\{\, s \ge 0 : W(s, v) \le 1/\delta \,\},$$

the right edge of the acceptance region, expressed as a deviation budget, where $\delta$ is the sequence's single false-alarm budget (its miscoverage). Under the uniform mixture, the wealth process as a function of the deviation $s$ is

$$W(s, v) = \int_0^1 \exp\!\left(\beta s - \psi_E(\beta) v\right) d\beta.$$

This function is continuous and strictly increasing in $s$: its derivative with respect to $s$, $\int_0^1 \beta \exp(\beta s - \psi_E(\beta) v)\, d\beta$, is strictly positive, and $W(s, v)$ rises from a value $\le 1$ at $s = 0$ to $\infty$ as $s \to \infty$. The supremum defining $u(v)$ is therefore attained at the unique root of

$$\int_0^1 \exp\!\left(\beta \, u(v) - \psi_E(\beta) \, v\right) d\beta = \frac{1}{\delta}.$$

The boundary is called "empirical-Bernstein" because its width adapts to the *observed* variability $V_t$ rather than assuming a fixed, known worst-case variance [[2](#ref-2)], which is what makes it tight in practice.

Dividing the deviation budget by the batch count converts it into a bound on the running mean. The anytime-valid lower bound on the running risk after $t$ batches is

$$L_t = \bar{R}_t - \frac{u(V_t)}{t}.$$

As an analytical landmark, consider the zero-variance case $v = 0$: the penalty term vanishes and $\int_0^1 e^{\beta s}\, d\beta = (e^{s} - 1)/s$, so $u(0)$ solves

$$\frac{e^{u(0)} - 1}{u(0)} = \frac{1}{\delta}.$$

---

## Normalization onto the unit interval

The empirical-Bernstein boundary of the previous section is valid only for variables confined to a *known* bounded range. A prediction-powered estimate of a metric in $[0, 1]$ does not itself lie in $[0, 1]$: the power-tuning weight $\lambda$ is a variance-minimizing regression slope, not a mixing weight confined to $[0, 1]$, so an under-dispersed or anti-correlated proxy can push $\lambda$, and with it the estimate, outside any fixed interval.

Clipping the power-tuning weight to $[0, \lambda_{\max}]$ confines a $[0, 1]$-metric estimate $\hat{R}$ to the range $[-\lambda_{\max},\, 1 + \lambda_{\max}]$. The affine map

$$X = \frac{\hat{R} + \lambda_{\max}}{1 + 2\lambda_{\max}}$$

sends that range exactly onto $[0, 1]$, producing the normalized estimate $X$ used throughout the previous two sections. A larger $\lambda_{\max}$ permits more reliance on the proxy, but it also widens the denominator of the map above, which scales up every deviation and therefore the width of the boundary $u(v)$: more proxy reliance is traded against slower detection.

---

## Predictable power-tuning

The power-tuning weight $\lambda_t$ used to form the estimate $\hat{R}_t$ on batch $t$ must itself be **predictable**: computed only from batches strictly earlier than $t$, never from batch $t$ itself. This is what makes each per-batch increment conditionally mean-zero under the null, which is exactly the condition that keeps $W_t$ a supermartingale.

Predictability is the only validity constraint here. *Which* prior batches are pooled to compute $\lambda_t$ is purely a matter of statistical power, and any predictable choice remains valid; only the width of the resulting boundary changes. GLIDE pools the full prior history of batches to compute $\lambda_t$. The first batch has no predecessor, so it simply uses the neutral weight $\lambda_1 = 1$, which is itself trivially predictable since it depends on no data at all.

---

## The alarm rule and the single budget

The user fixes a threshold $\tau$ in advance: the worst running risk they are willing to tolerate, in metric units. The drift alarm at batch $t$ fires when the anytime-valid lower bound on the running risk crosses it,

$$L_t > \tau.$$

Because $\tau$ is a known constant, not a quantity estimated from data, the only way to raise a false alarm under the null ($\bar{R}_t \le \tau$ for every $t$) is for the confidence sequence itself to fail to cover the running risk at some point in the horizon, an event with probability at most its single budget $\delta$. Under no drift, the probability of *ever* raising a false alarm is therefore at most

$$\delta,$$

the sequence's lone miscoverage. This is simpler than the two-sample setup in [[1](#ref-1)], where the fixed threshold is replaced by a reference bound $U_0$ *estimated* from a labeled reference sample; that reference bound carries its own miscoverage $\delta_S$, so the total false-alarm budget there is $\delta_S + \delta$, since miscoverages add. Fixing $\tau$ in advance removes $\delta_S$ entirely.

A smaller $\delta$ widens the confidence sequence and delays alarms, so the budget directly governs detection speed, not just the false-alarm rate.

Finally, a caveat on what this monitors: PPRM tracks the running *average* $\bar{R}_t$, so a short transient drift is progressively diluted as more batches accumulate into that average. Sensitivity to abrupt, recent drifts is recovered by restricting the running average to the most recent batches rather than the full history.

---

## References

<a id="ref-1"></a>[1] <a id="ref-1-link" href="https://arxiv.org/abs/2602.02229">Zhang, Guangyi, Yunlong Cai, Guanding Yu, and Osvaldo Simeone. "Prediction-Powered Risk Monitoring of Deployed Models for Detecting Harmful Distribution Shifts." arXiv preprint arXiv:2602.02229 (2026).</a>.

<a id="ref-2"></a>[2] <a id="ref-2-link" href="https://arxiv.org/abs/2010.09686">Waudby-Smith, Ian, and Aaditya Ramdas. "Estimating means of bounded random variables by betting." Journal of the Royal Statistical Society Series B: Statistical Methodology 86, no. 1 (2024): 1-27.</a>.

<a id="ref-3"></a>[3] <a id="ref-3-link" href="https://arxiv.org/abs/1810.08240">Howard, Steven R., Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon. "Time-uniform, nonparametric, nonasymptotic confidence sequences." The Annals of Statistics 49, no. 2 (2021): 1055-1080.</a>.
