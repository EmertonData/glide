# Monitors

Once a metric has been estimated validating an AI system's deployment, the question shifts from "what is the metric today?" to "has the metric drifted?" GLIDE's monitors answer this with **anytime-valid drift monitoring** [[2](#ref-2)]: a sequential procedure that watches a metric across successive batches of production data and raises an alarm the moment there is statistically valid evidence that it has crossed a threshold.

---

## The Monitoring Problem

After deployment, the metric is re-estimated on successive batches of production data $t = 1, 2, \dots$, for instance a fresh weekly batch with new labeled samples. Each batch $t$ yields an estimate $\hat{\theta}_t$. The question is whether the sequence proves that the metric has drifted past a threshold $\tau$ that the user fixes in advance: the worst running value they are willing to tolerate.

The naive approach is statistically invalid. Comparing each $\hat{\theta}_t$ to $\tau$ with an ordinary confidence interval at level $1 - \alpha$ carries its own false-alarm probability $\alpha$ for that single comparison. Checking after every batch means accumulating many such chances to be wrong. More precisely, the probability of a false alarm after $t$ tests is $1-(1-\alpha)^t$, so a false alarm becomes almost certain over a long enough horizon. This is a form of **multiple testing**, here called **peeking**, and it is exactly what invalidates repeated per-batch significance testing.

The fix is a **confidence sequence**: a sequence of intervals $\{C_t\}_{t \ge 1}$ that covers the target *simultaneously at all times*,

$$\Pr\!\left(\forall t \ge 1:\; \bar{\theta}_t \in C_t\right) \ge 1 - \delta,$$

so the user may look after every batch and the total false-alarm probability over the entire monitoring horizon still stays below the single budget $\delta$. This **anytime-valid** guarantee is what makes peeking safe. It stands in contrast to fixed-sample confidence intervals which are valid only at a single, pre-committed sample size and lose their guarantee the moment they are checked repeatedly.

Such a confidence sequence is built below from first principles, then applied to two settings: [Classical risk monitoring](#classical-risk-monitoring), where the per-batch estimate comes from human labels alone, and [Prediction-Powered Risk Monitoring (PPRM)](#prediction-powered-risk-monitoring-pprm), where it is combined with proxy labels for faster detection at the same false-alarm budget.

---

## Classical Risk Monitoring

The anytime-valid guarantee is derived here in the setting where the per-batch estimate comes directly from a batch of human-labeled samples, building up to the alarm rule that turns that guarantee into an actionable drift alarm.

### Setting

Each batch $t$ contributes a set of human labels.

| Value | Present for | Description |
|---|---|---|
| $Y_{t,j}$ | All labeled samples in batch $t$ | Ground-truth label |

In the following, we will assume that all labels $Y_{t,j}$ are in $[0, 1]$. Every batch is monitored relatively to a user-fixed threshold $\tau$.

The derivation treats the metric as a **risk** $R$, where lower is better (for example an error rate). The running risk after $t$ batches is the average of the per-batch estimates:

$$\bar{R}_t = \frac{1}{t} \sum_{s=1}^{t} \hat{R}_s,$$

where

$$\hat{R}_s = \frac{1}{n_s}\sum_{j=1}^{n_s} Y_{s,j}$$

is the classical sample mean of the $n_s$ labels in batch $s$ alone. A performance metric, where higher is better (for example accuracy), is monitored by applying the same methodology to $1 - R$ instead of $R$.

The derivation only requires $R$ to be **bounded** within some known range so that it can be affinely normalized onto $[0, 1]$ without compromising the statistical guarantees.

### From Markov to Ville: the Anytime-Valid Guarantee

The anytime-valid guarantee is built below from first principles, in three steps: a classical tail bound, its sequential upgrade, and a betting interpretation that makes the upgrade constructive.

**Markov's inequality.** For a nonnegative random variable $W$ with $E[W] \le 1$,

$$\Pr(W \ge 1/\delta) \le \delta \, E[W] \le \delta.$$

Any such $W$ turns the event $\{W \ge 1/\delta\}$ into a test of level $\delta$.

**Ville's inequality.** The sequential upgrade. For a nonnegative supermartingale $\{W_t\}_{t \ge 0}$ with $W_0 = 1$ and $E[W_t \mid W_{t-1}] \le W_{t-1}$,

$$\Pr\!\left(\exists t \ge 1:\; W_t \ge 1/\delta\right) \le \delta.$$

The difference from Markov is essential: the probability bounded here is that of *ever* crossing $1/\delta$, over an unbounded horizon, instead of the probability at one fixed time. This is the anytime-valid property needed for a confidence sequence.

**The betting / wealth reading.** Ville's inequality becomes constructive once $W_t$ is read as the wealth of a gambler betting against $H_0$, the null hypothesis "no drift", starting with one unit of capital. Under $H_0$, the sequence $W_t$ is a supermartingale, so the gambler cannot expect to get rich: Ville's inequality caps how much luck they can have, over the whole sequence of bets. Under genuine drift, however, the bets are informative and tend to pay off, so the wealth grows; reaching $W_t \ge 1/\delta$ is therefore calibrated evidence of drift.

The **betting parameter** $\beta \in (0, 1)$ controls how aggressively the gambler bets at each step. For a fixed $\beta$, the wealth process takes the form derived in [[1](#ref-1)]:

$$W_t(\beta) = \exp\!\left(\beta \, S_t - \psi_E(\beta) \, V_t\right),$$

where:

- $S_t = \sum_{s=1}^{t} (\hat{R}_s - c_s)$ is the cumulative deviation of the per-batch estimates $\hat{R}_s$ from their **predictable centers** $c_s$,
- $V_t = \sum_{s=1}^{t} (\hat{R}_s - c_s)^2$ is the running empirical variance of those deviations,
- $\psi_E(\beta) = -\log(1 - \beta) - \beta$ is a cumulant-generating-function penalty, chosen so that $W_t(\beta)$ is a supermartingale under $H_0$.

Intuitively, $S_t$ is the deviation signal: the larger it grows, the stronger the case that the batches are drifting away from $H_0$. $V_t$ is the noise: it grows with the raw variability of the deviations regardless of any drift, and its role is to temper how much a given $S_t$ should be trusted. The betting parameter $\beta$ scales this signal-to-noise ratio in the exponent.

A **predictable center** $c_s$ is a quantity fixed before batch $s$ is observed, most naturally the running mean of the estimates from all previous batches, $c_s = \bar{R}_{s-1}$ for $s \ge 2$. Fixing $c_s$ this way makes the increment $\hat{R}_s - c_s$ conditionally mean-zero under $H_0$.

There is no previous batch before the first one, so $c_1$ must be seeded by a constant fixed in advance: any such constant preserves predictability, and therefore validity. Seeding with the threshold $\tau$ is a natural choice, since it is the null-hypothesized running risk at the decision boundary, making the sequence more sensitive to a drift starting right away. This seed only enters the first term of $V_t$, so its influence vanishes as batches accumulate: a poorly chosen seed can inflate $V_t$ and widen the early boundary, delaying an alarm, but it cannot create a false one.

**The method of mixtures.** The betting parameter that would extract the most evidence fastest depends on the (unknown) size of the drift and on $V_t$, so rather than commit to a single $\beta$, one averages the wealth process over a density $q(\beta)$ on $(0, 1)$:

$$W_t = \int_0^1 W_t(\beta) \, q(\beta) \, d\beta.$$

A mixture of nonnegative supermartingales, each starting at $1$, is itself a nonnegative supermartingale starting at $1$, so Ville's inequality applies for *any* choice of $q$; the density affects the tightness of the resulting bound but not its validity. A clever choice of *conjugate* mixture $q$ (Gaussian, Gamma, ...) can help tighten the bound at a target horizon [[3](#ref-3)]. For simplicity, one can take $q$ uniform on $(0, 1)$. This parameter-free choice stays valid regardless of horizon, at the price of being less tight than a correctly tuned conjugate mixture.

### The Empirical-Bernstein Boundary

Fix the variance process at a value $v$. The **boundary** is the largest cumulative deviation still consistent with $H_0$, expressed through the mixture wealth process $W(s, v)$ introduced above, now viewed as a function of a candidate deviation $s$ at fixed variance $v$:

$$u(v) = \sup\{\, s \ge 0 : W(s, v) \le 1/\delta \,\},$$

the right edge of the acceptance region, expressed as a deviation budget, where $\delta$ is the sequence's single false-alarm budget (its miscoverage). Under the uniform mixture, $W(s, v)$ takes the explicit form

$$W(s, v) = \int_0^1 \exp\!\left(\beta s - \psi_E(\beta) v\right) d\beta.$$

This function is continuous and strictly increasing in $s$: its derivative with respect to $s$ is strictly positive, and $W(s, v)$ rises from a value $\le 1$ at $s = 0$ to $\infty$ as $s \to \infty$. The supremum defining $u(v)$ is therefore attained at the unique root of

$$\int_0^1 \exp\!\left(\beta \, u(v) - \psi_E(\beta) \, v\right) d\beta = \frac{1}{\delta}.$$

The boundary is called "empirical-Bernstein" because its width adapts to the *observed* variability $V_t$ rather than assuming a fixed, known worst-case variance [[1](#ref-1)], which is what makes it tight in practice.

Dividing the deviation budget by the batch count converts it into a bound on the running mean. The anytime-valid lower bound on the running risk after $t$ batches is

$$L_t = \bar{R}_t - \frac{u(V_t)}{t}.$$

### The Alarm Rule and the Single Budget

**Alarm rule.** The user fixes a threshold $\tau$ in advance: the worst running risk they are willing to tolerate, in metric units. This is the criterion the whole construction has been building toward: at every batch $t$, a **drift alarm fires** as soon as the anytime-valid lower bound on the running risk crosses the threshold,

$$L_t > \tau.$$

By construction, $L_t$ satisfies the simultaneous coverage guarantee $\Pr(\forall t \ge 1 : \bar{R}_t \ge L_t) \ge 1 - \delta$, now applied to the running risk. Under no drift, $\bar{R}_t$ never exceeds $\tau$, so a false alarm, $L_t > \tau \ge \bar{R}_t$, can only occur when this coverage fails. The probability of *ever* raising a false alarm is therefore at most $\delta$, i.e. the sequence's miscoverage.

A smaller $\delta$ widens the confidence sequence and delays alarms, so the budget directly governs detection speed, not just the false-alarm rate.

Finally, a caveat on what this monitors: $\bar{R}_t$ averages over the entire accumulated history, which gives a long stable run inertia that a recent drift must overcome. An isolated spike, drowned in that history, barely moves $\bar{R}_t$ and is unlikely to raise an alarm; a sustained drift will eventually push $L_t$ above $\tau$, but only after a delay that grows with the length of the preceding stable history. Sensitivity to recent drift is recovered by restricting the running average to the most recent batches rather than the full history.

---

## Prediction-Powered Risk Monitoring (PPRM)

The human labels collected in a batch can be scarce, which limits how quickly a monitor based on them alone can react to real drift. **Prediction-Powered Risk Monitoring (PPRM)** [[2](#ref-2)] instead combines those human labels with a large pool of cheap proxy labels, the same way [Prediction-Powered Inference (PPI++)](estimators.md#prediction-powered-inference-ppi) does for one-off estimation. The anytime-valid guarantee derived above, the confidence sequence, Ville's inequality, the betting supermartingale, the empirical-Bernstein boundary, and the alarm rule, carries over unchanged: only the per-batch estimate $\hat{R}_s$ changes, now obtained from the PPI++ estimator instead of a plain sample mean of the batch's labels. That single change brings two additional requirements, addressed in turn below: the estimate must be renormalized onto $[0, 1]$, and its power-tuning weight must be predictable.

### Setting

Each batch $t$ now carries the same inputs as [Prediction-Powered Inference (PPI++)](estimators.md#prediction-powered-inference-ppi): a small set of human labels together with a larger set of proxy labels, both specific to that batch.

| Value | Present for | Description |
|---|---|---|
| $\tilde{Y}_{t,i}$ | All samples in batch $t$ | Proxy label |
| $Y_{t,j}$ | Labeled samples in batch $t$ only | Ground-truth label |

As in the previous section, we assume that all labels $\tilde{Y}_{t,i}$ and $Y_{t,j}$ are in $[0, 1]$.

The per-batch estimate $\hat{R}_s$ used in the running risk $\bar{R}_t$ (introduced above) is now the PPI++ estimate on batch $s$. Denoting $\tilde{Y}_s^{\bullet}$ and $\tilde{Y}_s^{\circ}$ the labeled and unlabeled proxies of batch $s$ respectively, with $n_s$ and $N_s$ their counts,

$$\hat{R}_s = \frac{1}{n_s}\sum_{j=1}^{n_s} Y_{s,j} + \lambda_s\left[\frac{1}{N_s}\sum_{i=1}^{N_s} \tilde{Y}_{s,i}^{\circ} - \frac{1}{n_s}\sum_{j=1}^{n_s} \tilde{Y}_{s,j}^{\bullet}\right],$$

which is the [Prediction-Powered Inference (PPI++)](estimators.md#prediction-powered-inference-ppi) mean estimator applied within batch $s$, using its own power-tuning weight $\lambda_s$ (see [Predictable power-tuning](#predictable-power-tuning) below). Unlike a plain sample mean, $\hat{R}_s$ is not guaranteed to fall in $[0, 1]$, which the next section addresses.

### Normalization onto the Unit Interval

The empirical-Bernstein boundary derived above is valid only for variables confined to a *known* bounded range. A prediction-powered estimate of a metric in $[0, 1]$ does not itself lie in $[0, 1]$: the power-tuning weight $\lambda$ is a variance-minimizing regression slope which is not confined to $[0, 1]$, so an under-dispersed or anti-correlated proxy can push $\lambda$, and with it the estimate, outside any fixed interval.

Clipping the power-tuning weight to $[0, \lambda_{\max}]$ confines a $[0, 1]$-metric estimate $\hat{R}$ to the range $[-\lambda_{\max},\, 1 + \lambda_{\max}]$. The affine map

$$\frac{\hat{R} + \lambda_{\max}}{1 + 2\lambda_{\max}}$$

maps that range back onto $[0, 1]$, producing the normalized estimate that plays the role directly held by $\hat{R}_s$ in the classical section.

### Predictable Power-Tuning

The power-tuning weight $\lambda_t$ used to form the estimate $\hat{R}_t$ on batch $t$ must itself be **predictable**: computed only from batches strictly earlier than $t$, never from batch $t$ itself. This is what makes each per-batch increment conditionally mean-zero under $H_0$, which is the condition that keeps $W_t$ a supermartingale.

Predictability is the only validity constraint here. *Which* prior batches are pooled to compute $\lambda_t$ is purely a matter of statistical power, and any predictable choice remains valid; only the width of the resulting boundary changes. The simplest option is to pool the full prior history of batches to compute $\lambda_t$. The first batch has no predecessor, so it can use the neutral weight $\lambda_1 = 1$, which is itself trivially predictable since it depends on no data at all.

**Alarm rule.** With both additions in place, $\hat{R}_t$ satisfies the same assumptions used in [Classical risk monitoring](#classical-risk-monitoring), so the same criterion still decides when to raise a **drift alarm**,

$$L_t > \tau,$$

now backed by the more sample-efficient PPI++ estimate, at the same single false-alarm budget $\delta$.

---

## References

<a id="ref-1"></a>[1] <a id="ref-1-link" href="https://academic.oup.com/jrsssb/article/86/1/1/7043257">Waudby-Smith, Ian, and Aaditya Ramdas. "Estimating means of bounded random variables by betting." Journal of the Royal Statistical Society Series B: Statistical Methodology 86, no. 1 (2024): 1-27.</a>.

<a id="ref-2"></a>[2] <a id="ref-2-link" href="https://arxiv.org/abs/2602.02229">Zhang, Guangyi, Yunlong Cai, Guanding Yu, and Osvaldo Simeone. "Prediction-Powered Risk Monitoring of Deployed Models for Detecting Harmful Distribution Shifts." arXiv preprint arXiv:2602.02229 (2026).</a>.

<a id="ref-3"></a>[3] <a id="ref-3-link" href="https://arxiv.org/abs/1810.08240">Howard, Steven R., Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon. "Time-uniform, nonparametric, nonasymptotic confidence sequences." The Annals of Statistics 49, no. 2 (2021): 1055-1080.</a>.