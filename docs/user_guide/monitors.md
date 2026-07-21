# Monitors

Once a metric has been estimated validating an AI system's deployment, the question shifts from "what is the metric today?" to "has the metric drifted?" GLIDE's monitors answer this with **anytime-valid drift monitoring**: a sequential procedure that watches a metric across successive batches of production data and raises an alarm the moment there is statistically valid evidence that it has crossed a threshold.

---

## The Monitoring Problem

After deployment, the metric is re-estimated on successive batches of production data $t = 1, 2, \dots$, for instance a fresh weekly batch with new labeled samples. Each batch $t$ yields an estimate $\hat{\theta}_t$. The question is whether the sequence proves that the metric has drifted past a threshold $\tau$ that the user fixes in advance: the worst running value they are willing to tolerate.

The naive approach is statistically invalid. Comparing each $\hat{\theta}_t$ to $\tau$ with an ordinary confidence interval at level $1 - \alpha$ carries its own false-alarm probability $\alpha$ for that single comparison. Checking after every batch means accumulating many such chances to be wrong. More precisely, the probability that at least one confidence interval misses the true value after $t$ tests is $1-(1-\alpha)^t$, so a false alarm becomes almost certain over a long enough horizon. This is a form of **multiple testing**, here called **peeking**, and it is exactly what invalidates repeated per-batch significance testing.

The fix is a **confidence sequence**: a sequence of intervals $\{C_t\}_{t \ge 1}$ that covers the target *simultaneously at all times*,

$$\Pr\!\left(\forall t \ge 1:\; \bar{\theta}_t \in C_t\right) \ge 1 - \delta,$$

so the user may look after every batch and the total false-alarm probability over the entire monitoring horizon still stays below the single budget $\delta$. This **anytime-valid** guarantee is what makes peeking safe. It stands in contrast to fixed-sample confidence intervals which are valid only at a single, pre-committed sample size and lose their guarantee the moment they are checked repeatedly.

The following section presents anytime-valid constructions allowing to build confidence sequences. These will be leveraged further in this page to devise various risk monitoring methods.

---

## Confidence Sequences

### Empirical-Bernstein Confidence Sequences

One can build an anytime-valid lower bound on the running mean of any sequence of per-batch risk estimates $\hat{R}_s$ known to lie in $[0, 1]$, using only that boundedness. The running risk after $t$ batches is the average of the per-batch estimates,

$$\bar{R}_t = \frac{1}{t} \sum_{s=1}^{t} \hat{R}_s,$$

monitored against a user-fixed threshold $\tau$. The construction below does not depend on how $\hat{R}_s$ itself is computed.

#### From Markov to Ville: the Anytime-Valid Guarantee

The anytime-valid guarantee is built from first principles, in three steps: a classical tail bound, its sequential upgrade, and a betting interpretation that makes the upgrade constructive.

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

A mixture of nonnegative supermartingales, each starting at $1$, is itself a nonnegative supermartingale starting at $1$, so Ville's inequality applies for *any* choice of $q$; the density affects the tightness of the resulting bound but not its validity. A clever choice of *conjugate* mixture $q$ (Gaussian, Gamma, ...) can help tighten the bound at a target horizon [[2](#ref-2)]. For simplicity, one can take $q$ uniform on $(0, 1)$. This parameter-free choice stays valid regardless of horizon, at the price of being less tight than a correctly tuned conjugate mixture.

#### The Empirical-Bernstein Boundary

Fix the value of the variance process at $v$. The **boundary** is the largest cumulative deviation still consistent with $H_0$, expressed through the mixture wealth process $W(s, v)$ introduced above, now viewed as a function of a candidate deviation $s$ at fixed variance $v$:

$$u(v) = \sup\{\, s \ge 0 : W(s, v) \le 1/\delta \,\},$$

the right edge of the acceptance region, expressed as a deviation budget, where $\delta$ is the sequence's single false-alarm budget (its miscoverage). Under the uniform mixture, $W(s, v)$ takes the explicit form

$$W(s, v) = \int_0^1 \exp\!\left(\beta s - \psi_E(\beta) v\right) d\beta.$$

Given the choice of $\psi_E(\beta)$, this integral can be written in closed form via the following derivation

$$
\begin{align*}
    \int_0^1 &\exp(\beta s + (\log(1 - \beta) + \beta) v) d\beta \\
    &= \int_0^1 \exp((s + v) \beta)(1 - \beta)^v d\beta \\
    &= \int_0^1 \exp((s + v)(1 - \beta)) \beta^v d\beta \\
    &= \exp(s + v)\int_0^1 \exp(-\beta(s + v)) \beta^v d\beta \\
    &= \frac{\exp(s + v)}{(s + v)^{v + 1}} \underbrace{\int_0^{s + v} e^{-\beta} \beta^{(v + 1) - 1} d\beta}_{\Gamma(v + 1, s + v)}
\end{align*}
$$

Where $\Gamma(z, x) = \int_0^{x} e^{-t}t^{z-1} dt$ is the partial $\Gamma$ function. As a result, $W(s, v)$ is continuous and strictly increasing in $s$: its derivative with respect to $s$ is strictly positive, and it rises from a value $\le 1$ at $s = 0$ to $\infty$ as $s \to \infty$. The supremum defining $u(v)$ is therefore attained at the unique root of

$$\frac{\exp(s+v)}{(s+v)^{v+1}}\Gamma(v+1, s+v) = \frac{1}{\delta}.$$

The boundary is called "empirical-Bernstein" because its width adapts to the *observed* variability $V_t$ rather than assuming a fixed, known worst-case variance [[1](#ref-1)], which is what makes it tight in practice.

Dividing the deviation budget by the batch count converts it into a bound on the running mean. The anytime-valid lower bound on the running risk after $t$ batches is

$$L_t = \bar{R}_t - \frac{u(V_t)}{t}.$$

#### The Alarm Rule and the Single Budget

**Alarm rule.** The user fixes a threshold $\tau$ in advance: the worst running risk they are willing to tolerate, in metric units. A **drift alarm fires** as soon as the anytime-valid lower bound on the running risk crosses the threshold,

$$L_t > \tau.$$

By construction, $L_t$ satisfies the simultaneous coverage guarantee $\Pr(\forall t \ge 1 : \bar{R}_t \ge L_t) \ge 1 - \delta$. Under no drift, $\bar{R}_t$ never exceeds $\tau$, so a false alarm, $L_t > \tau \ge \bar{R}_t$, can only occur when this coverage fails. The probability of *ever* raising a false alarm is therefore at most $\delta$, i.e. the sequence's miscoverage.

A smaller $\delta$ widens the confidence sequence and delays alarms, so the budget directly governs detection speed, not just the false-alarm rate.

Finally, a caveat on what this monitors: $\bar{R}_t$ averages over the entire accumulated history, which gives a long stable run inertia that a recent drift must overcome. An isolated spike, drowned in that history, barely moves $\bar{R}_t$ and is unlikely to raise an alarm; a sustained drift will eventually push $L_t$ above $\tau$, but only after a delay that grows with the length of the preceding stable history. Sensitivity to recent drift is recovered by restricting the running average to the most recent batches rather than the full history.

### Asymptotic Confidence Sequences

This time, the confidence sequence is built by exploiting the variance of each batch estimate, rather than only the fact that it lies in $[0, 1]$.

#### Setting

Each batch $t$ contributes a per-batch estimate $\hat{R}_t$ together with its own standard error $\hat{\sigma}_t$. This applies, for example, when $\hat{R}_t$ is a sample mean of the batch's labeled values, with standard error computed according to the Central Limit Theorem. As before, the running risk after $t$ batches is $\bar{R}_t = \frac{1}{t}\sum_{s=1}^{t}\hat{R}_s$, monitored against a user-fixed threshold $\tau$, and each $\hat{R}_s$ is assumed to be approximately Gaussian around the batch's own risk.

Define the **intrinsic time**

$$\nu_t = \sum_{s \le t} \hat{\sigma}_s^2,$$

the accumulated variances of the per-batch estimates.

#### From a Gaussian Wealth Process to an Anytime-Valid Bound

The anytime-valid guarantee rests on the same two classical facts used throughout this guide. Markov's inequality states that a nonnegative random variable $W$ with $E[W] \le 1$ satisfies $\Pr(W \ge 1/\delta) \le \delta \, E[W] \le \delta$. Ville's inequality upgrades this to a nonnegative supermartingale $\{W_t\}_{t \ge 0}$ with $W_0 = 1$ yielding: $\Pr(\exists t \ge 1 : W_t \ge 1/\delta) \le \delta$, bounding the probability of *ever* crossing $1/\delta$ rather than the probability at one fixed time.

Read $W_t$ as the wealth of a gambler betting against $H_0$, the null hypothesis of no drift, starting with one unit of capital. For a betting rate $\lambda \in \mathbb{R}$, define

$$W_t(\lambda) = \exp\!\left(\lambda S_t - \frac{\lambda^2}{2}\nu_t\right),$$

where $S_t = \sum_{s \le t}(\hat{R}_s - c_s)$ is the cumulative deviation of the per-batch estimates from their predictable centers $c_s = \bar{R}_{s-1}$ (with $c_1 = \tau$ for the first batch). If the deviations $\hat{R}_s - c_s$ were exactly $\mathcal{N}(0, \hat{\sigma}_s^2)$-distributed, $W_t(\lambda)$ would be a supermartingale for every $\lambda$.

Rather than commit to one $\lambda$, mix over a **folded Gaussian** density of scale $\rho$: a Gaussian density restricted to $\lambda > 0$ and doubled so that it integrates to one. This is the natural conjugate mixing distribution for Gaussian increments [[5](#ref-5)]:

$$W_t = \int_0^\infty W_t(\lambda) \cdot \frac{2}{\sqrt{2\pi\rho^2}}\exp\!\left(-\frac{\lambda^2}{2\rho^2}\right) d\lambda.$$

A mixture of nonnegative supermartingales, each starting at $1$, is itself a nonnegative supermartingale starting at $1$, so Ville's inequality applies for any choice of $\rho > 0$; the scale determines the tightness of downstream bounds. Carrying out the Gaussian integral and inverting $W_t \ge 1/\delta$ for the largest deviation still consistent with $H_0$, the same "boundary" idea used elsewhere in this guide but solvable here in closed form, gives the anytime-valid lower bound on the running risk after $t$ batches,

$$L_t = \bar{R}_t - \sqrt{ \frac{2(\nu_t \rho^2 + 1)}{t^2 \rho^2} \log\!\left(1 + \frac{\sqrt{\nu_t \rho^2 + 1}}{2\delta}\right) }.$$

This bound is exact if the per-batch deviations are truly Gaussian. In practice they are only approximately so that a further argument with a strong approximation is needed (see [[4](#ref-4)] for details). It shows that the partial sums of per-batch estimates stay close to those of a genuinely Gaussian process so that the same boundary remains valid for a sufficient number of batches. This makes the guarantee asymptotic rather than exact.

#### Tuning and Interpreting the Boundary

No anytime-valid boundary can be tight at every batch: tightening it at one horizon necessarily loosens it at others, so the scale $\rho$ controls where that tightness is spent. Choosing

$$\rho^2 = \frac{-2\log(2\delta) + \log\bigl(-2\log(2\delta) + 1\bigr)}{\nu_{t^\star}}$$

makes $L_t$ tightest at a user-chosen target batch $t^\star$. Note, however, that the penalty for choosing a different target than needed is generally mild in practice.

The above boundary's width scales as $\sqrt{\nu_t \log \nu_t}/t$, shrinking with the actual precision of the per-batch estimates. Moreover, since $\nu_t$ accumulates each batch's own internal standard error rather than deviations between batches, a shift in the risk from one batch to the next only affects $\bar{R}_t$ and not $\nu_t$: the boundary is immune to drift-inflation.

#### The Alarm Rule and the Asymptotic Guarantee

A drift alarm fires the moment the anytime-valid lower bound on the running risk crosses the user-fixed threshold,

$$L_t > \tau,$$

at the same single false-alarm budget $\delta$: the probability of ever raising a false alarm this way is at most $\delta$ indepndently of the number of checked batches. A smaller $\delta$ widens the confidence sequence and delays alarms, so the budget influences both the detection speed and the false-alarm rate.

Note that the monitored quantity $\bar{R}_t$ averages over the entire accumulated history, which gives a long stable run inertia that a recent drift must overcome. An isolated spike, drowned in that history, barely moves $\bar{R}_t$ and is unlikely to raise an alarm; a sustained drift will eventually push $L_t$ above $\tau$, but only after a delay that grows with the length of the preceding stable history. 

Due to the previously mentioned asymptotic trait of this guarantee, it holds in the limit of enough batches with consistently estimated standard errors. This is because the false-alarm probability converges to at most $\delta$ as the horizon grows rather than being equal to $\delta$ from the start. This means the budget may be transiently exceeded over the first few batches but is rarely observed.

---

## Risk Monitoring

### Empirical Classical Risk Monitoring

The empirical Bernstein confidence sequence above is used here in the setting where the per-batch estimate comes directly from a batch of human-labeled samples.

#### Setting

Each batch $t$ contributes a set of human labels.

| Value | Present for | Description |
|---|---|---|
| $Y_{t,j}$ | All labeled samples in batch $t$ | Ground-truth label |

All labels $Y_{t,j}$ are assumed to lie in $[0, 1]$. Every batch is monitored relative to a user-fixed threshold $\tau$. The metric is treated as a **risk** $R$, where lower is better (for example an error rate); a performance metric, where higher is better (for example accuracy), is monitored by applying the same methodology to $1 - R$ instead of $R$. The metric only needs to be **bounded** within some known range so that it can be affinely normalized onto $[0, 1]$ without compromising the statistical guarantees.

The per-batch risk estimate is the classical sample mean of the $n_s$ labels in batch $s$ alone,

$$\hat{R}_s = \frac{1}{n_s}\sum_{j=1}^{n_s} Y_{s,j}.$$

#### Bound and Alarm Rule

Plugging this $\hat{R}_s$ into the [Empirical-Bernstein Confidence Sequences](#empirical-bernstein-confidence-sequences) construction above gives the anytime-valid lower bound on the running risk after $t$ batches,

$$L_t = \bar{R}_t - \frac{u(V_t)}{t},$$

and a **drift alarm fires** as soon as $L_t > \tau$.

### Empirical Prediction-Powered Risk Monitoring (Empirical PPRM)

The human labels collected in a batch can be scarce, which limits how quickly a monitor based on them alone can react to real drift. **Prediction-Powered Risk Monitoring (PPRM)** [[3](#ref-3)] instead combines those human labels with a large pool of cheap proxy labels, the same way [Prediction-Powered Inference (PPI++)](estimators.md#prediction-powered-inference-ppi) does for one-off estimation.

#### Setting

Each batch $t$ carries the following inputs: a small set of human labels together with a larger set of proxy labels, both specific to that batch.

| Value | Present for | Description |
|---|---|---|
| $\tilde{Y}_{t,i}$ | All samples in batch $t$ | Proxy label |
| $Y_{t,j}$ | Labeled samples in batch $t$ only | Ground-truth label |

All labels $\tilde{Y}_{t,i}$ and $Y_{t,j}$ are assumed to lie in $[0, 1]$.

The per-batch estimate $\hat{R}_s$ is the PPI++ estimate on batch $s$. Denoting $\tilde{Y}_s^{\bullet}$ and $\tilde{Y}_s^{\circ}$ the labeled and unlabeled proxies of batch $s$ respectively, with $n_s$ and $N_s$ their respective counts,

$$\hat{R}_s = \frac{1}{n_s}\sum_{j=1}^{n_s} Y_{s,j} + \lambda_s\left[\frac{1}{N_s}\sum_{i=1}^{N_s} \tilde{Y}_{s,i}^{\circ} - \frac{1}{n_s}\sum_{j=1}^{n_s} \tilde{Y}_{s,j}^{\bullet}\right],$$

with $\lambda_s$ a predictable power-tuning weight defined further below. 

Unlike a plain sample mean, $\hat{R}_s$ is not guaranteed to fall in $[0, 1]$ which is necessary for the empirical-Bernstein boundary to be valid. The power-tuning weight $\lambda_s$ is a variance-minimizing regression slope not confined to $[0, 1]$, so an under-dispersed or anti-correlated proxy can push $\lambda_s$, and with it the estimate, outside any fixed interval. 

This is fixed by clipping $\hat{R}_s$ to the $[0, 1]$ interval before it is plugged into the empirical Bernstein confidence sequence.

#### Predictable Power-Tuning

The power-tuning weight $\lambda_t$ used to form the estimate $\hat{R}_t$ on batch $t$ must itself be **predictable**: computed only from batches strictly earlier than $t$.

To compute predictable weights $\lambda_t$, the simplest option is to pool the full prior history of batches. The first batch has no predecessor, so it can use the neutral weight $\lambda_1 = 1$, which is itself trivially predictable.

#### Bound and Alarm Rule

With both additions in place, plugging the clipped $\hat{R}_t$ into the [Empirical-Bernstein Confidence Sequences](#empirical-bernstein-confidence-sequences) construction gives the anytime-valid lower bound $L_t = \bar{R}_t - u(V_t)/t$ defining the drift alarm rule as,

$$L_t > \tau,$$

now backed by the more sample-efficient PPI++ estimate, at the same single false-alarm budget $\delta$.

### Asymptotic Classical Risk Monitoring

The asymptotic confidence sequence above is applied here in the setting where the per-batch estimate comes from a batch of human-labeled samples.

#### Setting

Each batch $t$ contributes a set of human labels.

| Value | Present for | Description |
|---|---|---|
| $Y_{t,j}$ | All labeled samples in batch $t$ | Ground-truth label |

Every batch is monitored relative to a user-fixed threshold $\tau$ and the metric is treated as a **risk** $R$, where lower is better; a performance metric is monitored by applying the same methodology to $1 - R$ instead of $R$.

The per-batch risk estimate is the classical sample mean of the $n_s$ labels in batch $s$,

$$\hat{R}_s = \frac{1}{n_s}\sum_{j=1}^{n_s} Y_{s,j},$$

together with its standard error,

$$\hat{\sigma}_s = \sqrt{\frac{\widehat{\mathrm{Var}}(Y_s)}{n_s}}.$$

#### Bound and Alarm Rule

Plugging $\hat{R}_s$ and $\hat{\sigma}_s$ into the [Asymptotic Confidence Sequences](#asymptotic-confidence-sequences) gives the anytime-valid lower bound on the running risk,

$$L_t = \bar{R}_t - \sqrt{ \frac{2(\nu_t \rho^2 + 1)}{t^2 \rho^2} \log\!\left(1 + \frac{\sqrt{\nu_t \rho^2 + 1}}{2\delta}\right) },$$

with $\nu_t = \sum_{s \le t}\hat{\sigma}_s^2$ computed from these classical per-batch standard errors, and a **drift alarm fires** as soon as $L_t > \tau$, the user-fixed threshold.

### Asymptotic Prediction-Powered Risk Monitoring (Asymptotic PPRM)

This method combines scarce human labels with a large pool of cheap proxy labels, the same way [Prediction-Powered Inference (PPI++)](estimators.md#prediction-powered-inference-ppi) does for one-off estimation. Here, however, the per-batch estimate is plugged into the [Asymptotic Confidence Sequences](#asymptotic-confidence-sequences) construction instead, using the estimate's standard error.

#### Setting

Each batch $t$ carries as inputs: a small set of human labels together with a larger set of proxy labels, both specific to that batch.

| Value | Present for | Description |
|---|---|---|
| $\tilde{Y}_{t,i}$ | All samples in batch $t$ | Proxy label |
| $Y_{t,j}$ | Labeled samples in batch $t$ only | Ground-truth label |

The per-batch estimate $\hat{R}_s$ is the PPI++ estimate on batch $s$ as in Empirical PPRM together with its standard error. Unlike Empirical PPRM, $\hat{R}_s$ is used directly here, without clipping onto $[0, 1]$: the asymptotic construction does not require the estimate itself to lie in a bounded range, only that it is approximately Gaussian with a known standard error.

#### Predictable Power-Tuning

As in Empirical PPRM, the power-tuning weight $\lambda_t$ used to form $\hat{R}_t$ must be **predictable**: computed only from batches strictly earlier than $t$. The power-tuning factors are therefore computed using strictly earlier batches. The first batch has no predecessor, so it uses the neutral weight $\lambda_1 = 1$.

#### Bound and Alarm Rule

Plugging $\hat{R}_s$ and $\hat{\sigma}_s$ into the [Asymptotic Confidence Sequences](#asymptotic-confidence-sequences) construction gives the anytime-valid lower bound on the running risk,

$$L_t = \bar{R}_t - \sqrt{ \frac{2(\nu_t \rho^2 + 1)}{t^2 \rho^2} \log\!\left(1 + \frac{\sqrt{\nu_t \rho^2 + 1}}{2\delta}\right) },$$

now backed by the more sample-efficient PPI++ estimate, and a **drift alarm fires** as soon as $L_t > \tau$, the user-fixed threshold.

---

## References

<a id="ref-1"></a>[1] <a id="ref-1-link" href="https://academic.oup.com/jrsssb/article/86/1/1/7043257">Waudby-Smith, Ian, and Aaditya Ramdas. "Estimating means of bounded random variables by betting." Journal of the Royal Statistical Society Series B: Statistical Methodology 86, no. 1 (2024): 1-27.</a>.

<a id="ref-2"></a>[2] <a id="ref-2-link" href="https://projecteuclid.org/journals/The-Annals-of-Statistics/volume-49/issue-2/Time-uniform-nonparametric-nonasymptotic-confidence-sequences/10.1214/20-AOS1991.full">Howard, Steven R., Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon. "Time-uniform, nonparametric, nonasymptotic confidence sequences." The Annals of Statistics 49, no. 2 (2021): 1055-1080.</a>.

<a id="ref-3"></a>[3] <a id="ref-3-link" href="https://arxiv.org/abs/2602.02229">Zhang, Guangyi, Yunlong Cai, Guanding Yu, and Osvaldo Simeone. "Prediction-Powered Risk Monitoring of Deployed Models for Detecting Harmful Distribution Shifts." arXiv preprint arXiv:2602.02229 (2026).</a>.

<a id="ref-4"></a>[4] <a id="ref-4-link" href="https://doi.org/10.1214/24-AOS2408">Waudby-Smith, Ian, David Arbour, Ritwik Sinha, Edward H. Kennedy, and Aaditya Ramdas. "Time-uniform central limit theory and asymptotic confidence sequences." The Annals of Statistics 52, no. 6 (2024): 2613-2640.</a>.

<a id="ref-5"></a>[5] <a id="ref-5-link" href="https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-41/issue-5/Statistical-Methods-Related-to-the-Law-of-the-Iterated-Logarithm/10.1214/aoms/1177696786.full">Robbins, Herbert. "Statistical methods related to the law of the iterated logarithm." The Annals of Mathematical Statistics 41, no. 5 (1970): 1397-1409.</a>.
