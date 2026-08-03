# Monitors

Once a metric has been estimated validating an AI system's deployment, the question shifts from "what is the metric today?" to "has the metric drifted?" GLIDE's monitors answer this with **anytime-valid drift monitoring**: a sequential procedure that watches a metric across successive batches of production data and raises an alarm the moment there is statistically valid evidence that it has crossed a threshold.

---

## The Monitoring Problem

After deployment, the metric is re-estimated on successive batches of production data $t = 1, 2, \dots$, for instance a fresh weekly batch with new labeled samples. Each batch $t$ yields an estimate $\hat{\theta}_t$. The question is whether the sequence proves that the metric has drifted past a threshold $\tau$ that the user fixes in advance: the worst running value they are willing to tolerate.

The naive approach is statistically invalid. Comparing each $\hat{\theta}_t$ to $\tau$ with an ordinary confidence interval at level $1 - \alpha$ carries its own false-alarm probability $\alpha$ for that single comparison. Checking after every batch means accumulating many such chances to be wrong. More precisely, the probability that at least one confidence interval misses the true value after $t$ tests is $1-(1-\alpha)^t$, so a false alarm becomes almost certain over a long enough horizon. This is a form of **multiple testing**, here called **peeking**, and it is exactly what invalidates repeated per-batch significance testing.

The fix is a **confidence sequence**: a sequence of intervals $\{C_t\}_{t \ge 1}$ that covers the target *simultaneously at all times*,

$$\Pr\!\left(\forall t \ge 1:\; \bar{\theta}_t \in C_t\right) \ge 1 - \delta,$$

so the user may look after every batch and the total false-alarm probability over the entire monitoring horizon still stays below the single budget $\delta$. This **anytime-valid** guarantee is what makes peeking safe. It stands in contrast to fixed-sample confidence intervals which are valid only at a single, pre-committed sample size and lose their guarantee the moment they are checked repeatedly.

The following section presents an anytime-valid construction allowing to build confidence sequences. It will be leveraged further in this page to devise risk monitoring methods.

---

## Confidence Sequences

### From Markov to Ville: the Anytime-Valid Guarantee

The anytime-valid guarantee running through the confidence sequence in this guide rests on two classical facts: a tail bound, and its sequential upgrade.

**Markov's inequality.** For a nonnegative random variable $W$ with $E[W] \le 1$,

$$\Pr(W \ge 1/\delta) \le \delta \, E[W] \le \delta.$$

Any such $W$ turns the event $\{W \ge 1/\delta\}$ into a test of level $\delta$.

**Ville's inequality.** The sequential upgrade. For a nonnegative supermartingale $\{W_t\}_{t \ge 0}$ with $W_0 = 1$ and $E[W_t \mid W_{t-1}] \le W_{t-1}$,

$$\Pr\!\left(\exists t \ge 1:\; W_t \ge 1/\delta\right) \le \delta.$$

The difference from Markov is essential: the probability bounded here is that of *ever* crossing $1/\delta$, over an unbounded horizon, instead of the probability at one fixed time. This is the anytime-valid property needed for a confidence sequence.

**The betting / wealth reading.** Ville's inequality becomes constructive once $W_t$ is read as the wealth of a gambler betting against $H_0$, the null hypothesis "no drift", starting with one unit of capital. Under $H_0$, the sequence $W_t$ is a supermartingale, so the gambler cannot expect to get rich: Ville's inequality caps how much luck they can have, over the whole sequence of bets. Under genuine drift, however, the bets are informative and tend to pay off, so the wealth grows; reaching $W_t \ge 1/\delta$ is therefore calibrated evidence of drift. The construction below defines its own wealth process realizing this reading.

### Asymptotic Confidence Sequences

The confidence sequence is built by exploiting the variance of each batch estimate.

#### Setting

Each batch $t$ contributes a per-batch estimate $\hat{R}_t$ together with its own standard error $\hat{\sigma}_t$. This applies, for example, when $\hat{R}_t$ is a sample mean of the batch's labeled values, with standard error computed according to the Central Limit Theorem. The running risk after $t$ batches is $\bar{R}_t = \frac{1}{t}\sum_{s=1}^{t}\hat{R}_s$, monitored against a user-fixed threshold $\tau$, and each $\hat{R}_s$ is assumed to be approximately Gaussian around the batch's own risk.

Define the **intrinsic time**

$$\nu_t = \sum_{s \le t} \hat{\sigma}_s^2,$$

the accumulated variances of the per-batch estimates.

#### The Gaussian Wealth Process

Let $W_t$ represent the wealth of a gambler betting against $H_0$, the null hypothesis of no drift, starting with one unit of capital. For a betting rate $\lambda \in \mathbb{R}$, define

$$W_t(\lambda) = \exp\!\left(\lambda S_t - \frac{\lambda^2}{2}\nu_t\right),$$

where $S_t = \sum_{s \le t}(\hat{R}_s - c_s)$ is the cumulative deviation of the per-batch estimates from their predictable centers $c_s = \bar{R}_{s-1}$ (with $c_1 = \tau$ for the first batch). If the deviations $\hat{R}_s - c_s$ were exactly $\mathcal{N}(0, \hat{\sigma}_s^2)$-distributed, $W_t(\lambda)$ would be a supermartingale for every $\lambda$.

Rather than commit to one $\lambda$, mix over a **folded Gaussian** density of scale $\rho$: a Gaussian density restricted to $\lambda > 0$ and doubled so that it integrates to one. This is the natural conjugate mixing distribution for Gaussian increments [[3](#ref-3)]:

$$W_t = \int_0^\infty W_t(\lambda) \cdot \frac{2}{\sqrt{2\pi\rho^2}}\exp\!\left(-\frac{\lambda^2}{2\rho^2}\right) d\lambda.$$

A mixture of nonnegative supermartingales, each starting at $1$, is itself a nonnegative supermartingale starting at $1$, so Ville's inequality applies for any choice of $\rho > 0$; the scale determines the tightness of downstream bounds. Carrying out the Gaussian integral and inverting $W_t \ge 1/\delta$ for the largest deviation still consistent with $H_0$, we can obtain the anytime-valid lower bound on the running risk after $t$ batches with a closed form (see [[2](#ref-2), Proposition B.2]),

$$L_t = \bar{R}_t - \sqrt{ \frac{2(\nu_t \rho^2 + 1)}{t^2 \rho^2} \log\!\left(1 + \frac{\sqrt{\nu_t \rho^2 + 1}}{2\delta}\right) }.$$

This bound is exact if the per-batch deviations are truly Gaussian. In practice they are only approximately so that a further argument with a strong approximation is needed (see [[2](#ref-2)] for details). It shows that the partial sums of per-batch estimates stay close to those of a genuinely Gaussian process so that the same boundary remains valid for a sufficient number of batches. This makes the guarantee asymptotic rather than exact.

#### Tuning and Interpreting the Boundary

No anytime-valid boundary can be tight at every batch: tightening it at one horizon necessarily loosens it at others, so the scale $\rho$ controls where that tightness is spent. Choosing

$$\rho^2 = \frac{-2\log(2\delta) + \log\bigl(-2\log(2\delta) + 1\bigr)}{\nu_{t^\star}}$$

(see [[2](#ref-2), Equation (50)], used here with a doubled miscoverage $2\delta$ since this is a one-sided rather than two-sided confidence sequence) makes $L_t$ tightest at a user-chosen target batch $t^\star$. Note, however, that the penalty for choosing a different target than needed is generally mild in practice.

The above boundary's width scales as $\sqrt{\nu_t \log \nu_t}/t$, shrinking with the actual precision of the per-batch estimates.

#### The Alarm Rule and the Asymptotic Guarantee

A drift alarm fires the moment the anytime-valid lower bound on the running risk crosses the user-fixed threshold,

$$L_t > \tau,$$

at the same single false-alarm budget $\delta$: the probability of ever raising a false alarm this way is at most $\delta$ independently of the number of checked batches. A smaller $\delta$ widens the confidence sequence and delays alarms, so the budget influences both the detection speed and the false-alarm rate.

A caveat on what this monitors: the monitored quantity $\bar{R}_t$ averages over the entire accumulated history, which gives a long stable run inertia that a recent drift must overcome. An isolated spike, drowned in that history, barely moves $\bar{R}_t$ and is unlikely to raise an alarm; a sustained drift will eventually push $L_t$ above $\tau$, but only after a delay that grows with the length of the preceding stable history. Sensitivity to recent drift is recovered by restricting the running average to the most recent batches rather than the full history.

---

## Risk Monitoring

### Asymptotic Classical Risk Monitoring

The asymptotic confidence sequence above is applied here in the setting where the per-batch estimate comes from a batch of human-labeled samples.

#### Setting

Each batch $t$ contributes a set of human labels.

| Value | Present for | Description |
|---|---|---|
| $Y_{t,j}$ | All labeled samples in batch $t$ | Ground-truth label |

Every batch is monitored relative to a user-fixed threshold $\tau$ and the metric is treated as a **risk** $R$, where lower is better (for example an error rate); a performance metric, where higher is better (for example accuracy), is monitored by applying the same methodology to $1 - R$ instead of $R$.

The per-batch risk estimate is the classical sample mean of the $n_s$ labels in batch $s$,

$$\hat{R}_s = \frac{1}{n_s}\sum_{j=1}^{n_s} Y_{s,j},$$

together with its standard error,

$$\hat{\sigma}_s = \sqrt{\frac{\widehat{\mathrm{Var}}(Y_s)}{n_s}}.$$

#### Bound and Alarm Rule

Plugging the values $\hat{R}_s$ and $\hat{\sigma}_s$ into the [Asymptotic Confidence Sequences](#asymptotic-confidence-sequences) gives the anytime-valid lower bound $L_t$ derived there. The **drift alarm** fires as soon as $L_t > \tau$, the user-fixed threshold.

### Asymptotic Prediction-Powered Risk Monitoring (Asymptotic PPRM)

The human labels collected in a batch can be scarce, which limits how quickly a monitor based on them alone can react to real drift. **Prediction-Powered Risk Monitoring (PPRM)** [[1](#ref-1)] instead combines those human labels with a large pool of cheap proxy labels, the same way [Prediction-Powered Inference (PPI++)](estimators.md#prediction-powered-inference-ppi) does for one-off estimation. Here, the per-batch estimate is plugged into the [Asymptotic Confidence Sequences](#asymptotic-confidence-sequences) construction, using the estimate's standard error.

#### Setting

Each batch $t$ carries the following inputs: a small set of human labels together with a larger set of proxy labels, both specific to that batch.

| Value | Present for | Description |
|---|---|---|
| $\tilde{Y}_{t,i}$ | All samples in batch $t$ | Proxy label |
| $Y_{t,j}$ | Labeled samples in batch $t$ only | Ground-truth label |

The per-batch estimate $\hat{R}_s$ is the PPI++ estimate on batch $s$. Denoting $\tilde{Y}_s^{\bullet}$ and $\tilde{Y}_s^{\circ}$ the labeled and unlabeled proxies of batch $s$ respectively, with $n_s$ and $N_s$ their respective counts,

$$\hat{R}_s = \frac{1}{n_s}\sum_{j=1}^{n_s} Y_{s,j} + \lambda_s\left[\frac{1}{N_s}\sum_{i=1}^{N_s} \tilde{Y}_{s,i}^{\circ} - \frac{1}{n_s}\sum_{j=1}^{n_s} \tilde{Y}_{s,j}^{\bullet}\right],$$

together with its standard error $\hat{\sigma}_s$, obtained by applying the [Prediction-Powered Inference (PPI++)](estimators.md#prediction-powered-inference-ppi) variance formula within batch $s$, and $\lambda_s$ a predictable power-tuning weight defined further below.

The weight $\lambda_s$ is a variance-minimizing regression slope which improves the final estimate's precision.

#### Predictable Power-Tuning

The power-tuning weight $\lambda_t$ used to form the estimate $\hat{R}_t$ on batch $t$ must itself be **predictable**: computed only from batches strictly earlier than $t$.

To compute predictable weights $\lambda_t$, the simplest option is to pool the full prior history of batches. The first batch has no predecessor, so it can use the neutral weight $\lambda_1 = 1$, which is itself trivially predictable.

#### Bound and Alarm Rule

Plugging $\hat{R}_s$ and $\hat{\sigma}_s$ into the [Asymptotic Confidence Sequences](#asymptotic-confidence-sequences) construction gives the anytime-valid lower bound $L_t$ derived there. The **drift alarm** fires as soon as $L_t > \tau$, the user-fixed threshold.

---

## References

<a id="ref-1"></a>[1] <a id="ref-1-link" href="https://arxiv.org/abs/2602.02229">Zhang, Guangyi, Yunlong Cai, Guanding Yu, and Osvaldo Simeone. "Prediction-Powered Risk Monitoring of Deployed Models for Detecting Harmful Distribution Shifts." arXiv preprint arXiv:2602.02229 (2026)</a>.

<a id="ref-2"></a>[2] <a id="ref-2-link" href="https://doi.org/10.1214/24-AOS2408">Waudby-Smith, Ian, David Arbour, Ritwik Sinha, Edward H. Kennedy, and Aaditya Ramdas. "Time-uniform central limit theory and asymptotic confidence sequences." The Annals of Statistics 52, no. 6 (2024): 2613-2640</a>.

<a id="ref-3"></a>[3] <a id="ref-3-link" href="https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-41/issue-5/Statistical-Methods-Related-to-the-Law-of-the-Iterated-Logarithm/10.1214/aoms/1177696786.full">Robbins, Herbert. "Statistical methods related to the law of the iterated logarithm." The Annals of Mathematical Statistics 41, no. 5 (1970): 1397-1409</a>.
