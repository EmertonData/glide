# Scientific Validation Methodology

The validation notebooks in this section verify that GLIDE's estimators are statistically correct. They use three complementary measures: coverage validity, confidence interval width, and effective sample size. This page defines these measures and describes the Monte Carlo protocol used to compute them.

---

## Coverage Validity

A confidence interval is **valid at level** $1 - \alpha$ if the true value $\theta^*$ falls inside the interval with probability at least $1 - \alpha$ over repeated experiments:

$$\Pr(\theta^* \in C_\alpha) \geq 1 - \alpha$$

Coverage validity is the minimum requirement for a confidence interval to be useful. An interval that is narrow but invalid provides false precision and cannot be trusted.

### Monte Carlo verification protocol

Coverage is estimated empirically using the following protocol. For a fixed experimental setting and a target confidence level $1 - \alpha$:

1. Repeat $N_{\text{seeds}}$ times, each with a different random seed:
    - Generate a fresh dataset.
    - Compute a confidence interval $C_\alpha^{(s)}$ for each estimation method.
    - Record a binary hit $\mathbf{1}[\theta^* \in C_\alpha^{(s)}]$.
2. Estimate the empirical coverage as the mean of the hit indicators:

$$\hat{p} = \frac{1}{N_{\text{seeds}}} \sum_{s=1}^{N_{\text{seeds}}} \mathbf{1}\left[\theta^* \in C_\alpha^{(s)}\right]$$

3. Compute a confidence interval on $\hat{p}$ by applying the classical estimator to the binary hit array.

An estimator passes the coverage check if $\hat{p} \geq 1 - \alpha$ (within Monte Carlo noise) across a range of correlation levels and confidence levels.

---

## Confidence Interval Width

For a valid estimator, shorter intervals are better: the same dataset yields more precise estimates. The width of a confidence interval $C_\alpha = [\ell, u]$ is:

$$w = u - \ell$$

Across Monte Carlo repetitions, we report the mean width and a percentile band to characterize both average efficiency and variability. The band spans from the $\lfloor (1 - \alpha) / 2 \cdot 100 \rfloor$th to the $\lceil (1 + \alpha) / 2 \cdot 100 \rceil$th percentile of the per-seed width distribution.

---

## Effective Sample Size

The **effective sample size (ESS)** summarizes the efficiency gain of a method relative to using true labels only. It is the number of true labels that the classical estimator would need to match the mean confidence interval width achieved by the method under evaluation.

Since confidence interval width scales as $\propto 1/\sqrt{n}$, the ESS is:

$$\text{ESS} = n_{\text{true}} \times \left(\frac{\bar{w}_{\text{True only}}}{\bar{w}_{\text{method}}}\right)^2$$

where $\bar{w}_{\text{True only}}$ and $\bar{w}_{\text{method}}$ are the mean widths over Monte Carlo seeds and $n_{\text{true}}$ is the number of true labels used.

When the proxy is uninformative, ESS $\approx n_{\text{true}}$ (no gain). As proxy quality improves, ESS grows above $n_{\text{true}}$, meaning the method extracts as much statistical information as a larger labeled dataset would.
