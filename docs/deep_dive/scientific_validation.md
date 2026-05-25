# Scientific Validation Methodology

The validation notebooks in this section verify that GLIDE's estimators are statistically correct. They use three complementary measures: coverage validity, confidence interval width, and effective sample size. This page defines these measures and describes the Monte Carlo protocol used to compute them.

---

## Experimental Setup

Each validation notebook targets the problem of estimating the mean of a binary outcome, such as the hallucination rate of an AI system under evaluation. The setting involves two types of annotations per sample:

- **True labels** ($Y$): ground-truth human annotations, expensive to obtain but unbiased. Only a subset of the full pool is annotated.
- **Proxy labels** ($\tilde{Y}$): cheap, automated predictions (e.g., from an LLM judge), available for all samples but potentially biased.

The **Pearson correlation** between true and proxy labels is swept in each experiment. It controls how informative the proxy is: at zero correlation the proxy provides no useful signal, while as correlation approaches one the proxy closely tracks the true labels.

In all notebooks, the proxy mean differs from the true mean, therefore, naive use of proxy labels yields invalid inference. It is important to verify that the method under evaluation correctly corrects for that bias.

---

## Estimation Methods

Each notebook compares three estimation approaches across a range of proxy-true correlation levels.

**True only** estimates the mean using true labels via a classical CLT-based confidence interval. This method is always valid and serves as the correctness baseline, but it ignores the proxy entirely.

**Proxy only** estimates the mean using only the proxy labels without bias correction. Because the proxy is biased, this method fails the coverage check and is included to demonstrate that ignoring bias produces invalid inference.

**Method under test** (e.g. PPI++, ASI, ...) combines true and proxy labels and corrects for proxy bias. The goal is to preserve the coverage validity of True only while producing shorter confidence intervals when the proxy is sufficiently informative.

---

## Coverage Validity

A confidence interval $C_{1-\alpha}$ for an estimand $\theta^*$ is **valid at level** $1 - \alpha$ if $\theta^*$ falls inside the interval with probability at least $1 - \alpha$ over repeated experiments:

$$\Pr(\theta^* \in C_{1-\alpha}) \geq 1 - \alpha$$

Coverage validity is the minimum requirement for a confidence interval to be useful. An interval that is narrow but invalid provides false precision and cannot be trusted.

### Monte Carlo verification protocol

Coverage is estimated empirically using the following protocol. For a fixed experimental setting and a target confidence level $1 - \alpha$:

1. For each repetition $s = 1, \dots, N$, using a different random seed:
    - Generate a fresh dataset.
    - Compute a confidence interval $C_{1-\alpha}^{(s)}$ for each estimation method.
    - Record a binary hit $\mathbf{1}_{\{\theta^* \in C_{1-\alpha}^{(s)}\}}$.
2. Estimate the empirical coverage as the mean of the hit indicators:

    $$\hat{p} = \frac{1}{N} \sum_{s=1}^{N} \mathbf{1}_{\left\{\theta^* \in C_{1-\alpha}^{(s)}\right\}}$$

3. Compute a CLT-based confidence interval on $\hat{p}$ at the 90% confidence level by computing the mean and standard deviation of the binary hit indicators.

An estimator passes the coverage check if $1 - \alpha$ falls within or below the confidence interval on $\hat{p}$, meaning the data are consistent with true coverage meeting the nominal level.

---

## Confidence Interval Width

For a valid estimator, shorter intervals are better: the same dataset yields more precise estimates. The width of a confidence interval $C_{1-\alpha} = [\ell, u]$ is:

$$w = u - \ell$$

Across Monte Carlo repetitions, the mean width and a percentile band are reported to characterize both average efficiency and variability. The band spans from the $\lfloor \alpha / 2 \cdot 100 \rfloor$ to the $\lceil (1 - \alpha / 2) \cdot 100 \rceil$ percentile of the per-seed width distribution.

---

## Effective Sample Size

The **effective sample size (ESS)** summarizes the efficiency gain of a method relative to using true labels only. It is the number of true labels that a classical estimator would need to match the mean confidence interval width achieved by the method under evaluation.

Since confidence interval width scales as $\propto 1/\sqrt{n}$, the ESS is:

$$\text{ESS} = n_{\text{true}} \times \left(\frac{\bar{w}_{\text{True only}}}{\bar{w}_{\text{method}}}\right)^2$$

where $\bar{w}_{\text{True only}}$ and $\bar{w}_{\text{method}}$ are the mean widths over Monte Carlo seeds and $n_{\text{true}}$ is the number of true labels used.


When the proxy is uninformative (correlation close to zero), ESS $\approx n_{\text{true}}$ and the method provides no gain over a classical estimator. As the correlation grows toward one, ESS rises above $n_{\text{true}}$: the method extracts enough information from the proxy labels to match the precision that a larger true-label dataset would provide.