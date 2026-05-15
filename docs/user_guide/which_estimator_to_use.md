# Which Estimator to Use?

The workflow breaks into three sequential phases: **Sampling** (deciding which items to send for human annotation), **Annotation** (collecting human labels), and **Estimation** (computing debiased statistics from the combined human and proxy data). The decision tree below guides you through each phase.

<p align="center">
  <img src="../../assets/which-estimators-to-use.png" alt="Decision tree for choosing a GLIDE sampler and estimator" width="700">
</p>

---

## Phase 1: Sampling

Before any human annotation takes place, you must decide how to select the items that will be sent for review. Three questions determine the right sampler.

**1. Do you have cost estimates of human and proxy labels?**
If you know (or can estimate) the cost per human annotation and the cost of obtaining a proxy label, GLIDE can allocate your annotation budget optimally rather than uniformly. This unlocks the cost-aware samplers, which minimise the variance of downstream estimates for a fixed budget.

**2. Do you have uncertainty estimates on proxy labels?**
If your proxy model outputs a confidence score or probability alongside each label, the sampling probabilities can be derived from these scores. Uncertainty-aware sampling concentrates human review on the items where the proxy is least reliable, improving statistical efficiency.

**3. Is your dataset structured into strata?**
If your data partitions naturally into groups (by language, domain, question type, etc.) and you expect the proxy model to behave differently across groups, a stratified sampling strategy ensures that each group is represented in the annotated set.

## Phase 2: Annotation

Once a sample has been selected, human annotators label the chosen items. The size of this annotated set relative to your strata is the key quantity for the estimation phase.

---

## Phase 3: Estimation

With human labels in hand, the final question determines whether the Central Limit Theorem can be safely applied to construct confidence intervals.

**Do you have more than 50 human labels (per stratum for stratified methods)?**
PPI and Active Statistical Inference form the CLT-family, they rely on normal approximations to construct confidence intervals. This approximation requires a sufficient number of labeled samples, typically at least 50 per stratum (or in total for non-stratified methods). Below this threshold, the Predict-Then-Debias bootstrap family of estimators provides a simpler and more conservative alternative that remains valid with fewer annotations.