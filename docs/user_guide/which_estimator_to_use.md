# Which Estimator to Use?

Three questions determine the right sampler and estimator for your evaluation setup.

<p align="center">
  <img src="../../assets/which-estimators-to-use.png" alt="Decision tree for choosing a GLIDE estimator" width="1000">
</p>

**1. Do you have uncertainty estimates on proxy labels?**
If your proxy model outputs a confidence score or probability alongside each label, the sampling probabilities $\pi_i$ can be derived from these scores and fed into the [Active Sampler](samplers.md). This unlocks **Active Statistical Inference (ASI)**, which uses Inverse Probability Weighting to correct for the non-uniform selection of human annotations.

**2. Is your dataset structured into strata?**
If your data naturally partitions into groups (by language, domain, question type, etc.) and you expect the proxy model to behave differently across groups, use the [Stratified Sampler](samplers.md) and **Stratified PPI++**. This runs PPI++ independently within each stratum and combines the results with population-proportional weights, isolating proxy quality differences across strata.

**3. Do you have more than 50 human annotations per stratum (or in total)?**
PPI++, Stratified PPI++, and Active Statistical Inference all rely on the Central Limit Theorem to construct confidence intervals, which requires a sufficient number of labeled samples (typically $n \geq 50$ per stratum). Below this threshold, the CLT approximation may not hold and the confidence intervals may be unreliable. In that case, **Predict-Then-Debias** variants offer a simpler, more conservative alternative that remains valid with fewer annotations.