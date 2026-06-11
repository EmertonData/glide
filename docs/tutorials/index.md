# Overview

This section collects end-to-end tutorials for each combination of sampler and estimator available in GLIDE. Each tutorial walks through a complete path from sampling to annotation to estimation on a simulated dataset, so you can follow along with the use case that best matches your own.

If you are new to GLIDE, read the [Evaluation Workflow](../user_guide/evaluation_workflow.md) first: it explains the three-stage pipeline and the role of each component. The guidance below helps you pick the right sampler and estimator for each stage of that workflow.

## Choosing the right sampler and estimator

The workflow breaks into three sequential phases: **Sampling** (deciding which items to send for human annotation), **Annotation** (collecting human labels), and **Estimation** (computing debiased statistics from the combined human and proxy data). The decision tree below guides you through each phase.

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}}}%%
flowchart TD
    subgraph SAMPLING["<span style='font-size:24px'><b>Sampling</b></span>"]
        direction TB
        Q_COST["Do I have cost estimates of<br/>human and proxy labels?"]
        Q_UNC_L["Do I have uncertainty<br/>estimates on proxy labels?"]
        Q_UNC_R["Do I have uncertainty<br/>estimates on proxy labels?"]
        Q_STRATA["Is my dataset<br/>structured into strata?"]
        Q_CLUSTER["Is my dataset<br/>structured into clusters?"]
        S_COST_OPT["Cost Optimal<br/>Sampler"]
        S_COST_OPT_RAND["Cost Optimal<br/>Random Sampler"]
        S_ACTIVE["Active<br/>Sampler"]
        S_STRAT["Stratified<br/>Sampler"]
        S_CLUSTER["Cluster<br/>Sampler"]
        S_UNIFORM["Uniform<br/>Sampler"]

        Q_COST -- Yes --> Q_UNC_L
        Q_COST -- No --> Q_UNC_R
        Q_UNC_L -- Yes --> S_COST_OPT
        Q_UNC_L -- No --> S_COST_OPT_RAND
        Q_UNC_R -- Yes --> S_ACTIVE
        Q_UNC_R -- No --> Q_STRATA
        Q_STRATA -- Yes --> S_STRAT
        Q_STRATA -- No --> Q_CLUSTER
        Q_CLUSTER -- Yes --> S_CLUSTER
        Q_CLUSTER -- No --> S_UNIFORM
    end

    ANNOTATION["Annotation"]
    S_COST_OPT --> ANNOTATION
    S_COST_OPT_RAND --> ANNOTATION
    S_ACTIVE --> ANNOTATION
    S_STRAT --> ANNOTATION
    S_CLUSTER --> ANNOTATION
    S_UNIFORM --> ANNOTATION

    subgraph ESTIMATION["<span style='font-size:24px'><b>Estimation</b></span>"]
        direction TB
        Q_50_ASI["Do I have more than<br/>50 human labels?"]
        Q_50_STRAT["Do I have more than<br/>50 human labels<br/><i>per stratum</i>?"]
        Q_50_PPI["Do I have more than<br/>50 human labels?"]
        E_ASI["Active Statistical<br/>Inference"]
        E_IPW["IPW Predict-<br/>Then-Debias"]
        E_STRAT_PPI["Stratified<br/>PPI++"]
        E_STRAT_PTD["Stratified Predict-<br/>Then-Debias"]
        E_CLUSTER["Clustered<br/>PPI++"]
        E_PPI["PPI++"]
        E_PTD["Predict-<br/>Then-Debias"]

        Q_50_ASI -- Yes --> E_ASI
        Q_50_ASI -- No --> E_IPW
        Q_50_STRAT -- Yes --> E_STRAT_PPI
        Q_50_STRAT -- No --> E_STRAT_PTD
        Q_50_PPI -- Yes --> E_PPI
        Q_50_PPI -- No --> E_PTD
    end

    ANNOTATION --> Q_50_ASI
    ANNOTATION --> Q_50_STRAT
    ANNOTATION --> E_CLUSTER
    ANNOTATION --> Q_50_PPI

    classDef question fill:#cfe7e8,stroke:#3a8a8c,stroke-width:1px,color:#1a1a1a;
    classDef result fill:#dceccf,stroke:#6aa84f,stroke-width:1px,color:#1a1a1a;
    classDef phase fill:#e2e2e2,stroke:#9a9a9a,stroke-width:1px,color:#1a1a1a,font-weight:bold,font-size:24px;

    class Q_COST,Q_UNC_L,Q_UNC_R,Q_STRATA,Q_CLUSTER,Q_50_ASI,Q_50_STRAT,Q_50_PPI question;
    class S_COST_OPT,S_COST_OPT_RAND,S_ACTIVE,S_STRAT,S_CLUSTER,S_UNIFORM,E_ASI,E_IPW,E_STRAT_PPI,E_STRAT_PTD,E_CLUSTER,E_PPI,E_PTD result;
    class ANNOTATION phase;

    style SAMPLING fill:#f6d8d5,stroke:#c98a84,color:#1a1a1a;
    style ESTIMATION fill:#aecce0,stroke:#5f8bab,color:#1a1a1a;
```

---


## Phase 1: Sampling

Before any human annotation takes place, you must decide how to select the items that will be sent for review. Four questions determine the right sampler.

**1. Do you have cost estimates of human and proxy labels?**
If you know (or can estimate) the cost per human annotation and the cost of obtaining a proxy label, GLIDE can allocate your annotation budget optimally rather than uniformly. This unlocks the cost-aware samplers, which minimise the variance of downstream estimates for a fixed budget.

**2. Do you have uncertainty estimates on proxy labels?**
If your proxy model outputs a confidence score or probability alongside each label, the sampling probabilities can be derived from these scores. Uncertainty-aware sampling concentrates human review on the items where the proxy is least reliable, improving statistical efficiency.

**3. Is your dataset structured into strata?**
If your data partitions naturally into groups (by language, domain, question type, etc.) and you expect the proxy model to behave differently across groups, a stratified sampling strategy ensures that each group is represented in the annotated set.

**4. Is your dataset structured into clusters?**
If your samples are grouped into clusters of correlated items (turns within a conversation, sentences within a paragraph, etc.) that must be annotated together as a unit, a clustered sampling strategy draws whole clusters rather than individual samples and defines the annotation budget in terms of clusters. This keeps the statistical inference valid in the presence of within-cluster correlation.

## Phase 2: Annotation

Once a sample has been selected, human annotators label the chosen items. The size of this annotated set relative to your strata is the key quantity for the estimation phase.

---

## Phase 3: Estimation

With human labels in hand, the final question determines whether the Central Limit Theorem can be safely applied to construct confidence intervals.

**Do you have more than 50 human labels (per stratum for stratified methods)?**
PPI and Active Statistical Inference form the CLT-family, they rely on normal approximations to construct confidence intervals. This approximation requires a sufficient number of labeled samples, typically at least 50 per stratum (or in total for non-stratified methods). Below this threshold, the Predict-Then-Debias bootstrap family of estimators provides a simpler and more conservative alternative that remains valid with fewer annotations.

## Available tutorials

Each tutorial walks through one complete path in the above decision tree:
from sampling to annotation to estimation, on a simulated dataset.
Use the table below to find the tutorial that matches your situation.

| Cost estimates? | Uncertainty scores? | Stratified data? | Clustered data? | Phase 1 sampler | Phase 3 estimator | Tutorial |
|---|---|---|---|---|---|---|
| No | No | No | No | Uniform random | PPI++ | [Standard annotation budget (PPI++)](ppi.ipynb) |
| No | No | Yes | No | Stratified uniform | Stratified PPI++ | [Stratified data (Stratified PPI++)](stratified_ppi.ipynb) |
| No | No | No | Yes | Cluster uniform | Clustered PPI++ | [Clustered data (Clustered PPI++)](cluster_ppi.ipynb) |
| No | Yes | No | No | Uncertainty-aware | ASI | [Uncertainty scores available (ASI)](asi.ipynb) |
| Yes | No | No | No | Cost-optimal random | PPI++ | [Cost estimates available (Cost-Optimal Random Sampling)](cost_optimal_random.ipynb) |
| Yes | Yes | No | No | Cost-optimal | ASI | [Cost and uncertainty scores available (Cost-Optimal Sampling)](cost_optimal.ipynb) |

If your data contains fewer than 50 human labels: use the PTD variant of the estimators above (`PTDMeanEstimator` for PPI++, `StratifiedPTDMeanEstimator` for Stratified PPI++ and `IPWPTDMeanEstimator` for ASI). In the stratified case, the `StratifiedPTDMeanEstimator` should be used whenever one of the strata has fewer than 50 labels. The tutorial workflow for the respective estimators is identical; only the estimator class changes. Clustered PPI++ has no bootstrap variant: the clustered path applies regardless of the number of human labels, with the annotation budget counted in clusters.
