"""
StratifiedPTDMeanEstimator
=========================================

Stratified PTD extends Predict-Then-Debias to datasets partitioned into strata
(e.g., by language, domain, or data source). It computes a per-stratum power-tuning
parameter independently and combines them for narrower confidence intervals when
strata differ in proxy quality.
"""

##############################################################################
# Create a simulated dataset with two strata: one with high-quality proxy labels
# and another with lower-quality proxy labels. Each stratum has 1100 samples,
# 100 human-labeled and 1000 proxy-labeled. The ground-truth hallucination rate
# is 10% in both strata.

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from glide.estimators import PTDMeanEstimator
from glide.estimators.stratified_ptd import StratifiedPTDMeanEstimator
from glide.simulators import generate_binary_dataset

pio.renderers.default = "sphinx_gallery"


C_STRATIFIED = "#3498DB"
C_STANDARD = "#E74C3C"
C_TRUTH = "#2C3E50"

TRUE_RATE = 0.10

# Stratum A: high-quality proxy (correlation 0.75)
y_true_a, y_proxy_a = generate_binary_dataset(
    n_labeled=100,
    n_unlabeled=1000,
    true_mean=TRUE_RATE,
    proxy_mean=0.08,
    correlation=0.75,
    random_seed=60,
)

# Stratum B: lower-quality proxy (correlation 0.55)
y_true_b, y_proxy_b = generate_binary_dataset(
    n_labeled=100,
    n_unlabeled=1000,
    true_mean=TRUE_RATE,
    proxy_mean=0.12,
    correlation=0.55,
    random_seed=78,
)

##############################################################################
# Combine the two strata into a single dataset with group labels.

y_true_combined = np.hstack([y_true_a, y_true_b])
y_proxy_combined = np.hstack([y_proxy_a, y_proxy_b])
groups = np.hstack([np.full(len(y_true_a), "A"), np.full(len(y_true_b), "B")])

##############################################################################
# Estimate using both standard PTD (which ignores stratification) and
# Stratified PTD (which adapts lambda per stratum).

ptd_result = PTDMeanEstimator().estimate(
    y_true_combined, y_proxy_combined, metric_name="Hallucination Rate", random_seed=42
)

stratified_ptd_result = StratifiedPTDMeanEstimator().estimate(
    y_true_combined, y_proxy_combined, groups, metric_name="Hallucination Rate", random_seed=42
)

##############################################################################
# Plot both estimates with their confidence intervals next to the true rate.

mean_std = ptd_result.mean
lo_std = ptd_result.confidence_interval.lower_bound
hi_std = ptd_result.confidence_interval.upper_bound

mean_strat = stratified_ptd_result.mean
lo_strat = stratified_ptd_result.confidence_interval.lower_bound
hi_strat = stratified_ptd_result.confidence_interval.upper_bound

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=["Standard PTD"],
        y=[mean_std],
        marker_color=[C_STANDARD],
        width=0.3,
        error_y=dict(
            type="data",
            symmetric=False,
            array=[hi_std - mean_std],
            arrayminus=[mean_std - lo_std],
            color="black",
            thickness=2,
            width=6,
        ),
        showlegend=False,
    )
)

fig.add_trace(
    go.Bar(
        x=["Stratified PTD"],
        y=[mean_strat],
        marker_color=[C_STRATIFIED],
        width=0.3,
        error_y=dict(
            type="data",
            symmetric=False,
            array=[hi_strat - mean_strat],
            arrayminus=[mean_strat - lo_strat],
            color="black",
            thickness=2,
            width=6,
        ),
        showlegend=False,
    )
)

fig.add_annotation(
    x="Standard PTD",
    y=hi_std,
    text=f"{mean_std:.1%}<br>[{lo_std:.1%}, {hi_std:.1%}]",
    showarrow=False,
    xanchor="left",
    xshift=20,
    font=dict(size=12, color=C_STANDARD),
)

fig.add_annotation(
    x="Stratified PTD",
    y=hi_strat,
    text=f"{mean_strat:.1%}<br>[{lo_strat:.1%}, {hi_strat:.1%}]",
    showarrow=False,
    xanchor="left",
    xshift=20,
    font=dict(size=12, color=C_STRATIFIED),
)

fig.add_shape(
    type="line",
    x0=-0.5,
    x1=1.5,
    y0=TRUE_RATE,
    y1=TRUE_RATE,
    line=dict(color=C_TRUTH, width=2, dash="dash"),
)

fig.add_annotation(
    x=1.5,
    y=TRUE_RATE,
    xref="x",
    yref="y",
    text="True rate",
    showarrow=False,
    xanchor="left",
    xshift=8,
    font=dict(size=11, color=C_TRUTH),
)

fig.update_layout(
    title=dict(
        text="Stratified PTD yields narrower CIs on heterogeneous strata",
        font=dict(size=14),
    ),
    yaxis=dict(
        title="Hallucination Rate",
        tickformat=".0%",
        range=[0, max(hi_std, hi_strat) * 1.5],
        gridcolor="#E5E5E5",
    ),
    plot_bgcolor="#FAFAFA",
    paper_bgcolor="white",
    width=650,
    height=400,
)

fig
