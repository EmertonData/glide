"""
StratifiedPTDMeanEstimator
=========================================

Stratified PTD extends Predict-Then-Debias to datasets partitioned into strata.
It uses a different power-tuning parameter per stratum, yielding narrower confidence
intervals when strata differ in proxy quality.
"""

##############################################################################
# Create a simulated dataset with two strata: high-quality and lower-quality
# proxy. Each stratum has 1100 samples, 100 human-labeled
# and 1000 proxy-labeled. The ground-truth hallucination rate is 10% in both
# strata.

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from glide.estimators.stratified_ptd import StratifiedPTDMeanEstimator
from glide.simulators import generate_binary_dataset

pio.renderers.default = "sphinx_gallery"


C_GLIDE = "#3498DB"
C_TRUTH = "#2C3E50"

TRUE_RATE = 0.10

y_true_a, y_proxy_a = generate_binary_dataset(
    n_labeled=100,
    n_unlabeled=1000,
    true_mean=TRUE_RATE,
    proxy_mean=0.08,
    correlation=0.75,
    random_seed=60,
)

y_true_b, y_proxy_b = generate_binary_dataset(
    n_labeled=100,
    n_unlabeled=1000,
    true_mean=TRUE_RATE,
    proxy_mean=0.12,
    correlation=0.55,
    random_seed=78,
)

y_true = np.hstack([y_true_a, y_true_b])
y_proxy = np.hstack([y_proxy_a, y_proxy_b])
groups = np.hstack([np.full(len(y_true_a), "A"), np.full(len(y_true_b), "B")])

##############################################################################
# Compute the Stratified PTD estimate.

result = StratifiedPTDMeanEstimator().estimate(
    y_true, y_proxy, groups, metric_name="Hallucination Rate", random_seed=42
)

##############################################################################
# Plot the Stratified PTD estimate with its confidence interval next to the true rate.

mean = result.mean
lo = result.confidence_interval.lower_bound
hi = result.confidence_interval.upper_bound
label = "Stratified PTD (GLIDE)"

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=[label],
        y=[mean],
        marker_color=[C_GLIDE],
        width=0.2,
        error_y=dict(
            type="data",
            symmetric=False,
            array=[hi - mean],
            arrayminus=[mean - lo],
            color="black",
            thickness=2,
            width=8,
        ),
        showlegend=False,
    )
)

fig.add_annotation(
    x=label,
    y=hi,
    text=f"{mean:.1%}<br>[{lo:.1%}, {hi:.1%}]",
    showarrow=False,
    xanchor="left",
    xshift=20,
    font=dict(size=13, color=C_GLIDE),
)

fig.add_shape(
    type="line",
    x0=-0.5,
    x1=0.5,
    y0=TRUE_RATE,
    y1=TRUE_RATE,
    line=dict(color=C_TRUTH, width=2, dash="dash"),
)

fig.add_annotation(
    x=0.5,
    y=TRUE_RATE,
    xref="x",
    yref="y",
    text="True rate",
    showarrow=False,
    xanchor="left",
    xshift=8,
    font=dict(size=12, color=C_TRUTH),
)

fig.update_layout(
    title=dict(text="Stratified PTD delivers an unbiased, precise estimate", font=dict(size=14)),
    yaxis=dict(
        title="Hallucination Rate",
        tickformat=".0%",
        range=[0, max(hi, TRUE_RATE) * 1.6],
        gridcolor="#E5E5E5",
    ),
    plot_bgcolor="#FAFAFA",
    paper_bgcolor="white",
    width=600,
    height=400,
)

fig
