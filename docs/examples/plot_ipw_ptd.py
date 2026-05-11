"""
IPWPTDMeanEstimator
=========================================

IPW-PTD extends Predict-Then-Debias to handle non-uniform sampling probabilities
via inverse probability weighting.
"""

##############################################################################
# Create a simulated dataset with 1000 samples, all labeled by a proxy model.
# Then assign non-uniform sampling probabilities and randomly mask labels.

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from glide.estimators import IPWPTDMeanEstimator
from glide.simulators import generate_binary_dataset

pio.renderers.default = "sphinx_gallery"


C_GLIDE = "#E74C3C"
C_TRUTH = "#2C3E50"

TRUE_RATE = 0.10

# Generate a complete dataset with all samples labeled
y_true, y_proxy = generate_binary_dataset(
    n_labeled=1000,
    n_unlabeled=0,
    true_mean=TRUE_RATE,
    proxy_mean=0.08,
    correlation=0.70,
    random_seed=42,
)

##############################################################################
# Assign non-uniform sampling probabilities drawn from [0.5, 1] and randomly
# mask labels based on these probabilities. This ensures pi truly reflects
# the probability each label was observed.

rng = np.random.default_rng(seed=42)
pi = np.clip(rng.random(len(y_true)), 0.5, 1.0)
y_true[rng.random(len(y_true)) > pi] = np.nan

##############################################################################
# Compute the IPW-PTD estimate.

result = IPWPTDMeanEstimator().estimate(y_true, y_proxy, pi, metric_name="Hallucination Rate", random_seed=42)

##############################################################################
# Plot the IPW-PTD estimate with its confidence interval next to the true rate.

mean = result.mean
lo = result.confidence_interval.lower_bound
hi = result.confidence_interval.upper_bound
label = "IPW-PTD (GLIDE)"

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
    title=dict(text="IPW-PTD delivers an unbiased estimate under non-uniform sampling", font=dict(size=14)),
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
