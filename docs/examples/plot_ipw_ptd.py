"""
IPWPTDMeanEstimator
=========================================

IPW-PTD extends Predict-Then-Debias to handle non-uniform sampling probabilities.
It uses inverse probability weighting to correct for biased ground-truth labeling,
yielding valid estimates even when the labeling process is selective.
"""

##############################################################################
# Create a simulated dataset with 240 samples, all labeled by a proxy model
# and a subset labeled by humans with non-uniform probabilities.
# The ground-truth hallucination rate is 12%.

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from glide.estimators import IPWPTDMeanEstimator
from glide.simulators import generate_binary_dataset

pio.renderers.default = "sphinx_gallery"


C_GLIDE = "#E74C3C"
C_TRUTH = "#2C3E50"

TRUE_RATE = 0.12
N_TOTAL = 240
N_LABELED = 60

y_true_all, y_proxy_all = generate_binary_dataset(
    n_labeled=N_LABELED,
    n_unlabeled=N_TOTAL - N_LABELED,
    true_mean=TRUE_RATE,
    proxy_mean=0.10,
    correlation=0.68,
    random_seed=42,
)

##############################################################################
# Assign non-uniform sampling probabilities. The first 60 samples (with higher
# proxy quality) have a higher probability of being labeled. This creates
# selection bias that IPW-PTD corrects for via inverse probability weighting.

pi = np.zeros(N_TOTAL)
pi[:N_LABELED] = 0.60  # Higher prob for higher-quality samples
pi[N_LABELED:] = 0.20  # Lower prob for lower-quality samples

##############################################################################
# Select which samples are actually labeled. Labeled samples get their true
# values; unlabeled samples are marked with NaN.

y_true = np.full(N_TOTAL, np.nan)
y_true[:N_LABELED] = y_true_all[:N_LABELED]

##############################################################################
# Compute the IPW-PTD estimate. By incorporating sampling probabilities π,
# IPW-PTD corrects for selection bias and produces a valid confidence interval
# even when labeling is non-uniform.

ipw_ptd_result = IPWPTDMeanEstimator().estimate(
    y_true, y_proxy_all, pi, metric_name="Hallucination Rate", random_seed=42
)

##############################################################################
# Plot the IPW-PTD estimate with its confidence interval next to the true rate.

mean = ipw_ptd_result.mean
lo = ipw_ptd_result.confidence_interval.lower_bound
hi = ipw_ptd_result.confidence_interval.upper_bound
ipw_ptd_label = "IPW-PTD (GLIDE)"

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=[ipw_ptd_label],
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
    x=ipw_ptd_label,
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
    title=dict(text="IPW-PTD corrects for biased sampling", font=dict(size=14)),
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
