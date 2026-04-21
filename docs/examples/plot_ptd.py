"""
PTDMeanEstimator
=========================================

PTD combines a small set of expensive true evaluation labels with a large pool
of cheap proxy evaluation labels to produce a statistically valid, bias-corrected
quality metric.
"""

##############################################################################
# Create a simulated dataset with 2200 conversations, all of which are labeled
# by an LLM-judge and a subset of 200 samples are human-labeled. The
# ground-truth hallucination rate is 10%.

import plotly.graph_objects as go
import plotly.io as pio

from glide.estimators import PTDMeanEstimator
from glide.simulators import generate_binary_dataset

pio.renderers.default = "sphinx_gallery"


C_GLIDE = "#27AE60"
C_TRUTH = "#2C3E50"

TRUE_RATE = 0.10

y_true, y_proxy = generate_binary_dataset(
    n_labeled=200,  # human labeled samples
    n_unlabeled=2000,  # samples labeled only by the LLM-judge
    true_mean=TRUE_RATE,
    proxy_mean=0.05,
    correlation=0.65,
    random_seed=42,
)

##############################################################################
# Compute the PTD estimate, which corrects for proxy bias by combining both
# label sources.

ptd_result = PTDMeanEstimator().estimate(y_true, y_proxy, metric_name="Hallucination Rate")

##############################################################################
# Plot the PTD estimate with its confidence interval next to the true rate.

mean = ptd_result.mean
lo = ptd_result.confidence_interval.lower_bound
hi = ptd_result.confidence_interval.upper_bound
ptd_label = "PTD (GLIDE)"

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=[ptd_label],
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
    x=ptd_label,
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
    title=dict(text="PTD delivers an unbiased, precise estimate", font=dict(size=14)),
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
